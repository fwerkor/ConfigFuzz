from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml

from configfuzz.model import (
    Constraint,
    ConstraintKind,
    ConstraintSet,
    Evidence,
    EvidenceKind,
)


_TYPE_ALIASES = {
    "bool": "boolean",
    "boolean": "boolean",
    "float": "float",
    "int": "integer",
    "integer": "integer",
    "list": "list",
    "number": "float",
    "str": "string",
    "string": "string",
}

_SCHEMA_KEYS = {
    "choices",
    "enum",
    "exclusiveMaximum",
    "exclusiveMinimum",
    "ge",
    "gt",
    "le",
    "lt",
    "max",
    "max_val",
    "maximum",
    "min",
    "min_val",
    "minimum",
    "multiple_of",
    "required",
    "type",
}


class PythonDeclarationExtractor(ast.NodeVisitor):
    """Extract constraints declared by argparse and Python schema classes."""

    def __init__(self, parameters: Iterable[str], source: str):
        self.source = source
        self.parameters = list(dict.fromkeys(str(item) for item in parameters))
        self._by_leaf = {
            parameter.rsplit(".", 1)[-1].replace("-", "_"): parameter
            for parameter in self.parameters
        }
        self.results = {
            parameter: ConstraintSet(
                parameter=parameter,
                metadata={"extractor": "declarations", "source": source},
            )
            for parameter in self.parameters
        }
        self._schema_class_depth = 0

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            self._extract_argparse(node)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if not _is_schema_class(node):
            self.generic_visit(node)
            return
        self._schema_class_depth += 1
        try:
            for statement in node.body:
                self.visit(statement)
        finally:
            self._schema_class_depth -= 1

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if self._schema_class_depth <= 0 or not isinstance(node.target, ast.Name):
            return
        parameter = self._by_leaf.get(node.target.id)
        if parameter is None:
            return
        type_name, choices, nullable = _annotation_spec(node.annotation)
        if type_name is not None:
            self._add(
                parameter,
                f"{parameter}: {type_name}",
                ConstraintKind.TYPE,
                node.lineno,
                "schema annotation",
            )
        if choices:
            self._add_enum(parameter, choices, node.lineno, "schema Literal annotation")
        required = node.value is None or (
            isinstance(node.value, ast.Call) and _field_call_is_required(node.value)
        )
        if not nullable and required:
            self._add(
                parameter,
                f"{parameter} is not None",
                ConstraintKind.TYPE,
                node.lineno,
                "required schema field",
            )
        if isinstance(node.value, ast.Call):
            self._extract_field_call(parameter, node.value, node.lineno)

    def _extract_argparse(self, node: ast.Call) -> None:
        option_strings = [
            value
            for argument in node.args
            if (value := _literal_string(argument)) is not None
        ]
        keywords = {item.arg: item.value for item in node.keywords if item.arg is not None}
        dest = _literal_string(keywords.get("dest"))
        if dest is None:
            long_options = [item for item in option_strings if item.startswith("--")]
            if not long_options:
                return
            dest = max(long_options, key=len).removeprefix("--").replace("-", "_")
        parameter = self._by_leaf.get(dest.replace("-", "_"))
        if parameter is None:
            return

        action = _literal_string(keywords.get("action"))
        nargs = _literal_string(keywords.get("nargs"))
        type_name = None
        if action in {"store_true", "store_false"}:
            type_name = "boolean"
        elif nargs in {"+", "*"}:
            type_name = "list"
        elif "type" in keywords:
            type_name = _annotation_type(keywords["type"])
        if type_name is not None:
            self._add(
                parameter,
                f"{parameter}: {type_name}",
                ConstraintKind.TYPE,
                node.lineno,
                "argparse declaration",
            )

        choices = _literal_collection(keywords.get("choices"))
        if choices:
            self._add_enum(parameter, choices, node.lineno, "argparse choices")

        if _literal_bool(keywords.get("required")) is True:
            self._add(
                parameter,
                f"{parameter} is not None",
                ConstraintKind.TYPE,
                node.lineno,
                "required argparse option",
            )

    def _extract_field_call(self, parameter: str, call: ast.Call, line: int) -> None:
        name = _call_name(call.func)
        if name not in {"Field", "field"}:
            return
        keywords = {item.arg: item.value for item in call.keywords if item.arg is not None}
        metadata = _literal_mapping(keywords.get("metadata"))
        for key in ("ge", "gt", "le", "lt", "multiple_of"):
            if key in keywords:
                metadata[key] = _literal_value(keywords[key])
        self._add_schema_constraints(parameter, metadata, line, f"{name} declaration")

    def _add_schema_constraints(
        self,
        parameter: str,
        spec: dict[str, Any],
        line: int | None,
        detail: str,
    ) -> None:
        type_name = _normalize_type(spec.get("type"))
        if type_name is not None:
            self._add(parameter, f"{parameter}: {type_name}", ConstraintKind.TYPE, line, detail)

        choices = spec.get("enum", spec.get("choices"))
        if isinstance(choices, (list, tuple, set)) and choices:
            self._add_enum(parameter, list(choices), line, detail)

        bounds = [
            ("minimum", ">="),
            ("min", ">="),
            ("min_val", ">="),
            ("ge", ">="),
            ("exclusiveMinimum", ">"),
            ("gt", ">"),
            ("maximum", "<="),
            ("max", "<="),
            ("max_val", "<="),
            ("le", "<="),
            ("exclusiveMaximum", "<"),
            ("lt", "<"),
        ]
        emitted: set[tuple[str, Any]] = set()
        for key, operator in bounds:
            value = spec.get(key)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            marker = (operator, value)
            if marker in emitted:
                continue
            emitted.add(marker)
            self._add(
                parameter,
                f"{parameter} {operator} {_render_literal(value)}",
                ConstraintKind.RANGE,
                line,
                detail,
            )

        multiple = spec.get("multiple_of")
        if isinstance(multiple, int) and not isinstance(multiple, bool) and multiple > 0:
            self._add(
                parameter,
                f"{parameter} % {multiple} == 0",
                ConstraintKind.RELATION,
                line,
                detail,
            )

        if spec.get("required") is True:
            self._add(
                parameter,
                f"{parameter} is not None",
                ConstraintKind.TYPE,
                line,
                detail,
            )

    def _add_enum(
        self,
        parameter: str,
        values: Iterable[Any],
        line: int | None,
        detail: str,
    ) -> None:
        rendered = sorted({_render_literal(value) for value in values})
        if not rendered:
            return
        self._add(
            parameter,
            f"{parameter} in {{{', '.join(rendered)}}}",
            ConstraintKind.ENUM,
            line,
            detail,
        )

    def _add(
        self,
        parameter: str,
        expression: str,
        kind: ConstraintKind,
        line: int | None,
        detail: str,
    ) -> None:
        self.results[parameter].add(
            Constraint(
                expression=expression,
                kind=kind,
                parameters=(parameter,),
                confidence=1.0,
                evidence=(
                    Evidence(
                        kind=EvidenceKind.STATIC,
                        source=self.source,
                        line=line,
                        detail=detail,
                    ),
                ),
            )
        )


def scan_declaration_paths_multi(
    paths: Iterable[Path],
    parameters: Iterable[str],
) -> dict[str, ConstraintSet]:
    ordered_parameters = list(dict.fromkeys(str(item) for item in parameters))
    merged = {
        parameter: ConstraintSet(
            parameter=parameter,
            metadata={
                "extractor": "declarations",
                "python_files": 0,
                "yaml_files": 0,
                "json_files": 0,
                "parse_errors": [],
            },
        )
        for parameter in ordered_parameters
    }

    for path in _iter_source_files(paths):
        if path.suffix == ".py":
            _scan_python_declarations(path, ordered_parameters, merged)
        elif path.suffix == ".json":
            _scan_json_declarations(path, ordered_parameters, merged)
        else:
            _scan_yaml_declarations(path, ordered_parameters, merged)

    for result in merged.values():
        result.metadata["accepted_candidates"] = len(result.constraints)
    return merged


def _scan_python_declarations(
    path: Path,
    parameters: list[str],
    merged: dict[str, ConstraintSet],
) -> None:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
    except (OSError, SyntaxError) as exc:
        error = {"source": str(path), "error": str(exc)}
        for result in merged.values():
            result.metadata["parse_errors"].append(error)
        return
    source = _display_path(path)
    extractor = PythonDeclarationExtractor(parameters, source)
    extractor.visit(tree)
    for parameter, result in extractor.results.items():
        merged[parameter].metadata["python_files"] += 1
        merged[parameter].extend(result.constraints)


def _scan_yaml_declarations(
    path: Path,
    parameters: list[str],
    merged: dict[str, ConstraintSet],
) -> None:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, yaml.YAMLError) as exc:
        error = {"source": str(path), "error": str(exc)}
        for result in merged.values():
            result.metadata["parse_errors"].append(error)
        return
    if not isinstance(data, dict):
        return
    source = _display_path(path)
    extractor = PythonDeclarationExtractor(parameters, source)
    for parameter in parameters:
        leaf = parameter.rsplit(".", 1)[-1]
        for spec in _yaml_specs_for_parameter(data, leaf):
            extractor._add_schema_constraints(parameter, spec, None, "YAML schema declaration")
        enum_values = _yaml_enum_pool(data, leaf)
        if enum_values is not None:
            extractor._add_enum(parameter, enum_values, None, "YAML enum parameter pool")
        numeric_spec = _yaml_numeric_pool(data, leaf)
        if numeric_spec is not None:
            extractor._add_schema_constraints(parameter, numeric_spec, None, "YAML numeric parameter pool")
    for parameter, result in extractor.results.items():
        merged[parameter].metadata["yaml_files"] += 1
        merged[parameter].extend(result.constraints)


def _scan_json_declarations(
    path: Path,
    parameters: list[str],
    merged: dict[str, ConstraintSet],
) -> None:
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError) as exc:
        error = {"source": str(path), "error": str(exc)}
        for result in merged.values():
            result.metadata["parse_errors"].append(error)
        return
    if not isinstance(data, dict):
        return
    source = _display_path(path)
    extractor = PythonDeclarationExtractor(parameters, source)
    for parameter in parameters:
        leaf = parameter.rsplit(".", 1)[-1]
        for spec in _yaml_specs_for_parameter(data, leaf):
            extractor._add_schema_constraints(parameter, spec, None, "JSON schema declaration")
    for parameter, result in extractor.results.items():
        merged[parameter].metadata["json_files"] += 1
        merged[parameter].extend(result.constraints)


def _yaml_specs_for_parameter(data: dict[str, Any], leaf: str) -> Iterator[dict[str, Any]]:
    stack: list[Any] = [data]
    seen: set[int] = set()
    while stack:
        current = stack.pop()
        if not isinstance(current, dict) or id(current) in seen:
            continue
        seen.add(id(current))
        candidate = current.get(leaf)
        if isinstance(candidate, dict) and _SCHEMA_KEYS.intersection(candidate):
            yield candidate
        properties = current.get("properties")
        if isinstance(properties, dict):
            candidate = properties.get(leaf)
            if isinstance(candidate, dict) and _SCHEMA_KEYS.intersection(candidate):
                yield candidate
        stack.extend(value for value in current.values() if isinstance(value, dict))


def _yaml_enum_pool(data: dict[str, Any], leaf: str) -> list[Any] | None:
    section = data.get("enum_params")
    if not isinstance(section, dict):
        return None
    values = section.get(leaf)
    return list(values) if isinstance(values, (list, tuple, set)) else None


def _yaml_numeric_pool(data: dict[str, Any], leaf: str) -> dict[str, Any] | None:
    section = data.get("numeric_params")
    if not isinstance(section, dict):
        return None
    spec = section.get(leaf)
    return dict(spec) if isinstance(spec, dict) else None


def _iter_source_files(paths: Iterable[Path]) -> Iterator[Path]:
    ignored = {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "site-packages",
    }
    seen: set[Path] = set()
    for raw in paths:
        path = raw.resolve()
        if path.is_file() and path.suffix.lower() in {".py", ".yaml", ".yml", ".json"}:
            if path not in seen:
                seen.add(path)
                yield path
            continue
        if not path.is_dir():
            continue
        for suffix in ("*.py", "*.yaml", "*.yml", "*.json"):
            for candidate in sorted(path.rglob(suffix)):
                if any(part in ignored for part in candidate.parts) or candidate in seen:
                    continue
                seen.add(candidate)
                yield candidate


def _annotation_spec(node: ast.expr) -> tuple[str | None, list[Any], bool]:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        left_type, left_choices, left_nullable = _annotation_spec(node.left)
        right_type, right_choices, right_nullable = _annotation_spec(node.right)
        nullable = left_nullable or right_nullable or _is_none_annotation(node.left) or _is_none_annotation(node.right)
        return left_type or right_type, [*left_choices, *right_choices], nullable

    name = _annotation_name(node)
    if name in {"Optional", "typing.Optional"} and isinstance(node, ast.Subscript):
        type_name, choices, _ = _annotation_spec(node.slice)
        return type_name, choices, True
    if name in {"Union", "typing.Union"} and isinstance(node, ast.Subscript):
        items = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        type_name = None
        choices: list[Any] = []
        nullable = False
        for item in items:
            if _is_none_annotation(item):
                nullable = True
                continue
            item_type, item_choices, item_nullable = _annotation_spec(item)
            type_name = type_name or item_type
            choices.extend(item_choices)
            nullable = nullable or item_nullable
        return type_name, choices, nullable
    if name in {"Literal", "typing.Literal"} and isinstance(node, ast.Subscript):
        items = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        return None, [value for item in items if (value := _literal_value(item)) is not None], False
    if name in {"List", "Sequence", "list", "typing.List", "typing.Sequence"}:
        return "list", [], False
    return _normalize_type(name), [], False


def _annotation_type(node: ast.expr) -> str | None:
    name = _annotation_name(node)
    return _normalize_type(name)


def _annotation_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _annotation_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    if isinstance(node, ast.Subscript):
        return _annotation_name(node.value)
    return None


def _normalize_type(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return _TYPE_ALIASES.get(value.rsplit(".", 1)[-1].lower())


def _decorator_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Call):
        node = node.func
    return _annotation_name(node)


def _is_schema_class(node: ast.ClassDef) -> bool:
    if any((_decorator_name(item) or "").rsplit(".", 1)[-1] == "dataclass" for item in node.decorator_list):
        return True
    schema_bases = {
        "BaseModel",
        "ConfigModel",
        "DeepSpeedConfigModel",
    }
    return any(
        (_annotation_name(base) or "").rsplit(".", 1)[-1] in schema_bases
        for base in node.bases
    )


def _call_name(node: ast.expr) -> str | None:
    return _annotation_name(node)


def _field_call_is_required(call: ast.Call) -> bool:
    if _call_name(call.func) not in {"Field", "field"}:
        return False
    keywords = {item.arg for item in call.keywords if item.arg is not None}
    if "default" in keywords or "default_factory" in keywords:
        return False
    return not call.args


def _literal_collection(node: ast.expr | None) -> list[Any]:
    if not isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return []
    values: list[Any] = []
    for item in node.elts:
        value = _literal_value(item)
        if value is None and not (isinstance(item, ast.Constant) and item.value is None):
            return []
        values.append(value)
    return values


def _literal_mapping(node: ast.expr | None) -> dict[str, Any]:
    if not isinstance(node, ast.Dict):
        return {}
    result: dict[str, Any] = {}
    for key_node, value_node in zip(node.keys, node.values):
        key = _literal_string(key_node)
        if key is None:
            continue
        result[key] = _literal_value(value_node)
    return result


def _literal_value(node: ast.expr | None) -> Any:
    if node is None:
        return None
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        return None


def _literal_string(node: ast.expr | None) -> str | None:
    value = _literal_value(node)
    return value if isinstance(value, str) else None


def _literal_bool(node: ast.expr | None) -> bool | None:
    value = _literal_value(node)
    return value if isinstance(value, bool) else None


def _is_none_annotation(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and node.value is None or (
        isinstance(node, ast.Name) and node.id in {"None", "NoneType"}
    )


def _render_literal(value: Any) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return "null"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    return repr(value)


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)
