from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from configfuzz.model import (
    Constraint,
    ConstraintKind,
    ConstraintSet,
    Evidence,
    EvidenceKind,
)


_INVERSE_COMPARATORS: dict[type[ast.cmpop], type[ast.cmpop]] = {
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
    ast.Lt: ast.GtE,
    ast.LtE: ast.Gt,
    ast.Gt: ast.LtE,
    ast.GtE: ast.Lt,
    ast.In: ast.NotIn,
    ast.NotIn: ast.In,
    ast.Is: ast.IsNot,
    ast.IsNot: ast.Is,
}

_IGNORED_NAMES = {
    "False",
    "None",
    "True",
    "abs",
    "bool",
    "float",
    "int",
    "len",
    "list",
    "math",
    "max",
    "min",
    "os",
    "self",
    "set",
    "str",
    "sum",
    "tuple",
}


@dataclass(frozen=True, slots=True)
class ExtractedExpression:
    node: ast.expr
    line: int
    confidence: float
    detail: str


class _CanonicalizeTarget(ast.NodeTransformer):
    def __init__(self, parameter: str, aliases: set[str]):
        self.parameter = parameter
        self.aliases = aliases

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in self.aliases:
            return ast.copy_location(ast.Name(id=self.parameter, ctx=node.ctx), node)
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        if node.attr == self.parameter:
            return ast.copy_location(ast.Name(id=self.parameter, ctx=ast.Load()), node)
        return self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        key = _constant_string(node.slice)
        if key == self.parameter:
            return ast.copy_location(ast.Name(id=self.parameter, ctx=ast.Load()), node)
        return self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> ast.AST:
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and node.args
            and _constant_string(node.args[0]) == self.parameter
        ):
            return ast.copy_location(ast.Name(id=self.parameter, ctx=ast.Load()), node)
        return self.generic_visit(node)


class PythonConstraintExtractor(ast.NodeVisitor):
    """Extract explicit candidate constraints for one parameter from Python source.

    The first prototype intentionally favors high-precision patterns:

    * assertions that reference the parameter;
    * rejecting ``if`` guards whose body raises;
    * lm-sv validator guards whose body calls ``_apply_fix``;
    * aliases assigned from ``config.parameter``, ``config[parameter]``, or
      ``config.get(parameter)``.

    It is not a complete data-flow analysis. Every result is therefore stored as
    evidence-backed candidate information rather than treated as ground truth.
    """

    def __init__(self, parameter: str, source: str):
        self.parameter = parameter
        self.source = source
        self.aliases: set[str] = {parameter}
        self._expressions: list[ExtractedExpression] = []

    def extract(self, tree: ast.AST) -> ConstraintSet:
        self.visit(tree)
        result = ConstraintSet(
            parameter=self.parameter,
            metadata={"extractor": "python_ast", "source": self.source},
        )
        for extracted in self._expressions:
            canonical = _CanonicalizeTarget(self.parameter, self.aliases).visit(
                ast.fix_missing_locations(extracted.node)
            )
            ast.fix_missing_locations(canonical)
            expression = ast.unparse(canonical)
            parameters = _parameter_names(canonical, self.parameter)
            if self.parameter not in parameters:
                continue
            result.add(
                Constraint(
                    expression=expression,
                    kind=_classify(canonical, parameters),
                    parameters=tuple(parameters),
                    confidence=extracted.confidence,
                    evidence=(
                        Evidence(
                            kind=EvidenceKind.STATIC,
                            source=self.source,
                            line=extracted.line,
                            detail=extracted.detail,
                        ),
                    ),
                )
            )
        return result

    def visit_Assign(self, node: ast.Assign) -> None:
        if self._contains_target(node.value):
            for target in node.targets:
                self.aliases.update(_assigned_names(target))
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None and self._contains_target(node.value):
            self.aliases.update(_assigned_names(node.target))
        self.generic_visit(node)

    def visit_Assert(self, node: ast.Assert) -> None:
        if self._contains_target(node.test):
            for expression in _split_valid_condition(node.test):
                self._expressions.append(
                    ExtractedExpression(
                        node=expression,
                        line=node.lineno,
                        confidence=1.0,
                        detail="assertion",
                    )
                )
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        if self._contains_target(node.test) and _body_rejects_or_repairs(node.body):
            for expression in _invert_rejecting_condition(node.test):
                self._expressions.append(
                    ExtractedExpression(
                        node=expression,
                        line=node.lineno,
                        confidence=0.95,
                        detail="rejecting or repairing guard",
                    )
                )
        self.generic_visit(node)

    def _contains_target(self, node: ast.AST) -> bool:
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and child.id in self.aliases:
                return True
            if isinstance(child, ast.Attribute) and child.attr == self.parameter:
                return True
            if isinstance(child, ast.Subscript) and _constant_string(child.slice) == self.parameter:
                return True
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and child.func.attr == "get"
                and child.args
                and _constant_string(child.args[0]) == self.parameter
            ):
                return True
        return False


def extract_python_file(path: Path, parameter: str) -> ConstraintSet:
    source_text = path.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(source_text, filename=str(path))
    return PythonConstraintExtractor(
        parameter=parameter,
        source=_display_path(path),
    ).extract(tree)


def scan_python_paths(paths: Iterable[Path], parameter: str) -> ConstraintSet:
    merged = ConstraintSet(parameter=parameter, metadata={"extractor": "python_ast"})
    scanned_files = 0
    parse_errors: list[dict[str, str]] = []

    for path in _iter_python_files(paths):
        scanned_files += 1
        try:
            merged.extend(extract_python_file(path, parameter).constraints)
        except (OSError, SyntaxError) as exc:
            parse_errors.append({"source": str(path), "error": str(exc)})

    merged.metadata.update(
        {
            "scanned_files": scanned_files,
            "parse_errors": parse_errors,
        }
    )
    return merged


def _iter_python_files(paths: Iterable[Path]) -> Iterator[Path]:
    ignored_parts = {
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
    for input_path in paths:
        path = input_path.resolve()
        if path.is_file() and path.suffix == ".py":
            if path not in seen:
                seen.add(path)
                yield path
            continue
        if not path.is_dir():
            continue
        for candidate in sorted(path.rglob("*.py")):
            if any(part in ignored_parts for part in candidate.parts):
                continue
            if candidate not in seen:
                seen.add(candidate)
                yield candidate


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def _assigned_names(node: ast.AST) -> set[str]:
    return {child.id for child in ast.walk(node) if isinstance(child, ast.Name)}


def _constant_string(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _body_rejects_or_repairs(body: list[ast.stmt]) -> bool:
    for statement in body:
        for child in ast.walk(statement):
            if isinstance(child, ast.Raise):
                return True
            if isinstance(child, ast.Call):
                name = _call_name(child.func)
                if name in {"_apply_fix", "apply_fix"}:
                    return True
    return False


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _split_valid_condition(node: ast.expr) -> list[ast.expr]:
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
        result: list[ast.expr] = []
        for value in node.values:
            result.extend(_split_valid_condition(value))
        return result
    return [node]


def _invert_rejecting_condition(node: ast.expr) -> list[ast.expr]:
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
        result: list[ast.expr] = []
        for value in node.values:
            result.extend(_invert_rejecting_condition(value))
        return result

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return [node.operand]

    if isinstance(node, ast.Compare) and len(node.ops) == 1:
        inverse = _INVERSE_COMPARATORS.get(type(node.ops[0]))
        if inverse is not None:
            return [
                ast.Compare(
                    left=node.left,
                    ops=[inverse()],
                    comparators=node.comparators,
                )
            ]

    return [ast.UnaryOp(op=ast.Not(), operand=node)]


def _parameter_names(node: ast.AST, target: str) -> list[str]:
    names: list[str] = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Name):
            continue
        name = child.id
        if name in _IGNORED_NAMES or name.startswith("_"):
            continue
        if name not in names:
            names.append(name)
    if target in names:
        names.remove(target)
    return [target, *names]


def _classify(node: ast.AST, parameters: list[str]) -> ConstraintKind:
    text = ast.unparse(node).lower()
    if any(token in text for token in ("memory", "device_count", "world_size", "rank")):
        return ConstraintKind.ENVIRONMENT
    if any(isinstance(child, ast.Mod) for child in ast.walk(node)):
        return ConstraintKind.RELATION
    if len(parameters) > 1:
        return ConstraintKind.RELATION
    if isinstance(node, ast.Compare):
        if any(isinstance(op, (ast.In, ast.NotIn)) for op in node.ops):
            return ConstraintKind.ENUM
        if any(isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE)) for op in node.ops):
            return ConstraintKind.RANGE
    if isinstance(node, (ast.BoolOp, ast.IfExp)):
        return ConstraintKind.CONDITIONAL
    return ConstraintKind.OTHER
