from __future__ import annotations

import ast
import copy
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping

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
    "all",
    "any",
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

_CONFIG_ROOT_HINTS = {
    "args",
    "cfg",
    "config",
    "configuration",
    "model",
    "model_cfg",
    "model_config",
    "parallel",
    "parallel_cfg",
    "training",
    "training_cfg",
    "moe",
    "moe_cfg",
    "mla",
    "mla_cfg",
    "rope",
    "rope_cfg",
}

_CONFIG_CONTAINER_KEYS = {
    "config",
    "configs",
    "configuration",
    "distributed",
    "env",
    "environment",
    "mla",
    "model",
    "moe",
    "parallel",
    "rope",
    "training",
}

_SAFE_CALLS = {
    "abs",
    "all",
    "any",
    "bool",
    "float",
    "int",
    "len",
    "max",
    "min",
    "round",
    "str",
    "sum",
}

_ENVIRONMENT_CALLS = {
    "device_count",
    "get_world_size",
    "world_size",
}

_STANDARD_PARAMETER_ALIASES = {
    "cp": "context_parallel_size",
    "ep": "expert_model_parallel_size",
    "pp": "pipeline_model_parallel_size",
    "sp": "sequence_parallel",
    "tp": "tensor_model_parallel_size",
}

_DISALLOWED_ATTRIBUTES = {
    "block",
    "decoder",
    "dtype",
    "endswith",
    "is_file",
    "is_dir",
    "shape",
    "startswith",
    "suffix",
}


@dataclass(frozen=True, slots=True)
class ExtractedExpression:
    predicate: ast.expr
    condition: ast.expr | None
    line: int
    confidence: float
    detail: str
    source: str | None = None


@dataclass(frozen=True, slots=True)
class _FunctionSummary:
    name: str
    parameters: tuple[str, ...]
    predicate: str
    condition: str | None
    source: str
    line: int
    confidence: float


@dataclass(frozen=True, slots=True)
class _GuardAction:
    kind: str
    call: ast.Call | None = None


@dataclass(frozen=True, slots=True)
class _FileScanResult:
    error: dict[str, str] | None
    results: dict[str, ConstraintSet]


@dataclass(slots=True)
class _Scope:
    bindings: dict[str, ast.expr]

    def clone(self) -> "_Scope":
        return _Scope(bindings={name: copy.deepcopy(value) for name, value in self.bindings.items()})


class _SubstituteBindings(ast.NodeTransformer):
    def __init__(self, bindings: dict[str, ast.expr]):
        self.bindings = bindings
        self._active: set[str] = set()

    def visit_Name(self, node: ast.Name) -> ast.AST:
        replacement = self.bindings.get(node.id)
        if replacement is None or node.id in self._active:
            return node
        self._active.add(node.id)
        try:
            expanded = self.visit(copy.deepcopy(replacement))
        finally:
            self._active.remove(node.id)
        return ast.copy_location(expanded, node)


class _CanonicalizeConfigAccess(ast.NodeTransformer):
    def __init__(
        self,
        parameter: str,
        known_parameters: Iterable[str] = (),
        *,
        canonicalize_self_fields: bool = False,
    ):
        self.parameter = parameter
        self.parameter_leaf = parameter.rsplit(".", 1)[-1]
        self.known_parameters = {
            item.rsplit(".", 1)[-1] for item in known_parameters
        }
        self.canonicalize_self_fields = canonicalize_self_fields

    def visit_Name(self, node: ast.Name) -> ast.AST:
        canonical = _STANDARD_PARAMETER_ALIASES.get(node.id)
        if canonical is not None and canonical in self.known_parameters:
            return ast.copy_location(ast.Name(id=canonical, ctx=node.ctx), node)
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        node = self.generic_visit(node)
        if node.attr == "get" or node.attr in _ENVIRONMENT_CALLS:
            return node
        if (
            isinstance(node.value, ast.Name)
            and node.value.id == "self"
            and self.canonicalize_self_fields
            and not node.attr.startswith("_")
        ):
            return ast.copy_location(ast.Name(id=node.attr, ctx=ast.Load()), node)
        if (
            node.attr == self.parameter_leaf
            or _looks_like_config_container(node.value)
            or (
                isinstance(node.value, ast.Name)
                and node.value.id == "self"
                and node.attr in self.known_parameters
            )
        ):
            return ast.copy_location(ast.Name(id=node.attr, ctx=ast.Load()), node)
        return node

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        node = self.generic_visit(node)
        key = _constant_string(node.slice)
        if key is not None and (key == self.parameter_leaf or _looks_like_config_container(node.value)):
            return ast.copy_location(ast.Name(id=key, ctx=ast.Load()), node)
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        node = self.generic_visit(node)
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and node.args
        ):
            key = _constant_string(node.args[0])
            if key is not None and (
                key == self.parameter_leaf or _looks_like_config_container(node.func.value)
            ):
                return ast.copy_location(ast.Name(id=key, ctx=ast.Load()), node)
        return node


class _SimplifyConstraintExpression(ast.NodeTransformer):
    def visit_Call(self, node: ast.Call) -> ast.AST:
        node = self.generic_visit(node)
        if not (
            isinstance(node.func, ast.Name)
            and node.func.id == "max"
            and len(node.args) == 2
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == 1
        ):
            return node
        if _is_positive_alignment_expression(node.args[1]):
            return ast.copy_location(node.args[1], node)
        return node

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        node = self.generic_visit(node)
        if not (
            len(node.ops) == 1
            and len(node.comparators) == 1
            and isinstance(node.left, ast.BinOp)
            and isinstance(node.left.op, ast.FloorDiv)
            and isinstance(node.comparators[0], ast.Constant)
            and node.comparators[0].value in {0, 1}
            and isinstance(node.ops[0], (ast.Gt, ast.GtE))
            and _is_positive_alignment_expression(node.left.right)
        ):
            return node
        threshold = node.comparators[0].value
        if (
            isinstance(node.ops[0], ast.GtE)
            and threshold == 1
        ) or (
            isinstance(node.ops[0], ast.Gt)
            and threshold == 0
        ):
            return ast.copy_location(
                ast.Compare(
                    left=node.left.left,
                    ops=[ast.GtE()],
                    comparators=[node.left.right],
                ),
                node,
            )
        return node


class PythonConstraintExtractor(ast.NodeVisitor):
    """Extract high-precision configuration constraints from Python source.

    The extractor is intentionally bounded. It recognizes assertions and direct
    reject/repair guards, tracks exact aliases plus simple symbolic assignments
    within lexical scopes, and carries enclosing branch predicates into
    conditional constraints. Results remain candidates rather than ground truth.
    """

    def __init__(
        self,
        parameter: str,
        source: str,
        *,
        strict: bool = True,
        function_summaries: dict[str, tuple[_FunctionSummary, ...]] | None = None,
        known_parameters: Iterable[str] = (),
    ):
        self.parameter = parameter
        self.parameter_leaf = parameter.rsplit(".", 1)[-1]
        self.source = source
        self.strict = strict
        self._function_summaries = function_summaries or {}
        self._known_parameters = tuple(known_parameters)
        self._global_bindings: dict[str, ast.expr] = {}
        self._config_keys: set[str] = {self.parameter_leaf}
        self._scope = _Scope(bindings={})
        self._scope_stack: list[_Scope] = []
        self._path_conditions: list[ast.expr] = []
        self._class_stack: list[str] = []
        self._expressions: list[ExtractedExpression] = []
        self._raw_candidates = 0
        self._filtered_candidates = 0

    def extract(
        self,
        tree: ast.AST,
        *,
        exact_bindings: dict[str, ast.expr] | None = None,
    ) -> ConstraintSet:
        self._global_bindings = {
            name: copy.deepcopy(value)
            for name, value in (
                exact_bindings
                if exact_bindings is not None
                else _collect_exact_config_bindings(tree)
            ).items()
        }
        self._config_keys = {
            value.id
            for value in self._global_bindings.values()
            if isinstance(value, ast.Name)
        }
        self._config_keys.add(self.parameter_leaf)
        self._scope = self._fresh_scope()
        self.visit(tree)

        result = ConstraintSet(
            parameter=self.parameter,
            metadata={
                "extractor": "python_ast",
                "source": self.source,
                "mode": "strict" if self.strict else "broad",
                "exact_aliases": sorted(
                    name
                    for name, value in self._global_bindings.items()
                    if name != self.parameter_leaf
                    and isinstance(value, ast.Name)
                    and value.id == self.parameter_leaf
                ),
            },
        )
        for extracted in self._expressions:
            self._raw_candidates += 1
            predicate = _SimplifyConstraintExpression().visit(
                _simplify_boolean(extracted.predicate)
            )
            condition = (
                _SimplifyConstraintExpression().visit(
                    _simplify_boolean(extracted.condition)
                )
                if extracted.condition is not None
                else None
            )
            predicate, condition = _orient_conditional(
                predicate,
                condition,
                self.parameter_leaf,
            )
            if condition is not None and _is_literal_true(condition):
                condition = None
            if condition is not None and _same_expression(condition, predicate):
                self._filtered_candidates += 1
                continue
            if not (
                _contains_name(predicate, self.parameter_leaf)
                or (
                    condition is not None
                    and _contains_name(condition, self.parameter_leaf)
                )
            ):
                self._filtered_candidates += 1
                continue
            if not _is_supported_constraint(
                predicate,
                self.parameter_leaf,
                strict=self.strict,
                allow_boolean_name=condition is not None,
            ):
                self._filtered_candidates += 1
                continue
            if condition is not None and not _is_supported_condition(
                condition,
                self.parameter_leaf,
                strict=self.strict,
            ):
                self._filtered_candidates += 1
                continue

            expression = ast.unparse(predicate)
            parameters = _parameter_names(predicate, self.parameter_leaf)
            if condition is not None:
                expression = f"{ast.unparse(condition)} => {expression}"
                parameters = _merge_names(
                    _parameter_names(condition, self.parameter_leaf),
                    parameters,
                    target=self.parameter_leaf,
                )
            if self.parameter_leaf not in parameters:
                self._filtered_candidates += 1
                continue

            result.add(
                Constraint(
                    expression=expression,
                    kind=(
                        ConstraintKind.CONDITIONAL
                        if condition is not None
                        else _classify(predicate, parameters)
                    ),
                    parameters=tuple(
                        self.parameter if name == self.parameter_leaf else name
                        for name in parameters
                    ),
                    confidence=extracted.confidence,
                    evidence=(
                        Evidence(
                            kind=EvidenceKind.STATIC,
                            source=extracted.source or self.source,
                            line=extracted.line,
                            detail=extracted.detail,
                        ),
                    ),
                )
            )

        result.metadata.update(
            {
                "raw_candidates": self._raw_candidates,
                "filtered_candidates": self._filtered_candidates,
                "accepted_candidates": len(result.constraints),
            }
        )
        return result

    def _fresh_scope(self) -> _Scope:
        return _Scope(
            bindings={
                name: copy.deepcopy(value)
                for name, value in self._global_bindings.items()
            }
        )

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._class_stack.append(node.name)
        try:
            for statement in node.body:
                self.visit(statement)
        finally:
            self._class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self._scope_stack.append(self._scope)
        previous_path = self._path_conditions
        self._scope = self._fresh_scope()
        self._path_conditions = []
        try:
            for statement in node.body:
                self.visit(statement)
        finally:
            self._scope = self._scope_stack.pop()
            self._path_conditions = previous_path

    def visit_Assign(self, node: ast.Assign) -> None:
        value = self._prepare(node.value)
        for target in node.targets:
            self._update_binding(target, value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        value = self._prepare(node.value) if node.value is not None else None
        self._update_binding(node.target, value)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        for name in _assigned_names(node.target):
            self._scope.bindings.pop(name, None)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self._update_binding(node.target, self._prepare(node.value))

    def visit_Expr(self, node: ast.Expr) -> None:
        if isinstance(node.value, ast.Call):
            self._apply_function_summaries(node.value, node.lineno)

    def _update_binding(self, target: ast.AST, value: ast.expr | None) -> None:
        names = _assigned_names(target)
        for name in names:
            self._scope.bindings.pop(name, None)
        if value is None or not _safe_symbolic_expression(value):
            return
        if not _uses_only_symbolic_names(
            value,
            {*self._config_keys, *self._known_parameters},
        ):
            return
        for name in names:
            self._scope.bindings[name] = copy.deepcopy(value)

    def visit_Assert(self, node: ast.Assert) -> None:
        prepared = self._prepare(node.test)
        if _contains_name(prepared, self.parameter_leaf):
            for predicate in _split_top_level_and(prepared):
                self._record(
                    predicate=predicate,
                    condition=_combine_and(self._path_conditions),
                    line=node.lineno,
                    confidence=1.0,
                    detail="assertion",
                )

    def visit_If(self, node: ast.If) -> None:
        test = self._prepare(node.test)
        action = _direct_guard_action(node.body)
        opposite_action = _direct_guard_action(node.orelse)
        if (
            action is not None
            and action.kind == "repair"
            and action.call is not None
            and self._is_dynamic_repair_change_guard(test, action.call)
        ):
            action = None
        if (
            action is not None
            and opposite_action is None
            and _contains_name(test, self.parameter_leaf)
        ):
            confidence = 0.98 if action.kind == "reject" else 0.90
            detail = "rejecting guard" if action.kind == "reject" else "repairing guard"
            for predicate, local_condition in _valid_constraints_from_rejecting_guard(
                test,
                self.parameter_leaf,
            ):
                conditions = [*self._path_conditions]
                if local_condition is not None:
                    conditions.append(local_condition)
                self._record(
                    predicate=predicate,
                    condition=_combine_and(conditions),
                    line=node.lineno,
                    confidence=confidence,
                    detail=detail,
                )

        before = self._scope.clone()
        previous_path = self._path_conditions

        self._scope = before.clone()
        self._path_conditions = [*previous_path, test]
        for statement in node.body:
            self.visit(statement)
        body_scope = self._scope

        self._scope = before.clone()
        self._path_conditions = [*previous_path, _negate(test)]
        for statement in node.orelse:
            self.visit(statement)
        else_scope = self._scope

        self._scope = _merge_branch_scopes(
            before,
            body_scope,
            else_scope,
            has_else=bool(node.orelse),
        )
        self._path_conditions = previous_path

    def visit_For(self, node: ast.For) -> None:
        before = self._scope.clone()
        loop_scope = before.clone()
        self._scope = loop_scope
        for name in _assigned_names(node.target):
            self._scope.bindings.pop(name, None)
        for statement in node.body:
            self.visit(statement)
        self._scope = _merge_branch_scopes(before, self._scope, before, has_else=False)
        for statement in node.orelse:
            self.visit(statement)

    def visit_While(self, node: ast.While) -> None:
        before = self._scope.clone()
        self._scope = before.clone()
        previous_path = self._path_conditions
        test = self._prepare(node.test)
        self._path_conditions = [*previous_path, test]
        for statement in node.body:
            self.visit(statement)
        self._scope = _merge_branch_scopes(before, self._scope, before, has_else=False)
        self._path_conditions = previous_path
        for statement in node.orelse:
            self.visit(statement)

    def visit_Try(self, node: ast.Try) -> None:
        before = self._scope.clone()
        previous_path = self._path_conditions

        self._scope = before.clone()
        for statement in node.body:
            self.visit(statement)
        for statement in node.orelse:
            self.visit(statement)
        branch_scopes = [self._scope]

        for handler in node.handlers:
            self._scope = before.clone()
            if handler.name:
                self._scope.bindings.pop(handler.name, None)
            for statement in handler.body:
                self.visit(statement)
            branch_scopes.append(self._scope)

        self._scope = _merge_equal_scopes(branch_scopes)
        self._path_conditions = previous_path
        for statement in node.finalbody:
            self.visit(statement)

    def _is_dynamic_repair_change_guard(
        self,
        test: ast.expr,
        call: ast.Call,
    ) -> bool:
        if len(call.args) < 3:
            return False
        if not (
            isinstance(test, ast.Compare)
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.NotEq)
            and len(test.comparators) == 1
        ):
            return False
        old_value = self._prepare(call.args[1])
        new_value = self._prepare(call.args[2])
        left = test.left
        right = test.comparators[0]
        compares_old_new = (
            _same_expression(left, old_value) and _same_expression(right, new_value)
        ) or (
            _same_expression(left, new_value) and _same_expression(right, old_value)
        )
        return compares_old_new and _is_dynamic_repair_value(
            new_value,
            self._config_keys,
        )

    def _apply_function_summaries(self, call: ast.Call, line: int) -> None:
        name = _call_name(call.func)
        if name is None:
            return
        summaries = self._function_summaries.get(name)
        if not summaries:
            return
        for summary in summaries:
            bindings: dict[str, ast.expr] = {}
            for formal, actual in zip(summary.parameters, call.args):
                bindings[formal] = self._prepare_summary_argument(actual)
            for keyword in call.keywords:
                if keyword.arg in summary.parameters:
                    bindings[keyword.arg] = self._prepare_summary_argument(keyword.value)
            predicate = ast.parse(summary.predicate, mode="eval").body
            condition = None
            if summary.condition is not None:
                condition = ast.parse(summary.condition, mode="eval").body
            used_formals = {
                child.id
                for expression in (predicate, condition)
                if expression is not None
                for child in ast.walk(expression)
                if isinstance(child, ast.Name) and child.id in summary.parameters
            }
            if any(formal not in bindings for formal in used_formals):
                continue
            predicate = _SubstituteBindings(bindings).visit(predicate)
            if condition is not None:
                condition = _SubstituteBindings(bindings).visit(condition)
            predicate = self._prepare(predicate)
            condition = self._prepare(condition) if condition is not None else None
            if not (
                _contains_name(predicate, self.parameter_leaf)
                or (
                    condition is not None
                    and _contains_name(condition, self.parameter_leaf)
                )
            ):
                continue
            conditions = [*self._path_conditions]
            if condition is not None:
                conditions.append(condition)
            self._record(
                predicate=predicate,
                condition=_combine_and(conditions),
                line=summary.line,
                confidence=summary.confidence,
                detail=f"interprocedural summary instantiated at line {line}",
                source=summary.source,
            )

    def _prepare_summary_argument(self, node: ast.expr) -> ast.expr:
        prepared = self._prepare(node)
        prepared = _SubstituteBindings(self._global_bindings).visit(prepared)
        prepared = _CanonicalizeConfigAccess(
            self.parameter_leaf,
            self._known_parameters,
            canonicalize_self_fields=self._in_configuration_class(),
        ).visit(prepared)
        ast.fix_missing_locations(prepared)
        assert isinstance(prepared, ast.expr)
        return prepared

    def _prepare(self, node: ast.expr) -> ast.expr:
        expanded = _SubstituteBindings(self._scope.bindings).visit(copy.deepcopy(node))
        canonical = _CanonicalizeConfigAccess(
            self.parameter_leaf,
            self._known_parameters,
            canonicalize_self_fields=self._in_configuration_class(),
        ).visit(expanded)
        canonical = _SimplifyConstraintExpression().visit(canonical)
        ast.fix_missing_locations(canonical)
        assert isinstance(canonical, ast.expr)
        return canonical

    def _in_configuration_class(self) -> bool:
        for name in self._class_stack:
            lowered = name.lower()
            if lowered.endswith(("args", "arguments", "config", "validator")):
                return True
        return False

    def _record(
        self,
        *,
        predicate: ast.expr,
        condition: ast.expr | None,
        line: int,
        confidence: float,
        detail: str,
        source: str | None = None,
    ) -> None:
        self._expressions.append(
            ExtractedExpression(
                predicate=copy.deepcopy(predicate),
                condition=copy.deepcopy(condition),
                line=line,
                confidence=max(0.0, confidence - (0.03 if condition is not None else 0.0)),
                detail=detail,
                source=source,
            )
        )


def extract_python_file(path: Path, parameter: str, *, strict: bool = True) -> ConstraintSet:
    source_text = path.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(source_text, filename=str(path))
    exact_bindings = _collect_exact_config_bindings(tree)
    function_summaries = _collect_function_summaries_from_tree(
        tree,
        source=_display_path(path),
        strict=strict,
    )
    return extract_python_tree(
        tree,
        source=_display_path(path),
        parameter=parameter,
        strict=strict,
        exact_bindings=exact_bindings,
        function_summaries=function_summaries,
    )


def extract_python_tree(
    tree: ast.AST,
    *,
    source: str,
    parameter: str,
    strict: bool = True,
    exact_bindings: dict[str, ast.expr] | None = None,
    function_summaries: dict[str, tuple[_FunctionSummary, ...]] | None = None,
    known_parameters: Iterable[str] = (),
) -> ConstraintSet:
    known = tuple(dict.fromkeys(str(item).rsplit(".", 1)[-1] for item in known_parameters))
    result = PythonConstraintExtractor(
        parameter=parameter,
        source=source,
        strict=strict,
        function_summaries=function_summaries,
        known_parameters=known,
    ).extract(tree, exact_bindings=exact_bindings)
    inferred = _infer_model_semantic_constraints(
        tree,
        source=source,
        parameter=parameter,
        known_parameters=known,
    )
    result.extend(inferred)
    result.metadata["inferred_model_semantics"] = len(inferred)
    result.metadata["accepted_candidates"] = len(result.constraints)
    return result


def _infer_model_semantic_constraints(
    tree: ast.AST,
    *,
    source: str,
    parameter: str,
    known_parameters: Iterable[str],
) -> tuple[Constraint, ...]:
    """Infer bounded shape/API relations that are implicit in model code.

    Framework model implementations often encode configuration requirements in
    shape arithmetic rather than explicit validators.  We keep this pass narrow:
    exact configuration-backed floor division yields a divisibility candidate,
    and MoE router ``topk`` calls yield a bound only when both the requested k
    and the expert/group count are unambiguously backed by configuration fields
    in the same class.
    """

    target = parameter.rsplit(".", 1)[-1]
    known = set(known_parameters)
    known.add(target)
    constraints: list[Constraint] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()

    scopes: list[tuple[ast.AST, dict[str, str]]] = [(tree, {})]
    scopes.extend((node, _self_field_config_aliases(node)) for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
    for scope, aliases in scopes:
        for node in ast.walk(scope):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.FloorDiv):
                numerator = _config_reference_key(node.left, aliases, known)
                denominator = _config_reference_key(node.right, aliases, known)
                if (
                    numerator is not None
                    and denominator is not None
                    and numerator != denominator
                    and target in {numerator, denominator}
                ):
                    _append_inferred_constraint(
                        constraints,
                        seen,
                        parameter=parameter,
                        target=target,
                        expression=f"{numerator} % {denominator} == 0",
                        parameters=(numerator, denominator),
                        source=source,
                        line=getattr(node, "lineno", None),
                        confidence=_shape_division_confidence(numerator, denominator),
                        detail="configuration-backed shape floor-division",
                    )

            if not _is_topk_call(node):
                continue
            k_node = _topk_k_argument(node)
            if k_node is None:
                continue
            k_parameter = _config_reference_key(k_node, aliases, known)
            upper = _topk_upper_parameter(k_parameter, aliases, known)
            if k_parameter is None or upper is None or target not in {k_parameter, upper}:
                continue
            _append_inferred_constraint(
                constraints,
                seen,
                parameter=parameter,
                target=target,
                expression=f"{k_parameter} <= {upper}",
                parameters=(k_parameter, upper),
                source=source,
                line=getattr(node, "lineno", None),
                confidence=0.90,
                detail="top-k API bound over configuration-backed router dimension",
            )

    return tuple(constraints)


def _shape_division_confidence(numerator: str, denominator: str) -> float:
    exact_groupings = {
        ("num_attention_heads", "num_key_value_heads"),
        ("num_attention_heads", "num_query_groups"),
        ("n_routed_experts", "n_group"),
        ("num_experts", "num_expert_groups"),
    }
    if (numerator, denominator) in exact_groupings:
        return 0.92
    return 0.72


def _append_inferred_constraint(
    output: list[Constraint],
    seen: set[tuple[str, tuple[str, ...]]],
    *,
    parameter: str,
    target: str,
    expression: str,
    parameters: tuple[str, ...],
    source: str,
    line: int | None,
    confidence: float,
    detail: str,
) -> None:
    rendered_parameters = tuple(
        parameter if name == target else name for name in parameters
    )
    fingerprint = (expression, rendered_parameters)
    if fingerprint in seen:
        return
    seen.add(fingerprint)
    output.append(
        Constraint(
            expression=expression,
            kind=ConstraintKind.RELATION,
            parameters=rendered_parameters,
            confidence=confidence,
            evidence=(
                Evidence(
                    kind=EvidenceKind.STATIC,
                    source=source,
                    line=line,
                    detail=detail,
                ),
            ),
        )
    )


def _self_field_config_aliases(node: ast.ClassDef) -> dict[str, str]:
    discovered: dict[str, str] = {}
    ambiguous: set[str] = set()
    for child in ast.walk(node):
        value: ast.expr | None = None
        targets: list[ast.AST] = []
        if isinstance(child, ast.Assign):
            value = child.value
            targets = list(child.targets)
        elif isinstance(child, ast.AnnAssign) and child.value is not None:
            value = child.value
            targets = [child.target]
        if value is None:
            continue
        key = _direct_config_key(value)
        if key is None:
            continue
        for item in targets:
            if not (
                isinstance(item, ast.Attribute)
                and isinstance(item.value, ast.Name)
                and item.value.id == "self"
            ):
                continue
            previous = discovered.get(item.attr)
            if previous is not None and previous != key:
                ambiguous.add(item.attr)
            else:
                discovered[item.attr] = key
    return {name: key for name, key in discovered.items() if name not in ambiguous}


def _config_reference_key(
    node: ast.AST,
    aliases: Mapping[str, str],
    known: set[str],
) -> str | None:
    key = _direct_config_key(node)
    if key in known:
        return key
    if isinstance(node, ast.Name) and node.id in known:
        return node.id
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    ):
        alias = aliases.get(node.attr)
        if alias in known:
            return alias
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
        and len(node.args) >= 2
        and _looks_like_config_container(node.args[0])
    ):
        key = _constant_string(node.args[1])
        if key in known:
            return key
    return None


def _is_topk_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "topk"
    )


def _topk_k_argument(node: ast.Call) -> ast.AST | None:
    for keyword in node.keywords:
        if keyword.arg == "k":
            return keyword.value
    if not isinstance(node.func, ast.Attribute):
        return None
    root = _expression_root_name(node.func.value)
    index = 1 if root == "torch" else 0
    return node.args[index] if len(node.args) > index else None


def _topk_upper_parameter(
    k_parameter: str | None,
    aliases: Mapping[str, str],
    known: set[str],
) -> str | None:
    if k_parameter is None:
        return None
    preferred: tuple[str, ...]
    if k_parameter in {"num_experts_per_tok", "moe_router_topk"}:
        preferred = ("num_local_experts", "n_routed_experts", "num_experts")
    elif k_parameter in {"topk_group", "moe_topk_group"}:
        preferred = ("n_group", "moe_router_num_groups")
    else:
        return None
    available = set(aliases.values()) | known
    matches = [name for name in preferred if name in available]
    return matches[0] if matches else None


def scan_python_paths(
    paths: Iterable[Path],
    parameter: str,
    *,
    strict: bool = True,
) -> ConstraintSet:
    return scan_python_paths_multi(paths, [parameter], strict=strict)[parameter]


def scan_python_paths_multi(
    paths: Iterable[Path],
    parameters: Iterable[str],
    *,
    strict: bool = True,
    jobs: int = 1,
) -> dict[str, ConstraintSet]:
    ordered_parameters = list(dict.fromkeys(str(parameter) for parameter in parameters))
    known_parameters = tuple(
        parameter.rsplit(".", 1)[-1] for parameter in ordered_parameters
    )
    results = {
        parameter: ConstraintSet(
            parameter=parameter,
            metadata={
                "extractor": "python_ast",
                "mode": "strict" if strict else "broad",
                "scanned_files": 0,
                "parse_errors": [],
                "raw_candidates": 0,
                "filtered_candidates": 0,
            },
        )
        for parameter in ordered_parameters
    }

    files = list(_iter_python_files(paths))
    worker_count = _resolve_worker_count(jobs, len(files))
    serialized_summaries = tuple(
        summary
        for summaries in _collect_global_function_summaries(files, strict=strict).values()
        for summary in summaries
    )
    work_items = [
        (
            str(path),
            tuple(ordered_parameters),
            strict,
            serialized_summaries,
            known_parameters,
        )
        for path in files
    ]
    if worker_count == 1:
        _merge_file_scan_results(results, map(_scan_python_file, work_items))
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            _merge_file_scan_results(
                results,
                executor.map(_scan_python_file, work_items),
            )

    for result in results.values():
        result.metadata["accepted_candidates"] = len(result.constraints)
    return results


def _scan_python_file(
    work_item: tuple[
        str,
        tuple[str, ...],
        bool,
        tuple[_FunctionSummary, ...],
        tuple[str, ...],
    ],
) -> _FileScanResult:
    raw_path, parameters, strict, serialized_summaries, known_parameters = work_item
    path = Path(raw_path)
    try:
        source_text = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source_text, filename=str(path))
    except (OSError, SyntaxError) as exc:
        return _FileScanResult(
            error={"source": str(path), "error": str(exc)},
            results={},
        )

    exact_bindings = _collect_exact_config_bindings(tree)
    function_summaries: dict[str, list[_FunctionSummary]] = {}
    for summary in serialized_summaries:
        if summary.name in source_text:
            function_summaries.setdefault(summary.name, []).append(summary)
    frozen_summaries = {
        name: tuple(items) for name, items in function_summaries.items()
    }
    source_name = _display_path(path)
    extracted: dict[str, ConstraintSet] = {}
    for parameter in parameters:
        leaf = parameter.rsplit(".", 1)[-1]
        if leaf not in source_text:
            continue
        extracted[parameter] = extract_python_tree(
            tree,
            source=source_name,
            parameter=parameter,
            strict=strict,
            exact_bindings=exact_bindings,
            function_summaries=frozen_summaries,
            known_parameters=known_parameters,
        )
    return _FileScanResult(error=None, results=extracted)


def _merge_file_scan_results(
    merged_results: dict[str, ConstraintSet],
    file_results: Iterable[_FileScanResult],
) -> None:
    for file_result in file_results:
        if file_result.error is not None:
            for result in merged_results.values():
                result.metadata["parse_errors"].append(file_result.error)
            continue
        for parameter, extracted in file_result.results.items():
            merged = merged_results[parameter]
            merged.metadata["scanned_files"] += 1
            merged.extend(extracted.constraints)
            merged.metadata["raw_candidates"] += int(
                extracted.metadata.get("raw_candidates", 0)
            )
            merged.metadata["filtered_candidates"] += int(
                extracted.metadata.get("filtered_candidates", 0)
            )


def _resolve_worker_count(jobs: int, file_count: int) -> int:
    if file_count <= 1:
        return 1
    if jobs < 0:
        raise ValueError("jobs must be non-negative")
    if jobs == 0:
        jobs = min(4, os.cpu_count() or 1)
    return max(1, min(jobs, file_count))


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


def _collect_exact_config_bindings(tree: ast.AST) -> dict[str, ast.expr]:
    """Collect unambiguous ``local_name -> config_key`` bindings.

    This pre-pass is deliberately one-hop. Chaining aliases across the whole
    module would turn branch-local repairs such as ``cp = tp`` into global
    equivalences and corrupt unrelated functions.
    """

    discovered: dict[str, str] = {}
    ambiguous: set[str] = set()
    for node in ast.walk(tree):
        value: ast.expr | None = None
        targets: list[ast.AST] = []
        if isinstance(node, ast.Assign):
            value = node.value
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            value = node.value
            targets = [node.target]
        if value is None:
            continue
        key = _direct_config_key(value)
        if key is None:
            continue
        for target in targets:
            for name in _assigned_names(target):
                if name == key or key in _CONFIG_CONTAINER_KEYS:
                    continue
                previous = discovered.get(name)
                if previous is not None and previous != key:
                    ambiguous.add(name)
                    continue
                discovered[name] = key

    return {
        name: ast.Name(id=key, ctx=ast.Load())
        for name, key in discovered.items()
        if name not in ambiguous
    }


def _collect_global_function_summaries(
    files: Iterable[Path],
    *,
    strict: bool,
) -> dict[str, tuple[_FunctionSummary, ...]]:
    collected: dict[str, list[_FunctionSummary]] = {}
    origins: dict[str, set[str]] = {}
    for path in files:
        try:
            tree = ast.parse(
                path.read_text(encoding="utf-8", errors="replace"),
                filename=str(path),
            )
        except (OSError, SyntaxError):
            continue
        source = _display_path(path)
        for name, summaries in _collect_function_summaries_from_tree(
            tree,
            source=source,
            strict=strict,
        ).items():
            collected.setdefault(name, []).extend(summaries)
            origins.setdefault(name, set()).add(source)

    result: dict[str, tuple[_FunctionSummary, ...]] = {}
    for name, summaries in collected.items():
        if len(origins.get(name, ())) != 1:
            continue
        unique = {
            (
                item.parameters,
                item.predicate,
                item.condition,
                item.source,
                item.line,
            ): item
            for item in summaries
        }
        result[name] = tuple(unique[key] for key in sorted(unique, key=repr))
    return result


class _FunctionSummaryCollector(ast.NodeVisitor):
    def __init__(
        self,
        parameters: tuple[str, ...],
        *,
        source: str,
        strict: bool,
    ):
        self.parameters = parameters
        self.parameter_set = set(parameters)
        self.source = source
        self.strict = strict
        self.scope = _Scope(
            bindings={
                parameter: ast.Name(id=parameter, ctx=ast.Load())
                for parameter in parameters
            }
        )
        self.path_conditions: list[ast.expr] = []
        self.records: list[ExtractedExpression] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        for statement in node.body:
            self.visit(statement)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        for statement in node.body:
            self.visit(statement)

    def visit_Assign(self, node: ast.Assign) -> None:
        value = self._prepare(node.value)
        for target in node.targets:
            self._update_binding(target, value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        value = self._prepare(node.value) if node.value is not None else None
        self._update_binding(node.target, value)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        for name in _assigned_names(node.target):
            self.scope.bindings.pop(name, None)

    def visit_Assert(self, node: ast.Assert) -> None:
        test = self._prepare(node.test)
        for predicate in _split_top_level_and(test):
            self._record(
                predicate,
                _combine_and(self.path_conditions),
                node.lineno,
                1.0,
                "assertion",
            )

    def visit_If(self, node: ast.If) -> None:
        test = self._prepare(node.test)
        action = _direct_guard_action(node.body)
        opposite_action = _direct_guard_action(node.orelse)
        if action is not None and opposite_action is None:
            confidence = 0.98 if action.kind == "reject" else 0.90
            detail = "rejecting guard" if action.kind == "reject" else "repairing guard"
            for parameter in self.parameters:
                if not _contains_name(test, parameter):
                    continue
                for predicate, local_condition in _valid_constraints_from_rejecting_guard(
                    test,
                    parameter,
                ):
                    conditions = [*self.path_conditions]
                    if local_condition is not None:
                        conditions.append(local_condition)
                    self._record(
                        predicate,
                        _combine_and(conditions),
                        node.lineno,
                        confidence,
                        detail,
                    )

        before = self.scope.clone()
        previous_path = self.path_conditions

        self.scope = before.clone()
        self.path_conditions = [*previous_path, test]
        for statement in node.body:
            self.visit(statement)
        body_scope = self.scope

        self.scope = before.clone()
        self.path_conditions = [*previous_path, _negate(test)]
        for statement in node.orelse:
            self.visit(statement)
        else_scope = self.scope

        self.scope = _merge_branch_scopes(
            before,
            body_scope,
            else_scope,
            has_else=bool(node.orelse),
        )
        self.path_conditions = previous_path

    def visit_Try(self, node: ast.Try) -> None:
        before = self.scope.clone()
        previous_path = self.path_conditions
        branch_scopes: list[_Scope] = []

        self.scope = before.clone()
        for statement in node.body:
            self.visit(statement)
        for statement in node.orelse:
            self.visit(statement)
        branch_scopes.append(self.scope)

        for handler in node.handlers:
            self.scope = before.clone()
            for statement in handler.body:
                self.visit(statement)
            branch_scopes.append(self.scope)

        self.scope = _merge_equal_scopes(branch_scopes)
        self.path_conditions = previous_path
        for statement in node.finalbody:
            self.visit(statement)

    def _prepare(self, node: ast.expr) -> ast.expr:
        prepared = _SubstituteBindings(self.scope.bindings).visit(copy.deepcopy(node))
        prepared = _SimplifyConstraintExpression().visit(prepared)
        ast.fix_missing_locations(prepared)
        assert isinstance(prepared, ast.expr)
        return prepared

    def _update_binding(self, target: ast.AST, value: ast.expr | None) -> None:
        names = _assigned_names(target)
        for name in names:
            self.scope.bindings.pop(name, None)
        if value is None or not _safe_symbolic_expression(value):
            return
        referenced = {
            child.id
            for child in ast.walk(value)
            if isinstance(child, ast.Name) and child.id not in _IGNORED_NAMES
        }
        if not referenced.issubset(self.parameter_set):
            return
        for name in names:
            self.scope.bindings[name] = copy.deepcopy(value)

    def _record(
        self,
        predicate: ast.expr,
        condition: ast.expr | None,
        line: int,
        confidence: float,
        detail: str,
    ) -> None:
        predicate = _SimplifyConstraintExpression().visit(_simplify_boolean(predicate))
        condition = (
            _SimplifyConstraintExpression().visit(_simplify_boolean(condition))
            if condition is not None
            else None
        )
        combined_nodes = [predicate]
        if condition is not None:
            combined_nodes.append(condition)
        names = {
            child.id
            for expression in combined_nodes
            for child in ast.walk(expression)
            if isinstance(child, ast.Name) and child.id not in _IGNORED_NAMES
        }
        if not names or not names.issubset(self.parameter_set):
            return
        self.records.append(
            ExtractedExpression(
                predicate=copy.deepcopy(predicate),
                condition=copy.deepcopy(condition),
                line=line,
                confidence=confidence,
                detail=detail,
                source=self.source,
            )
        )


def _collect_function_summaries_from_tree(
    tree: ast.AST,
    *,
    source: str,
    strict: bool,
) -> dict[str, tuple[_FunctionSummary, ...]]:
    summaries: dict[str, list[_FunctionSummary]] = {}
    definitions: dict[str, int] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        definitions[node.name] = definitions.get(node.name, 0) + 1
        parameters = tuple(
            argument.arg
            for argument in (
                *node.args.posonlyargs,
                *node.args.args,
                *node.args.kwonlyargs,
            )
            if argument.arg not in {"self", "cls"}
        )
        if not parameters or not _function_contains_constraint_guard(node):
            continue
        collector = _FunctionSummaryCollector(
            parameters,
            source=source,
            strict=strict,
        )
        collector.visit(copy.deepcopy(node))
        seen: set[tuple[str, str | None]] = set()
        for record in collector.records:
            predicate_text = ast.unparse(record.predicate)
            condition_text = ast.unparse(record.condition) if record.condition is not None else None
            if (predicate_text, condition_text) in seen:
                continue
            if not _summary_expression_is_local(
                predicate_text,
                condition_text,
                parameters,
            ):
                continue
            seen.add((predicate_text, condition_text))
            summaries.setdefault(node.name, []).append(
                _FunctionSummary(
                    name=node.name,
                    parameters=parameters,
                    predicate=predicate_text,
                    condition=condition_text,
                    source=source,
                    line=record.line,
                    confidence=max(0.0, record.confidence - 0.03),
                )
            )

    return {
        name: tuple(items)
        for name, items in summaries.items()
        if definitions.get(name) == 1
    }


def _function_contains_constraint_guard(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Assert):
            return True
        if isinstance(child, ast.If) and _direct_guard_action(child.body) is not None:
            return True
    return False


def _summary_expression_is_local(
    predicate: str,
    condition: str | None,
    parameters: tuple[str, ...],
) -> bool:
    expressions = [ast.parse(predicate, mode="eval").body]
    if condition is not None:
        expressions.append(ast.parse(condition, mode="eval").body)
    allowed = set(parameters)
    for expression in expressions:
        if any(isinstance(child, ast.Attribute) for child in ast.walk(expression)):
            return False
        names = {
            child.id
            for child in ast.walk(expression)
            if isinstance(child, ast.Name) and child.id not in _IGNORED_NAMES
        }
        if not names.issubset(allowed):
            return False
    return True


def _direct_config_key(node: ast.expr) -> str | None:
    if isinstance(node, ast.Attribute) and _looks_like_config_container(node.value):
        return node.attr
    if isinstance(node, ast.Subscript) and _looks_like_config_container(node.value):
        return _constant_string(node.slice)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
        and node.args
        and _looks_like_config_container(node.func.value)
    ):
        return _constant_string(node.args[0])
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"bool", "float", "int", "str"}
        and len(node.args) == 1
    ):
        return _direct_config_key(node.args[0])
    return None


def _assigned_names(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, (ast.Tuple, ast.List)):
        names: set[str] = set()
        for item in node.elts:
            names.update(_assigned_names(item))
        return names
    if isinstance(node, ast.Starred):
        return _assigned_names(node.value)
    return set()


def _constant_string(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _looks_like_config_container(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        name = node.id.lower()
        return name in _CONFIG_ROOT_HINTS or name.endswith(("_cfg", "_config", "_args"))
    if isinstance(node, ast.Attribute):
        if node.attr.lower() in _CONFIG_ROOT_HINTS:
            return True
        return _looks_like_config_container(node.value)
    return False


def _direct_guard_action(body: list[ast.stmt]) -> _GuardAction | None:
    for statement in body:
        if isinstance(statement, ast.Raise):
            return _GuardAction(kind="reject")
        call = _top_level_call(statement)
        if call is not None and _call_name(call.func) in {"_apply_fix", "apply_fix"}:
            return _GuardAction(kind="repair", call=call)
    return None


def _top_level_call(statement: ast.stmt) -> ast.Call | None:
    value: ast.AST | None = None
    if isinstance(statement, ast.Expr):
        value = statement.value
    elif isinstance(statement, (ast.Assign, ast.AnnAssign)):
        value = statement.value
    elif isinstance(statement, ast.Return):
        value = statement.value
    if isinstance(value, ast.Call):
        return value
    return None


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _valid_constraints_from_rejecting_guard(
    node: ast.expr,
    target: str,
) -> list[tuple[ast.expr, ast.expr | None]]:
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
        target_parts = [part for part in node.values if _contains_name(part, target)]
        context_parts = [part for part in node.values if not _contains_name(part, target)]
        if target_parts and context_parts:
            invalid_target = _combine_and(target_parts)
            assert invalid_target is not None
            valid_target = _negate(invalid_target)
            condition = _combine_and(context_parts)
            return [(valid_target, condition)]

        activation_parts = [part for part in node.values if _is_activation_guard(part)]
        invalid_parts = [part for part in node.values if part not in activation_parts]
        if activation_parts and invalid_parts:
            invalid_target = _combine_and(invalid_parts)
            assert invalid_target is not None
            return [(_negate(invalid_target), _combine_and(activation_parts))]

    valid = _negate(node)
    return [(part, None) for part in _split_top_level_and(valid)]


def _is_activation_guard(node: ast.expr) -> bool:
    if isinstance(node, ast.Name):
        return _looks_boolean_name(node.id)
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.Not)
        and isinstance(node.operand, ast.Name)
    ):
        return _looks_boolean_name(node.operand.id)
    if not (
        isinstance(node, ast.Compare)
        and len(node.ops) == 1
        and len(node.comparators) == 1
    ):
        return False
    left = node.left
    right = node.comparators[0]
    return (
        isinstance(left, ast.Name)
        and isinstance(right, ast.Constant)
        and isinstance(node.ops[0], (ast.Eq, ast.NotEq, ast.Gt, ast.GtE, ast.Is, ast.IsNot))
    )


def _negate(node: ast.expr) -> ast.expr:
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return copy.deepcopy(node.operand)
    if isinstance(node, ast.Compare) and len(node.ops) == 1:
        inverse = _INVERSE_COMPARATORS.get(type(node.ops[0]))
        if inverse is not None:
            return ast.Compare(
                left=copy.deepcopy(node.left),
                ops=[inverse()],
                comparators=[copy.deepcopy(item) for item in node.comparators],
            )
    if isinstance(node, ast.BoolOp):
        op: ast.boolop = ast.And() if isinstance(node.op, ast.Or) else ast.Or()
        return ast.BoolOp(op=op, values=[_negate(value) for value in node.values])
    return ast.UnaryOp(op=ast.Not(), operand=copy.deepcopy(node))


def _split_top_level_and(node: ast.expr) -> list[ast.expr]:
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
        result: list[ast.expr] = []
        for value in node.values:
            result.extend(_split_top_level_and(value))
        return result
    return [node]


def _combine_and(nodes: Iterable[ast.expr]) -> ast.expr | None:
    values = [copy.deepcopy(node) for node in nodes]
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return ast.BoolOp(op=ast.And(), values=values)


def _simplify_boolean(node: ast.expr | None) -> ast.expr:
    assert node is not None
    if isinstance(node, ast.BoolOp):
        values = [_simplify_boolean(value) for value in node.values]
        flattened: list[ast.expr] = []
        for value in values:
            if isinstance(value, ast.BoolOp) and type(value.op) is type(node.op):
                flattened.extend(value.values)
            else:
                flattened.append(value)
        if len(flattened) == 1:
            return flattened[0]
        return ast.BoolOp(op=copy.deepcopy(node.op), values=flattened)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        operand = _simplify_boolean(node.operand)
        if isinstance(operand, ast.UnaryOp) and isinstance(operand.op, ast.Not):
            return _simplify_boolean(operand.operand)
        return ast.UnaryOp(op=ast.Not(), operand=operand)
    return copy.deepcopy(node)


def _orient_conditional(
    predicate: ast.expr,
    condition: ast.expr | None,
    target: str,
) -> tuple[ast.expr, ast.expr | None]:
    """Prefer ``target state => enabled flag`` over its contrapositive.

    Validator code often rejects ``target_state and not feature_flag``. A
    literal inversion yields ``not feature_flag => not target_state``, which is
    correct but awkward for both humans and downstream mutation logic. This
    rewrite preserves semantics while placing the target state in the
    antecedent and the required Boolean flag in the consequent.
    """

    if condition is None or not isinstance(predicate, ast.Compare):
        return predicate, condition
    if len(predicate.ops) != 1 or len(predicate.comparators) != 1:
        return predicate, condition
    if not _contains_name(predicate, target):
        return predicate, condition

    clauses = (
        list(condition.values)
        if isinstance(condition, ast.BoolOp) and isinstance(condition.op, ast.And)
        else [condition]
    )
    for index, clause in enumerate(clauses):
        if not (
            isinstance(clause, ast.UnaryOp)
            and isinstance(clause.op, ast.Not)
            and isinstance(clause.operand, ast.Name)
            and _looks_boolean_name(clause.operand.id)
        ):
            continue
        required_state = _negate(predicate)
        remaining = [copy.deepcopy(item) for i, item in enumerate(clauses) if i != index]
        remaining.append(required_state)
        return copy.deepcopy(clause.operand), _combine_and(remaining)
    return predicate, condition


def _looks_boolean_name(name: str) -> bool:
    lowered = name.lower()
    return (
        lowered.startswith(("allow_", "disable_", "enable_", "has_", "is_", "use_"))
        or lowered.endswith(("_enabled", "_flag"))
        or lowered in {
            "sequence_parallel",
            "moe_router_pre_softmax",
            "parallel_output",
            "swiglu",
        }
    )


def _safe_symbolic_expression(node: ast.expr) -> bool:
    allowed = (
        ast.Attribute,
        ast.BinOp,
        ast.BoolOp,
        ast.Call,
        ast.Compare,
        ast.Constant,
        ast.IfExp,
        ast.Name,
        ast.Subscript,
        ast.UnaryOp,
    )
    if not all(isinstance(child, allowed + (ast.Load, ast.operator, ast.boolop, ast.unaryop, ast.cmpop)) for child in ast.walk(node)):
        return False
    return not _contains_disallowed_access(node)


def _uses_only_symbolic_names(node: ast.expr, config_keys: set[str]) -> bool:
    allowed = {
        *config_keys,
        "available_memory",
        "device_count",
        "rank",
        "world_size",
    }
    names = {
        child.id
        for child in ast.walk(node)
        if isinstance(child, ast.Name)
        and child.id not in _IGNORED_NAMES
        and not child.id.startswith("_")
    }
    return all(name in allowed or _looks_configuration_symbol(name) for name in names)


def _looks_configuration_symbol(name: str) -> bool:
    return name.endswith(
        (
            "_cap",
            "_count",
            "_groups",
            "_heads",
            "_iters",
            "_layers",
            "_length",
            "_limit",
            "_rank",
            "_size",
        )
    )


def _is_dynamic_repair_value(node: ast.expr, config_keys: set[str]) -> bool:
    if isinstance(node, ast.Constant):
        return False
    if isinstance(node, ast.Name):
        return node.id not in config_keys and node.id not in {
            "available_memory",
            "device_count",
            "rank",
            "world_size",
        }
    return True


def _is_positive_alignment_expression(node: ast.expr) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (int, float)) and node.value > 0
    if isinstance(node, ast.Name):
        lowered = node.id.lower()
        return lowered.endswith("_parallel_size") or lowered in {
            "world_size",
            "device_count",
        }
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        return _is_positive_alignment_expression(node.left) and _is_positive_alignment_expression(node.right)
    return False


def _is_supported_constraint(
    node: ast.expr,
    target: str,
    *,
    strict: bool,
    allow_boolean_name: bool = False,
) -> bool:
    if _is_tautology(node):
        return False
    if strict:
        if _contains_disallowed_access(node):
            return False
        if _target_used_as_container(node, target):
            return False
    if isinstance(node, ast.Compare):
        for operator, comparator in zip(node.ops, node.comparators):
            if isinstance(operator, (ast.In, ast.NotIn)) and not isinstance(
                comparator,
                (ast.List, ast.Tuple, ast.Set),
            ):
                return False
        return True
    if isinstance(node, ast.BoolOp):
        return all(_is_supported_condition(value, target, strict=strict) for value in node.values)
    if isinstance(node, ast.Name) and allow_boolean_name:
        return _looks_boolean_name(node.id)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return isinstance(node.operand, ast.Name) and node.operand.id == target
    return not strict


def _is_supported_condition(node: ast.expr, target: str, *, strict: bool) -> bool:
    if _is_literal_true(node) or _is_literal_false(node):
        return True
    if strict and _contains_disallowed_access(node):
        return False
    if isinstance(node, ast.Compare):
        return not _is_tautology(node)
    if isinstance(node, ast.Name):
        return True
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return isinstance(node.operand, (ast.Name, ast.Compare, ast.Call))
    if isinstance(node, ast.BoolOp):
        return all(_is_supported_condition(value, target, strict=strict) for value in node.values)
    return not strict


def _contains_disallowed_access(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Attribute) and child.attr in _DISALLOWED_ATTRIBUTES:
            return True
        if isinstance(child, ast.Call):
            name = _call_name(child.func)
            if name in _SAFE_CALLS or name in _ENVIRONMENT_CALLS:
                continue
            if isinstance(child.func, ast.Name):
                return True
            if isinstance(child.func, ast.Attribute) and child.func.attr not in _ENVIRONMENT_CALLS:
                return True
    return False


def _target_used_as_container(node: ast.AST, target: str) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Attribute) and _expression_root_name(child.value) == target:
            return True
        if isinstance(child, ast.Subscript) and _expression_root_name(child.value) == target:
            return True
    return False


def _expression_root_name(node: ast.AST) -> str | None:
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _is_tautology(node: ast.expr) -> bool:
    if not isinstance(node, ast.Compare) or len(node.ops) != 1 or len(node.comparators) != 1:
        return False
    left = node.left
    right = node.comparators[0]
    if not _same_expression(left, right):
        return False
    return isinstance(node.ops[0], (ast.Eq, ast.LtE, ast.GtE, ast.Is))


def _same_expression(left: ast.AST, right: ast.AST) -> bool:
    return ast.dump(left, include_attributes=False) == ast.dump(right, include_attributes=False)


def _contains_name(node: ast.AST, name: str) -> bool:
    return any(isinstance(child, ast.Name) and child.id == name for child in ast.walk(node))


def _is_literal_true(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value is True


def _is_literal_false(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value is False


def _merge_branch_scopes(
    before: _Scope,
    body: _Scope,
    other: _Scope,
    *,
    has_else: bool,
) -> _Scope:
    right = other if has_else else before
    merged: dict[str, ast.expr] = {}
    for name in set(body.bindings) & set(right.bindings):
        if _same_expression(body.bindings[name], right.bindings[name]):
            merged[name] = copy.deepcopy(body.bindings[name])
    return _Scope(bindings=merged)


def _merge_equal_scopes(scopes: list[_Scope]) -> _Scope:
    if not scopes:
        return _Scope(bindings={})
    common_names = set(scopes[0].bindings)
    for scope in scopes[1:]:
        common_names &= set(scope.bindings)
    merged: dict[str, ast.expr] = {}
    for name in common_names:
        first = scopes[0].bindings[name]
        if all(_same_expression(first, scope.bindings[name]) for scope in scopes[1:]):
            merged[name] = copy.deepcopy(first)
    return _Scope(bindings=merged)


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


def _merge_names(first: list[str], second: list[str], *, target: str) -> list[str]:
    merged = [target]
    for name in [*first, *second]:
        if name != target and name not in merged:
            merged.append(name)
    return merged


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
        if any(isinstance(op, (ast.Is, ast.IsNot)) for op in node.ops):
            return ConstraintKind.TYPE
    if isinstance(node, (ast.BoolOp, ast.IfExp)):
        return ConstraintKind.CONDITIONAL
    return ConstraintKind.OTHER
