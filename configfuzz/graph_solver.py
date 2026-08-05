from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

import z3

from configfuzz.dependencies import (
    DependencyEdge,
    DependencyGraph,
    DependencyNodeKind,
    DependencyRelation,
    DependencyStatus,
)


class SolveStatus(str, Enum):
    SAT = "sat"
    UNSAT = "unsat"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class SolverMutationPlan:
    status: SolveStatus
    target_parameter: str
    requested_value: Any
    proposed_changes: tuple[tuple[str, Any], ...] = ()
    mutable_parameters: tuple[str, ...] = ()
    compiled_edges: tuple[str, ...] = ()
    hard_edges: tuple[str, ...] = ()
    soft_edges: tuple[str, ...] = ()
    violated_soft_edges: tuple[str, ...] = ()
    unsupported_edges: tuple[str, ...] = ()
    excluded_edges: tuple[str, ...] = ()
    out_of_scope_edges: tuple[str, ...] = ()
    missing_context: tuple[str, ...] = ()
    reason: str | None = None

    @property
    def changes(self) -> dict[str, Any]:
        return dict(self.proposed_changes)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": self.status.value,
            "target_parameter": self.target_parameter,
            "requested_value": self.requested_value,
            "proposed_changes": self.changes,
            "mutable_parameters": list(self.mutable_parameters),
            "compiled_edges": list(self.compiled_edges),
            "hard_edges": list(self.hard_edges),
            "soft_edges": list(self.soft_edges),
            "violated_soft_edges": list(self.violated_soft_edges),
            "unsupported_edges": list(self.unsupported_edges),
            "excluded_edges": list(self.excluded_edges),
            "out_of_scope_edges": list(self.out_of_scope_edges),
            "missing_context": list(self.missing_context),
        }
        if self.reason is not None:
            payload["reason"] = self.reason
        return payload


@dataclass(frozen=True, slots=True)
class InterventionCase:
    role: str
    status: SolveStatus
    configuration: tuple[tuple[str, Any], ...] = ()
    changes: tuple[tuple[str, Any], ...] = ()
    hard_edges: tuple[str, ...] = ()
    soft_edges: tuple[str, ...] = ()
    violated_soft_edges: tuple[str, ...] = ()
    unsupported_edges: tuple[str, ...] = ()
    missing_context: tuple[str, ...] = ()
    reason: str | None = None

    @property
    def assignment(self) -> dict[str, Any]:
        return dict(self.configuration)

    @property
    def changed_values(self) -> dict[str, Any]:
        return dict(self.changes)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "role": self.role,
            "status": self.status.value,
            "configuration": self.assignment,
            "changes": self.changed_values,
            "hard_edges": list(self.hard_edges),
            "soft_edges": list(self.soft_edges),
            "violated_soft_edges": list(self.violated_soft_edges),
            "unsupported_edges": list(self.unsupported_edges),
            "missing_context": list(self.missing_context),
        }
        if self.reason is not None:
            payload["reason"] = self.reason
        return payload


@dataclass(frozen=True, slots=True)
class InterventionPlan:
    intervention_id: str
    edge_id: str
    expression: str
    primary_parameter: str
    mutable_parameters: tuple[str, ...]
    satisfying: InterventionCase
    violating: InterventionCase
    repaired: InterventionCase | None = None
    provenance: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "intervention_id": self.intervention_id,
            "edge_id": self.edge_id,
            "expression": self.expression,
            "primary_parameter": self.primary_parameter,
            "mutable_parameters": list(self.mutable_parameters),
            "provenance": list(self.provenance),
            "cases": {
                "satisfying": self._case_payload(self.satisfying),
                "violating": self._case_payload(self.violating),
            },
        }
        if self.repaired is not None:
            payload["cases"]["repaired"] = self._case_payload(self.repaired)
        return payload

    def _case_payload(self, case: InterventionCase) -> dict[str, Any]:
        payload = case.to_dict()
        assignment = case.assignment
        if case.status is SolveStatus.SAT and self.primary_parameter in assignment:
            payload["probe_sample_template"] = {
                "parameter": self.primary_parameter,
                "value": assignment[self.primary_parameter],
                "assignments": {
                    name: value
                    for name, value in assignment.items()
                    if name != self.primary_parameter
                },
                "intervention_id": self.intervention_id,
                "intervention_edge_id": self.edge_id,
                "intervention_role": case.role,
            }
        return payload


class _ValueKind(str, Enum):
    BOOL = "bool"
    INT = "int"
    REAL = "real"
    STRING = "string"


@dataclass(frozen=True, slots=True)
class _CompiledExpression:
    value: z3.ExprRef
    side_constraints: tuple[z3.BoolRef, ...] = ()


class _CompileError(ValueError):
    def __init__(self, message: str, missing: set[str] | None = None):
        super().__init__(message)
        self.missing = frozenset(missing or ())


def solve_graph_mutation(
    graph: DependencyGraph,
    baseline: Mapping[str, Any],
    parameter: str,
    value: Any,
    *,
    static_as_hard: bool = False,
) -> SolverMutationPlan:
    if parameter not in graph.nodes:
        raise KeyError(f"unknown dependency node: {parameter}")

    context = normalize_context(graph, baseline)
    context[parameter] = value
    kinds = _infer_value_kinds(graph, context, parameter, value)
    variables = {
        name: _make_variable(name, kind)
        for name, kind in kinds.items()
    }
    target = variables.get(parameter)
    if target is None:
        return SolverMutationPlan(
            status=SolveStatus.UNKNOWN,
            target_parameter=parameter,
            requested_value=value,
            reason="target parameter type is unsupported",
        )

    mutable = {
        parameter,
        *(
            name
            for name in graph.affected_parameters(parameter)
            if name in variables
            and graph.nodes.get(name) is not None
            and graph.nodes[name].kind
            in {DependencyNodeKind.PARAMETER, DependencyNodeKind.FEATURE}
        ),
    }

    optimizer = z3.Optimize()
    optimizer.set(priority="lex")
    try:
        optimizer.add(target == _literal(value, kinds[parameter]))
    except _CompileError as exc:
        return SolverMutationPlan(
            status=SolveStatus.UNKNOWN,
            target_parameter=parameter,
            requested_value=value,
            reason=str(exc),
        )

    missing: set[str] = set()
    missing.update(name for name in mutable if name not in context)
    for name, variable in variables.items():
        if name in mutable:
            continue
        if name not in context:
            missing.add(name)
            continue
        try:
            optimizer.add(variable == _literal(context[name], kinds[name]))
        except _CompileError:
            missing.add(name)

    compiled: list[str] = []
    hard_edges: list[str] = []
    soft_edges: list[str] = []
    compiled_constraints: dict[str, z3.BoolRef] = {}
    unsupported: list[str] = []
    excluded: list[str] = []
    out_of_scope: list[str] = []
    for edge in sorted(graph.edges.values(), key=lambda item: item.id):
        if not mutable.intersection(edge.participants):
            out_of_scope.append(edge.id)
            continue
        if edge.status is DependencyStatus.CONTRADICTED:
            excluded.append(edge.id)
            continue
        missing_fixed = {
            name
            for name in edge.participants
            if name in variables and name not in mutable and name not in context
        }
        if missing_fixed:
            unsupported.append(edge.id)
            missing.update(missing_fixed)
            continue
        try:
            full_constraint = _compile_edge_formula(edge, variables, kinds)
            compiled_constraints[edge.id] = full_constraint
            is_hard = static_as_hard or edge.status in {
                DependencyStatus.CONFIRMED,
                DependencyStatus.ENVIRONMENT_SPECIFIC,
            }
            if is_hard:
                optimizer.add(full_constraint)
                hard_edges.append(edge.id)
            else:
                optimizer.add_soft(
                    full_constraint,
                    weight=str(max(1, round(edge.confidence * 100))),
                    id="static_dependencies",
                )
                soft_edges.append(edge.id)
            compiled.append(edge.id)
        except _CompileError as exc:
            unsupported.append(edge.id)
            missing.update(exc.missing)
        except (TypeError, z3.Z3Exception):
            unsupported.append(edge.id)

    _add_reference_objectives(
        optimizer,
        variables,
        kinds,
        mutable,
        context,
        missing,
    )

    result = optimizer.check()
    if result == z3.unsat:
        return SolverMutationPlan(
            status=SolveStatus.UNSAT,
            target_parameter=parameter,
            requested_value=value,
            mutable_parameters=tuple(sorted(mutable)),
            compiled_edges=tuple(compiled),
            hard_edges=tuple(hard_edges),
            soft_edges=tuple(soft_edges),
            unsupported_edges=tuple(unsupported),
            excluded_edges=tuple(excluded),
            out_of_scope_edges=tuple(out_of_scope),
            missing_context=tuple(sorted(missing)),
            reason="active compiled dependency constraints are unsatisfiable",
        )
    if result != z3.sat:
        return SolverMutationPlan(
            status=SolveStatus.UNKNOWN,
            target_parameter=parameter,
            requested_value=value,
            mutable_parameters=tuple(sorted(mutable)),
            compiled_edges=tuple(compiled),
            hard_edges=tuple(hard_edges),
            soft_edges=tuple(soft_edges),
            unsupported_edges=tuple(unsupported),
            excluded_edges=tuple(excluded),
            out_of_scope_edges=tuple(out_of_scope),
            missing_context=tuple(sorted(missing)),
            reason=optimizer.reason_unknown(),
        )

    model = optimizer.model()
    violated_soft = tuple(
        edge_id
        for edge_id in soft_edges
        if not z3.is_true(
            model.eval(compiled_constraints[edge_id], model_completion=True)
        )
    )
    changes: list[tuple[str, Any]] = []
    order = (parameter, *graph.affected_parameters(parameter))
    seen: set[str] = set()
    for name in order:
        if name in seen or name not in mutable or name not in variables:
            continue
        seen.add(name)
        solved = _model_value(model, variables[name], kinds[name])
        if name == parameter or name not in context or solved != context[name]:
            changes.append((name, solved))
    for name in sorted(mutable - seen):
        solved = _model_value(model, variables[name], kinds[name])
        if name == parameter or name not in context or solved != context[name]:
            changes.append((name, solved))

    return SolverMutationPlan(
        status=SolveStatus.SAT,
        target_parameter=parameter,
        requested_value=value,
        proposed_changes=tuple(changes),
        mutable_parameters=tuple(sorted(mutable)),
        compiled_edges=tuple(compiled),
        hard_edges=tuple(hard_edges),
        soft_edges=tuple(soft_edges),
        violated_soft_edges=violated_soft,
        unsupported_edges=tuple(unsupported),
        excluded_edges=tuple(excluded),
        out_of_scope_edges=tuple(out_of_scope),
        missing_context=tuple(sorted(missing)),
    )


def design_edge_intervention(
    graph: DependencyGraph,
    baseline: Mapping[str, Any],
    edge_id: str,
    *,
    include_repair: bool = True,
) -> InterventionPlan:
    if edge_id not in graph.edges:
        raise KeyError(f"unknown dependency edge: {edge_id}")
    edge = graph.edges[edge_id]
    context = normalize_context(graph, baseline)
    seed_mutable = {
        name
        for name in edge.participants
        if graph.nodes.get(name) is not None
        and graph.nodes[name].kind
        in {DependencyNodeKind.PARAMETER, DependencyNodeKind.FEATURE}
    }
    mutable = set(seed_mutable)
    for name in tuple(seed_mutable):
        mutable.update(
            affected
            for affected in graph.affected_parameters(name)
            if graph.nodes.get(affected) is not None
            and graph.nodes[affected].kind
            in {DependencyNodeKind.PARAMETER, DependencyNodeKind.FEATURE}
        )
    primary = next(
        (name for name in edge.dependents if name in seed_mutable),
        next((name for name in edge.participants if name in seed_mutable), ""),
    )
    if not primary:
        primary = edge.participants[0]

    intervention_id = _intervention_id(edge_id, context, edge.participants)
    satisfying = _solve_intervention_case(
        graph,
        edge_id,
        context,
        mutable,
        target_satisfied=True,
        role="satisfying",
    )
    violating = _solve_intervention_case(
        graph,
        edge_id,
        context,
        mutable,
        target_satisfied=False,
        role="violating",
    )
    repaired: InterventionCase | None = None
    if include_repair and violating.status is SolveStatus.SAT:
        repair_reference = dict(context)
        repair_reference.update(violating.assignment)
        repaired = _solve_intervention_case(
            graph,
            edge_id,
            repair_reference,
            mutable,
            target_satisfied=True,
            role="repaired",
        )
    return InterventionPlan(
        intervention_id=intervention_id,
        edge_id=edge.id,
        expression=edge.expression,
        primary_parameter=primary,
        mutable_parameters=tuple(sorted(mutable)),
        satisfying=satisfying,
        violating=violating,
        repaired=repaired,
        provenance=tuple(item.to_dict() for item in edge.evidence),
    )


def _solve_intervention_case(
    graph: DependencyGraph,
    edge_id: str,
    reference: Mapping[str, Any],
    mutable: set[str],
    *,
    target_satisfied: bool,
    role: str,
) -> InterventionCase:
    edge = graph.edges[edge_id]
    primary = next(iter(sorted(mutable)), edge.participants[0])
    primary_value = reference.get(primary)
    kinds = _infer_value_kinds(graph, reference, primary, primary_value)
    variables = {name: _make_variable(name, kind) for name, kind in kinds.items()}
    missing = {name for name in mutable if name not in reference or name not in variables}
    if missing:
        return InterventionCase(
            role=role,
            status=SolveStatus.UNKNOWN,
            missing_context=tuple(sorted(missing)),
            reason="mutable intervention parameters require typed baseline values",
        )

    optimizer = z3.Optimize()
    optimizer.set(priority="lex")
    for name, variable in variables.items():
        if name in mutable:
            continue
        if name not in reference:
            continue
        try:
            optimizer.add(variable == _literal(reference[name], kinds[name]))
        except _CompileError:
            missing.add(name)

    hard_edges = [edge_id]
    soft_edges: list[str] = []
    unsupported: list[str] = []
    compiled_soft: dict[str, z3.BoolRef] = {}
    try:
        target_formula = _compile_target_intervention(
            edge,
            variables,
            kinds,
            target_satisfied=target_satisfied,
        )
        optimizer.add(target_formula)
    except (_CompileError, TypeError, z3.Z3Exception) as exc:
        if isinstance(exc, _CompileError):
            missing.update(exc.missing)
        return InterventionCase(
            role=role,
            status=SolveStatus.UNKNOWN,
            hard_edges=(edge_id,),
            unsupported_edges=(edge_id,),
            missing_context=tuple(sorted(missing)),
            reason=f"target edge cannot be encoded exactly: {exc}",
        )

    for current in sorted(graph.edges.values(), key=lambda item: item.id):
        if current.id == edge_id or not mutable.intersection(current.participants):
            continue
        if current.status in {
            DependencyStatus.CONTRADICTED,
            DependencyStatus.SCOPE_DISPUTED,
        }:
            continue
        missing_fixed = {
            name
            for name in current.participants
            if name in variables and name not in mutable and name not in reference
        }
        if missing_fixed:
            unsupported.append(current.id)
            missing.update(missing_fixed)
            continue
        try:
            formula = _compile_edge_formula(current, variables, kinds)
        except _CompileError as exc:
            unsupported.append(current.id)
            missing.update(exc.missing)
            continue
        except (TypeError, z3.Z3Exception):
            unsupported.append(current.id)
            continue
        if current.status in {
            DependencyStatus.CONFIRMED,
            DependencyStatus.ENVIRONMENT_SPECIFIC,
        }:
            optimizer.add(formula)
            hard_edges.append(current.id)
        else:
            optimizer.add_soft(
                formula,
                weight=str(max(1, round(current.confidence * 100))),
                id="intervention_dependencies",
            )
            soft_edges.append(current.id)
            compiled_soft[current.id] = formula

    _add_reference_objectives(
        optimizer,
        variables,
        kinds,
        mutable,
        reference,
        missing,
    )

    result = optimizer.check()
    if result == z3.unsat:
        return InterventionCase(
            role=role,
            status=SolveStatus.UNSAT,
            hard_edges=tuple(hard_edges),
            soft_edges=tuple(soft_edges),
            unsupported_edges=tuple(unsupported),
            missing_context=tuple(sorted(missing)),
            reason="target polarity conflicts with the active hard constraints",
        )
    if result != z3.sat:
        return InterventionCase(
            role=role,
            status=SolveStatus.UNKNOWN,
            hard_edges=tuple(hard_edges),
            soft_edges=tuple(soft_edges),
            unsupported_edges=tuple(unsupported),
            missing_context=tuple(sorted(missing)),
            reason=optimizer.reason_unknown(),
        )

    model = optimizer.model()
    assignment = tuple(
        (name, _model_value(model, variables[name], kinds[name]))
        for name in sorted(mutable)
    )
    changes = tuple(
        (name, value)
        for name, value in assignment
        if name not in reference or reference[name] != value
    )
    violated_soft = tuple(
        current_id
        for current_id in soft_edges
        if not z3.is_true(model.eval(compiled_soft[current_id], model_completion=True))
    )
    return InterventionCase(
        role=role,
        status=SolveStatus.SAT,
        configuration=assignment,
        changes=changes,
        hard_edges=tuple(hard_edges),
        soft_edges=tuple(soft_edges),
        violated_soft_edges=violated_soft,
        unsupported_edges=tuple(unsupported),
        missing_context=tuple(sorted(missing)),
    )


def _compile_target_intervention(
    edge: DependencyEdge,
    variables: Mapping[str, z3.ExprRef],
    kinds: Mapping[str, _ValueKind],
    *,
    target_satisfied: bool,
) -> z3.BoolRef:
    predicate = _compile_text(edge.predicate, variables, kinds)
    predicate_value = _as_bool(predicate.value)
    side_constraints = list(predicate.side_constraints)
    if edge.guard is None:
        target = predicate_value if target_satisfied else z3.Not(predicate_value)
        return z3.And(*side_constraints, target)
    guard = _compile_text(edge.guard, variables, kinds)
    side_constraints.extend(guard.side_constraints)
    guard_value = _as_bool(guard.value)
    target = predicate_value if target_satisfied else z3.Not(predicate_value)
    return z3.And(*side_constraints, guard_value, target)


def _add_reference_objectives(
    optimizer: z3.Optimize,
    variables: Mapping[str, z3.ExprRef],
    kinds: Mapping[str, _ValueKind],
    names: set[str],
    reference: Mapping[str, Any],
    missing: set[str],
) -> None:
    change_terms: list[z3.ArithRef] = []
    distance_terms: list[z3.ArithRef] = []
    for name in sorted(names):
        if name not in reference:
            missing.add(name)
            continue
        variable = variables[name]
        try:
            original = _literal(reference[name], kinds[name])
        except _CompileError:
            missing.add(name)
            continue
        change_terms.append(z3.If(variable != original, 1, 0))
        if kinds[name] in {_ValueKind.INT, _ValueKind.REAL}:
            distance_terms.append(
                z3.If(variable >= original, variable - original, original - variable)
            )
    if change_terms:
        optimizer.minimize(z3.Sum(change_terms))
    if distance_terms:
        optimizer.minimize(z3.Sum(distance_terms))


def _compile_edge_formula(
    edge: DependencyEdge,
    variables: Mapping[str, z3.ExprRef],
    kinds: Mapping[str, _ValueKind],
) -> z3.BoolRef:
    predicate = _compile_text(edge.predicate, variables, kinds)
    constraint = _as_bool(predicate.value)
    side_constraints = list(predicate.side_constraints)
    if edge.guard is not None:
        guard = _compile_text(edge.guard, variables, kinds)
        side_constraints.extend(guard.side_constraints)
        constraint = z3.Implies(_as_bool(guard.value), constraint)
    return z3.And(*side_constraints, constraint)


def _intervention_id(
    edge_id: str,
    context: Mapping[str, Any],
    participants: tuple[str, ...],
) -> str:
    payload = {
        "edge_id": edge_id,
        "context": {name: context.get(name) for name in sorted(participants)},
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=repr).encode()
    ).hexdigest()[:20]
    return f"intervention-{digest}"


def normalize_context(
    graph: DependencyGraph,
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    flattened = _flatten_mapping(baseline)
    result: dict[str, Any] = dict(flattened)
    suffixes: dict[str, list[str]] = {}
    for path in flattened:
        suffixes.setdefault(path.rsplit(".", 1)[-1], []).append(path)
    for name in graph.nodes:
        if name in flattened:
            result[name] = flattened[name]
            continue
        matches = suffixes.get(name, ())
        if len(matches) == 1:
            result[name] = flattened[matches[0]]
    return result


def _flatten_mapping(
    value: Mapping[str, Any],
    prefix: str = "",
) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for raw_key, item in value.items():
        key = str(raw_key)
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(item, Mapping):
            flattened.update(_flatten_mapping(item, path))
        else:
            flattened[path] = item
    return flattened


def _infer_value_kinds(
    graph: DependencyGraph,
    context: Mapping[str, Any],
    parameter: str,
    value: Any,
) -> dict[str, _ValueKind]:
    explicit: dict[str, _ValueKind] = {}
    for edge in graph.edges.values():
        if edge.relation is not DependencyRelation.TYPE or ":" not in edge.predicate:
            continue
        name, raw_kind = (part.strip() for part in edge.predicate.split(":", 1))
        mapped = {
            "boolean": _ValueKind.BOOL,
            "integer": _ValueKind.INT,
            "number": _ValueKind.REAL,
            "float": _ValueKind.REAL,
            "string": _ValueKind.STRING,
        }.get(raw_kind.lower())
        if mapped is not None:
            explicit[name] = mapped

    kinds: dict[str, _ValueKind] = {}
    for name, node in graph.nodes.items():
        kind = explicit.get(name)
        if kind is None and name == parameter:
            kind = _kind_from_value(value)
        if kind is None and name in context:
            kind = _kind_from_value(context[name])
        if kind is None and node.kind is DependencyNodeKind.FEATURE:
            kind = _ValueKind.BOOL
        if kind is not None:
            kinds[name] = kind
    return kinds


def _kind_from_value(value: Any) -> _ValueKind | None:
    if isinstance(value, bool):
        return _ValueKind.BOOL
    if isinstance(value, int):
        return _ValueKind.INT
    if isinstance(value, float):
        return _ValueKind.REAL
    if isinstance(value, str):
        return _ValueKind.STRING
    return None


def _make_variable(name: str, kind: _ValueKind) -> z3.ExprRef:
    if kind is _ValueKind.BOOL:
        return z3.Bool(name)
    if kind is _ValueKind.INT:
        return z3.Int(name)
    if kind is _ValueKind.REAL:
        return z3.Real(name)
    return z3.String(name)


def _literal(value: Any, kind: _ValueKind) -> z3.ExprRef:
    if kind is _ValueKind.BOOL and isinstance(value, bool):
        return z3.BoolVal(value)
    if kind is _ValueKind.INT and isinstance(value, int) and not isinstance(value, bool):
        return z3.IntVal(value)
    if kind is _ValueKind.REAL and isinstance(value, (int, float)) and not isinstance(value, bool):
        return z3.RealVal(str(value))
    if kind is _ValueKind.STRING and isinstance(value, str):
        return z3.StringVal(value)
    raise _CompileError(f"value {value!r} does not match solver type {kind.value}")


def _compile_text(
    expression: str,
    variables: Mapping[str, z3.ExprRef],
    kinds: Mapping[str, _ValueKind],
) -> _CompiledExpression:
    if ":" in expression and "=>" not in expression:
        name, _ = (part.strip() for part in expression.split(":", 1))
        if name in variables:
            return _CompiledExpression(z3.BoolVal(True))
    try:
        node = ast.parse(expression, mode="eval").body
    except SyntaxError as exc:
        raise _CompileError(f"unsupported expression syntax: {expression}") from exc
    return _compile_ast(node, variables, kinds)


def _compile_ast(
    node: ast.AST,
    variables: Mapping[str, z3.ExprRef],
    kinds: Mapping[str, _ValueKind],
) -> _CompiledExpression:
    if isinstance(node, ast.Constant):
        if node.value is None:
            raise _CompileError("None-valued constraints are not solver-encoded")
        kind = _kind_from_value(node.value)
        if kind is None:
            raise _CompileError(f"unsupported literal: {node.value!r}")
        return _CompiledExpression(_literal(node.value, kind))
    if isinstance(node, ast.Name):
        if node.id in variables:
            return _CompiledExpression(variables[node.id])
        raise _CompileError(f"missing solver symbol: {node.id}", {node.id})
    if isinstance(node, ast.Attribute):
        path = _attribute_path(node)
        if path is not None and path in variables:
            return _CompiledExpression(variables[path])
        raise _CompileError(f"missing solver symbol: {path or node.attr}", {path or node.attr})
    if isinstance(node, ast.UnaryOp):
        operand = _compile_ast(node.operand, variables, kinds)
        if isinstance(node.op, ast.Not):
            value = z3.Not(_as_bool(operand.value))
        elif isinstance(node.op, ast.USub):
            value = -operand.value
        elif isinstance(node.op, ast.UAdd):
            value = operand.value
        else:
            raise _CompileError(f"unsupported unary operator: {type(node.op).__name__}")
        return _CompiledExpression(value, operand.side_constraints)
    if isinstance(node, ast.BoolOp):
        values = [_compile_ast(item, variables, kinds) for item in node.values]
        bools = [_as_bool(item.value) for item in values]
        value = z3.And(*bools) if isinstance(node.op, ast.And) else z3.Or(*bools)
        return _CompiledExpression(value, _merge_side(values))
    if isinstance(node, ast.BinOp):
        left = _compile_ast(node.left, variables, kinds)
        right = _compile_ast(node.right, variables, kinds)
        side = [*left.side_constraints, *right.side_constraints]
        if isinstance(node.op, ast.Add):
            value = left.value + right.value
        elif isinstance(node.op, ast.Sub):
            value = left.value - right.value
        elif isinstance(node.op, ast.Mult):
            value = left.value * right.value
        elif isinstance(node.op, (ast.Div, ast.FloorDiv)):
            side.append(right.value != 0)
            value = left.value / right.value
        elif isinstance(node.op, ast.Mod):
            side.append(right.value != 0)
            value = left.value % right.value
        elif isinstance(node.op, ast.Pow):
            value = left.value**right.value
        else:
            raise _CompileError(f"unsupported binary operator: {type(node.op).__name__}")
        return _CompiledExpression(value, tuple(side))
    if isinstance(node, ast.Compare):
        left_compiled = _compile_ast(node.left, variables, kinds)
        side_constraints = list(left_compiled.side_constraints)
        clauses: list[z3.BoolRef] = []
        left = left_compiled.value
        for index, op in enumerate(node.ops):
            right_node = node.comparators[index]
            if (
                isinstance(op, (ast.Is, ast.IsNot))
                and isinstance(right_node, ast.Constant)
                and right_node.value is None
            ):
                clauses.append(z3.BoolVal(isinstance(op, ast.IsNot)))
                if index + 1 < len(node.ops):
                    raise _CompileError("chained None comparisons are unsupported")
                continue
            if isinstance(op, (ast.In, ast.NotIn)):
                choices = _collection_items(right_node, variables, kinds)
                clause = z3.Or(*(left == item for item in choices))
                clauses.append(z3.Not(clause) if isinstance(op, ast.NotIn) else clause)
                if index + 1 < len(node.ops):
                    raise _CompileError("chained membership comparisons are unsupported")
                continue
            right_compiled = _compile_ast(right_node, variables, kinds)
            side_constraints.extend(right_compiled.side_constraints)
            right = right_compiled.value
            if isinstance(op, ast.Eq):
                clauses.append(left == right)
            elif isinstance(op, ast.NotEq):
                clauses.append(left != right)
            elif isinstance(op, ast.Lt):
                clauses.append(left < right)
            elif isinstance(op, ast.LtE):
                clauses.append(left <= right)
            elif isinstance(op, ast.Gt):
                clauses.append(left > right)
            elif isinstance(op, ast.GtE):
                clauses.append(left >= right)
            elif isinstance(op, (ast.Is, ast.IsNot)):
                clause = left == right
                clauses.append(z3.Not(clause) if isinstance(op, ast.IsNot) else clause)
            else:
                raise _CompileError(f"unsupported comparator: {type(op).__name__}")
            left = right
        return _CompiledExpression(z3.And(*clauses), tuple(side_constraints))
    if isinstance(node, ast.IfExp):
        test = _compile_ast(node.test, variables, kinds)
        body = _compile_ast(node.body, variables, kinds)
        other = _compile_ast(node.orelse, variables, kinds)
        return _CompiledExpression(
            z3.If(_as_bool(test.value), body.value, other.value),
            _merge_side((test, body, other)),
        )
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        arguments = [_compile_ast(item, variables, kinds) for item in node.args]
        if node.func.id == "abs" and len(arguments) == 1:
            value = z3.If(arguments[0].value >= 0, arguments[0].value, -arguments[0].value)
        elif node.func.id in {"max", "min"} and len(arguments) >= 2:
            value = arguments[0].value
            for item in arguments[1:]:
                condition = value >= item.value if node.func.id == "max" else value <= item.value
                value = z3.If(condition, value, item.value)
        elif node.func.id in {"int", "float", "bool", "str"} and len(arguments) == 1:
            value = arguments[0].value
        else:
            raise _CompileError(f"unsupported call: {node.func.id}")
        return _CompiledExpression(value, _merge_side(arguments))
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        raise _CompileError("collection literals are supported only in membership comparisons")
    raise _CompileError(f"unsupported AST node: {type(node).__name__}")


def _collection_items(
    node: ast.AST,
    variables: Mapping[str, z3.ExprRef],
    kinds: Mapping[str, _ValueKind],
) -> tuple[z3.ExprRef, ...]:
    if not isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        raise _CompileError("membership requires a literal collection")
    return tuple(_compile_ast(item, variables, kinds).value for item in node.elts)


def _merge_side(items: Any) -> tuple[z3.BoolRef, ...]:
    return tuple(side for item in items for side in item.side_constraints)


def _as_bool(value: z3.ExprRef) -> z3.BoolRef:
    if z3.is_bool(value):
        return value
    raise _CompileError("constraint expression is not Boolean")


def _attribute_path(node: ast.Attribute) -> str | None:
    parts = [node.attr]
    current: ast.AST = node.value
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    return ".".join(reversed(parts))


def _model_value(
    model: z3.ModelRef,
    variable: z3.ExprRef,
    kind: _ValueKind,
) -> Any:
    value = model.eval(variable, model_completion=True)
    if kind is _ValueKind.BOOL:
        return z3.is_true(value)
    if kind is _ValueKind.INT:
        return value.as_long()
    if kind is _ValueKind.REAL:
        if isinstance(value, z3.RatNumRef):
            numerator = value.numerator_as_long()
            denominator = value.denominator_as_long()
            return numerator / denominator
        return float(value.as_decimal(20).rstrip("?"))
    return value.as_string()
