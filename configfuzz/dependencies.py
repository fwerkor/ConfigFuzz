from __future__ import annotations

import ast
import hashlib
import math
import operator
from collections import Counter, deque
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from configfuzz.model import Constraint, ConstraintKind, ConstraintSet, Evidence


class DependencyNodeKind(str, Enum):
    PARAMETER = "parameter"
    DERIVED = "derived"
    FEATURE = "feature"
    ENVIRONMENT = "environment"


class DependencyRelation(str, Enum):
    TYPE = "type"
    RANGE = "range"
    ENUM = "enum"
    REQUIRES = "requires"
    CONFLICTS = "conflicts"
    DIVISIBILITY = "divisibility"
    ALIGNMENT = "alignment"
    BOUND = "bound"
    EQUALITY = "equality"
    PRODUCT_LIMIT = "product_limit"
    CONDITIONAL = "conditional"
    ENVIRONMENT = "environment"
    RESOURCE = "resource"
    OTHER = "other"


class DependencyStatus(str, Enum):
    STATIC_CANDIDATE = "static_candidate"
    DYNAMICALLY_SUPPORTED = "dynamically_supported"
    CONFIRMED = "confirmed"
    ENVIRONMENT_SPECIFIC = "environment_specific"
    SCOPE_DISPUTED = "scope_disputed"
    CONTRADICTED = "contradicted"


@dataclass(frozen=True, slots=True)
class DependencyNode:
    name: str
    kind: DependencyNodeKind

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "kind": self.kind.value}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DependencyNode":
        return cls(
            name=str(payload["name"]),
            kind=DependencyNodeKind(str(payload["kind"])),
        )


@dataclass(frozen=True, slots=True)
class DependencyEdge:
    id: str
    expression: str
    predicate: str
    relation: DependencyRelation
    participants: tuple[str, ...]
    drivers: tuple[str, ...]
    dependents: tuple[str, ...]
    guard: str | None = None
    status: DependencyStatus = DependencyStatus.STATIC_CANDIDATE
    confidence: float = 1.0
    scope: tuple[tuple[str, str], ...] = ()
    components: tuple[str, ...] = ()
    evidence: tuple[Evidence, ...] = ()

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("dependency edge id must not be empty")
        if not self.expression.strip():
            raise ValueError("dependency expression must not be empty")
        if not self.participants:
            raise ValueError("dependency edge must have at least one participant")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("dependency confidence must be in [0, 1]")
        participants = tuple(dict.fromkeys(self.participants))
        object.__setattr__(self, "participants", participants)
        object.__setattr__(
            self,
            "drivers",
            tuple(name for name in dict.fromkeys(self.drivers) if name in participants),
        )
        object.__setattr__(
            self,
            "dependents",
            tuple(name for name in dict.fromkeys(self.dependents) if name in participants),
        )
        object.__setattr__(self, "components", tuple(dict.fromkeys(self.components)))
        object.__setattr__(self, "scope", tuple(sorted(dict(self.scope).items())))

    @property
    def scope_dict(self) -> dict[str, str]:
        return dict(self.scope)

    def merge(self, other: "DependencyEdge") -> "DependencyEdge":
        if self.id != other.id:
            raise ValueError("only identical dependency edges can be merged")
        return replace(
            self,
            participants=tuple(dict.fromkeys((*self.participants, *other.participants))),
            drivers=tuple(dict.fromkeys((*self.drivers, *other.drivers))),
            dependents=tuple(dict.fromkeys((*self.dependents, *other.dependents))),
            confidence=max(self.confidence, other.confidence),
            components=tuple(dict.fromkeys((*self.components, *other.components))),
            evidence=tuple(dict.fromkeys((*self.evidence, *other.evidence))),
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "expression": self.expression,
            "predicate": self.predicate,
            "relation": self.relation.value,
            "participants": list(self.participants),
            "drivers": list(self.drivers),
            "dependents": list(self.dependents),
            "status": self.status.value,
            "confidence": self.confidence,
            "scope": self.scope_dict,
            "components": list(self.components),
            "evidence": [item.to_dict() for item in self.evidence],
        }
        if self.guard is not None:
            payload["guard"] = self.guard
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DependencyEdge":
        evidence = tuple(Evidence.from_dict(item) for item in payload.get("evidence", ()))
        scope = payload.get("scope", {})
        if not isinstance(scope, Mapping):
            raise ValueError("dependency edge scope must be an object")
        return cls(
            id=str(payload["id"]),
            expression=str(payload["expression"]),
            predicate=str(payload.get("predicate", payload["expression"])),
            relation=DependencyRelation(str(payload["relation"])),
            participants=tuple(str(item) for item in payload.get("participants", ())),
            drivers=tuple(str(item) for item in payload.get("drivers", ())),
            dependents=tuple(str(item) for item in payload.get("dependents", ())),
            guard=str(payload["guard"]) if payload.get("guard") is not None else None,
            status=DependencyStatus(
                str(payload.get("status", DependencyStatus.STATIC_CANDIDATE.value))
            ),
            confidence=float(payload.get("confidence", 1.0)),
            scope=tuple((str(key), str(value)) for key, value in scope.items()),
            components=tuple(str(item) for item in payload.get("components", ())),
            evidence=evidence,
        )


@dataclass(frozen=True, slots=True)
class EdgeEvaluation:
    edge_id: str
    active: bool | None
    satisfied: bool | None
    missing_context: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "active": self.active,
            "satisfied": self.satisfied,
            "missing_context": list(self.missing_context),
        }


@dataclass(frozen=True, slots=True)
class MutationPlan:
    target_parameter: str
    requested_value: Any
    proposed_changes: tuple[tuple[str, Any], ...]
    affected_parameters: tuple[str, ...]
    validation_order: tuple[str, ...]
    impacted_edges: tuple[str, ...]
    violated_edges: tuple[str, ...]
    unresolved_edges: tuple[str, ...]
    missing_context: tuple[str, ...]

    @property
    def changes(self) -> dict[str, Any]:
        return dict(self.proposed_changes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_parameter": self.target_parameter,
            "requested_value": self.requested_value,
            "proposed_changes": self.changes,
            "affected_parameters": list(self.affected_parameters),
            "validation_order": list(self.validation_order),
            "impacted_edges": list(self.impacted_edges),
            "violated_edges": list(self.violated_edges),
            "unresolved_edges": list(self.unresolved_edges),
            "missing_context": list(self.missing_context),
        }


@dataclass(slots=True)
class DependencyGraph:
    nodes: dict[str, DependencyNode] = field(default_factory=dict)
    edges: dict[str, DependencyEdge] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_constraint_sets(
        cls,
        constraint_sets: Iterable[ConstraintSet],
        *,
        scope: Mapping[str, str] | None = None,
    ) -> "DependencyGraph":
        sets = list(constraint_sets)
        constraints = [
            constraint
            for item in sets
            for constraint in item.constraints
        ]
        known_parameters = {
            name
            for constraint in constraints
            for name in _constraint_participants(constraint)
        }
        known_parameters.update(item.parameter for item in sets)
        boolean_parameters = _boolean_parameters(sets)
        graph = cls(metadata={"scope": dict(scope or {})})
        for constraint in constraints:
            edge = _edge_from_constraint(
                constraint,
                known_parameters=known_parameters,
                scope=scope or {},
            )
            existing = graph.edges.get(edge.id)
            graph.edges[edge.id] = edge if existing is None else existing.merge(edge)

        for edge in graph.edges.values():
            for name in edge.participants:
                kind = _node_kind(
                    name,
                    boolean_parameters=boolean_parameters,
                    environment_edge=edge.relation
                    in {DependencyRelation.ENVIRONMENT, DependencyRelation.RESOURCE},
                )
                existing = graph.nodes.get(name)
                if existing is None or (
                    existing.kind is DependencyNodeKind.PARAMETER
                    and kind is not DependencyNodeKind.PARAMETER
                ):
                    graph.nodes[name] = DependencyNode(name=name, kind=kind)
        return graph

    @classmethod
    def from_scan_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        scope: Mapping[str, str] | None = None,
    ) -> "DependencyGraph":
        results = payload.get("results")
        if not isinstance(results, Sequence):
            raise ValueError("scan payload must contain a results array")
        constraint_sets = [ConstraintSet.from_dict(item) for item in results]
        inferred_scope: dict[str, str] = {}
        framework = payload.get("framework")
        if isinstance(framework, Mapping):
            for source_key, target_key in (
                ("name", "framework"),
                ("commit", "version"),
                ("source_subdir", "source_subdir"),
                ("profile", "framework_profile"),
                ("backend", "backend"),
                ("accelerator", "accelerator"),
            ):
                value = framework.get(source_key)
                if value is not None:
                    inferred_scope[target_key] = str(value)
            if "source_subdir" not in inferred_scope:
                source_subdirs = framework.get("source_subdirs")
                if isinstance(source_subdirs, Sequence) and not isinstance(source_subdirs, (str, bytes)):
                    inferred_scope["source_subdir"] = ",".join(str(item) for item in source_subdirs)
            if "version" not in inferred_scope:
                repositories = framework.get("repositories")
                if isinstance(repositories, Sequence) and not isinstance(repositories, (str, bytes)):
                    commits = [
                        str(item["commit"])
                        for item in repositories
                        if isinstance(item, Mapping) and item.get("commit") is not None
                    ]
                    if commits:
                        inferred_scope["version"] = "+".join(commits)
        inferred_scope.update(dict(scope or {}))
        return cls.from_constraint_sets(constraint_sets, scope=inferred_scope)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DependencyGraph":
        graph_payload: Mapping[str, Any] = payload
        nested = payload.get("dependency_graph")
        if isinstance(nested, Mapping):
            graph_payload = nested
        else:
            active_validation = payload.get("active_validation")
            if isinstance(active_validation, Mapping):
                nested = active_validation.get("dependency_graph")
                if isinstance(nested, Mapping):
                    graph_payload = nested
        nodes = {
            node.name: node
            for item in graph_payload.get("nodes", ())
            for node in (DependencyNode.from_dict(item),)
        }
        edges = {
            edge.id: edge
            for item in graph_payload.get("edges", ())
            for edge in (DependencyEdge.from_dict(item),)
        }
        metadata = graph_payload.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError("dependency graph metadata must be an object")
        return cls(nodes=nodes, edges=edges, metadata=dict(metadata))

    def to_dict(self) -> dict[str, Any]:
        components = self.connected_components()
        relation_counts = Counter(edge.relation.value for edge in self.edges.values())
        status_counts = Counter(edge.status.value for edge in self.edges.values())
        node_counts = Counter(node.kind.value for node in self.nodes.values())
        return {
            "schema_version": 1,
            "metadata": self.metadata,
            "summary": {
                "nodes": len(self.nodes),
                "edges": len(self.edges),
                "connected_components": len(components),
                "node_kinds": dict(sorted(node_counts.items())),
                "relations": dict(sorted(relation_counts.items())),
                "statuses": dict(sorted(status_counts.items())),
            },
            "nodes": [self.nodes[name].to_dict() for name in sorted(self.nodes)],
            "edges": [self.edges[edge_id].to_dict() for edge_id in sorted(self.edges)],
        }

    def add_edge(self, edge: DependencyEdge) -> None:
        existing = self.edges.get(edge.id)
        self.edges[edge.id] = edge if existing is None else existing.merge(edge)
        for name in edge.participants:
            self.nodes.setdefault(
                name,
                DependencyNode(name=name, kind=DependencyNodeKind.PARAMETER),
            )

    def update_status(self, edge_id: str, status: DependencyStatus) -> None:
        edge = self.edges[edge_id]
        self.edges[edge_id] = replace(edge, status=status)

    def constraints_for(self, parameter: str) -> tuple[DependencyEdge, ...]:
        return tuple(
            edge
            for edge in self.edges.values()
            if parameter in edge.participants
        )

    def related_parameters(self, parameter: str) -> tuple[str, ...]:
        related = {
            name
            for edge in self.constraints_for(parameter)
            for name in edge.participants
            if name != parameter
        }
        return tuple(sorted(related))

    def affected_parameters(
        self,
        parameter: str,
        *,
        transitive: bool = True,
    ) -> tuple[str, ...]:
        adjacency = self._directed_adjacency()
        if not transitive:
            return tuple(sorted(adjacency.get(parameter, ())))
        visited: set[str] = set()
        queue = deque(sorted(adjacency.get(parameter, ())))
        while queue:
            current = queue.popleft()
            if current == parameter or current in visited:
                continue
            visited.add(current)
            queue.extend(sorted(adjacency.get(current, ())))
        return tuple(sorted(visited))

    def connected_components(self) -> tuple[tuple[str, ...], ...]:
        adjacency: dict[str, set[str]] = {name: set() for name in self.nodes}
        for edge in self.edges.values():
            for left in edge.participants:
                adjacency.setdefault(left, set()).update(
                    right for right in edge.participants if right != left
                )
        remaining = set(adjacency)
        components: list[tuple[str, ...]] = []
        while remaining:
            start = min(remaining)
            queue = deque([start])
            component: set[str] = set()
            while queue:
                current = queue.popleft()
                if current in component:
                    continue
                component.add(current)
                queue.extend(sorted(adjacency.get(current, ())))
            remaining.difference_update(component)
            components.append(tuple(sorted(component)))
        return tuple(sorted(components, key=lambda item: (item[0], len(item))))

    def evaluate_edge(
        self,
        edge: str | DependencyEdge,
        context: Mapping[str, Any],
    ) -> EdgeEvaluation:
        item = self.edges[edge] if isinstance(edge, str) else edge
        missing: set[str] = set()
        active: bool | None = True
        if item.guard is not None:
            guard_result = _evaluate_text(item.guard, context)
            missing.update(guard_result.missing)
            active = bool(guard_result.value) if guard_result.known else None
        if active is False:
            return EdgeEvaluation(
                edge_id=item.id,
                active=False,
                satisfied=True,
                missing_context=tuple(sorted(missing)),
            )
        predicate_result = (
            _evaluate_type_constraint(item.predicate, context)
            if item.relation is DependencyRelation.TYPE
            else _evaluate_text(item.predicate, context)
        )
        missing.update(predicate_result.missing)
        satisfied = bool(predicate_result.value) if predicate_result.known else None
        return EdgeEvaluation(
            edge_id=item.id,
            active=active,
            satisfied=satisfied,
            missing_context=tuple(sorted(missing)),
        )

    def active_constraints(
        self,
        context: Mapping[str, Any],
        *,
        include_unknown: bool = True,
    ) -> tuple[DependencyEdge, ...]:
        active: list[DependencyEdge] = []
        for edge in self.edges.values():
            evaluation = self.evaluate_edge(edge, context)
            if evaluation.active is True or (
                include_unknown and evaluation.active is None
            ):
                active.append(edge)
        return tuple(sorted(active, key=lambda item: item.id))

    def plan_joint_mutation(
        self,
        parameter: str,
        value: Any,
        baseline: Mapping[str, Any],
    ) -> MutationPlan:
        if parameter not in self.nodes:
            raise KeyError(f"unknown dependency node: {parameter}")
        working = dict(baseline)
        working[parameter] = value
        proposed: dict[str, Any] = {parameter: value}
        affected = self.affected_parameters(parameter)
        relevant_nodes = {parameter, *affected}
        impacted = {
            edge.id: edge
            for edge in self.edges.values()
            if relevant_nodes.intersection(edge.participants)
        }

        for _ in range(3):
            changed = False
            for edge in sorted(impacted.values(), key=lambda item: item.id):
                evaluation = self.evaluate_edge(edge, working)
                if evaluation.active is False or evaluation.satisfied is not False:
                    continue
                repairs = self._repair_edge(edge, working, fixed_parameter=parameter)
                for name, repaired_value in repairs.items():
                    if name == parameter or working.get(name) == repaired_value:
                        continue
                    working[name] = repaired_value
                    proposed[name] = repaired_value
                    changed = True
            if not changed:
                break

        violated: list[str] = []
        unresolved: list[str] = []
        missing: set[str] = set()
        for edge in sorted(impacted.values(), key=lambda item: item.id):
            evaluation = self.evaluate_edge(edge, working)
            missing.update(evaluation.missing_context)
            if evaluation.active is False:
                continue
            if evaluation.satisfied is False:
                violated.append(edge.id)
            elif evaluation.active is None or evaluation.satisfied is None:
                unresolved.append(edge.id)

        validation_order = self._validation_order(parameter, affected)
        changes = tuple(
            (name, proposed[name])
            for name in validation_order
            if name in proposed
        )
        for name in sorted(proposed):
            if name not in dict(changes):
                changes = (*changes, (name, proposed[name]))
        return MutationPlan(
            target_parameter=parameter,
            requested_value=value,
            proposed_changes=changes,
            affected_parameters=affected,
            validation_order=validation_order,
            impacted_edges=tuple(sorted(impacted)),
            violated_edges=tuple(violated),
            unresolved_edges=tuple(unresolved),
            missing_context=tuple(sorted(missing)),
        )

    def _directed_adjacency(self) -> dict[str, set[str]]:
        adjacency: dict[str, set[str]] = {name: set() for name in self.nodes}
        for edge in self.edges.values():
            if edge.drivers and edge.dependents:
                for driver in edge.drivers:
                    adjacency.setdefault(driver, set()).update(
                        item for item in edge.dependents if item != driver
                    )
                continue
            for source in edge.participants:
                adjacency.setdefault(source, set()).update(
                    target for target in edge.participants if target != source
                )
        return adjacency

    def _validation_order(
        self,
        parameter: str,
        affected: Sequence[str],
    ) -> tuple[str, ...]:
        adjacency = self._directed_adjacency()
        order: list[str] = []
        seen: set[str] = set()
        queue = deque([parameter])
        allowed = {parameter, *affected}
        while queue:
            current = queue.popleft()
            if current in seen or current not in allowed:
                continue
            seen.add(current)
            order.append(current)
            queue.extend(sorted(adjacency.get(current, ())))
        order.extend(sorted(allowed - seen))
        return tuple(order)

    def _repair_edge(
        self,
        edge: DependencyEdge,
        context: Mapping[str, Any],
        *,
        fixed_parameter: str,
    ) -> dict[str, Any]:
        try:
            predicate = ast.parse(edge.predicate, mode="eval").body
        except SyntaxError:
            return {}
        if isinstance(predicate, ast.Name):
            return self._repair_name(predicate.id, True, fixed_parameter)
        if (
            isinstance(predicate, ast.UnaryOp)
            and isinstance(predicate.op, ast.Not)
            and isinstance(predicate.operand, ast.Name)
        ):
            return self._repair_name(predicate.operand.id, False, fixed_parameter)
        if not (
            isinstance(predicate, ast.Compare)
            and len(predicate.ops) == 1
            and len(predicate.comparators) == 1
        ):
            return {}

        comparator = predicate.comparators[0]
        op = predicate.ops[0]
        if (
            isinstance(predicate.left, ast.BinOp)
            and isinstance(predicate.left.op, ast.Mod)
            and isinstance(op, ast.Eq)
            and isinstance(comparator, ast.Constant)
            and comparator.value == 0
            and isinstance(predicate.left.left, ast.Name)
        ):
            dependent = predicate.left.left.id
            if dependent == fixed_parameter or not self._repairable_node(dependent):
                return {}
            divisor_result = _evaluate_ast(predicate.left.right, context)
            current = context.get(dependent)
            if (
                not divisor_result.known
                or not isinstance(divisor_result.value, int)
                or divisor_result.value <= 0
                or not isinstance(current, int)
            ):
                return {}
            divisor = divisor_result.value
            repaired = max(divisor, math.ceil(max(current, 1) / divisor) * divisor)
            return {dependent: repaired}

        if not isinstance(predicate.left, ast.Name):
            return {}
        dependent = predicate.left.id
        if dependent == fixed_parameter or not self._repairable_node(dependent):
            return {}
        right = _evaluate_ast(comparator, context)
        if not right.known:
            return {}
        current = context.get(dependent)
        repaired = _repair_comparison_value(current, right.value, op)
        return {} if repaired is _NO_REPAIR else {dependent: repaired}

    def _repair_name(
        self,
        name: str,
        value: bool,
        fixed_parameter: str,
    ) -> dict[str, Any]:
        if name == fixed_parameter or not self._repairable_node(name):
            return {}
        return {name: value}

    def _repairable_node(self, name: str) -> bool:
        node = self.nodes.get(name)
        return node is not None and node.kind in {
            DependencyNodeKind.PARAMETER,
            DependencyNodeKind.FEATURE,
        }


@dataclass(frozen=True, slots=True)
class _EvaluationResult:
    value: Any = None
    known: bool = False
    missing: frozenset[str] = frozenset()


_NO_REPAIR = object()


def _edge_from_constraint(
    constraint: Constraint,
    *,
    known_parameters: set[str],
    scope: Mapping[str, str],
) -> DependencyEdge:
    guard, predicate = _split_implication(constraint.expression)
    participants = _constraint_participants(constraint)
    relation = _classify_relation(constraint.kind, guard, predicate)
    drivers, dependents = _infer_direction(
        guard,
        predicate,
        participants,
        known_parameters=known_parameters,
    )
    status = (
        DependencyStatus.ENVIRONMENT_SPECIFIC
        if relation in {DependencyRelation.ENVIRONMENT, DependencyRelation.RESOURCE}
        else DependencyStatus.STATIC_CANDIDATE
    )
    components = tuple(
        dict.fromkeys(
            component
            for item in constraint.evidence
            for component in (_component_from_source(item.source),)
            if component
        )
    )
    edge_id = _edge_id(constraint.expression, constraint.kind)
    return DependencyEdge(
        id=edge_id,
        expression=constraint.expression,
        predicate=predicate,
        guard=guard,
        relation=relation,
        participants=participants,
        drivers=drivers,
        dependents=dependents,
        status=status,
        confidence=constraint.confidence,
        scope=tuple((str(key), str(value)) for key, value in scope.items()),
        components=components,
        evidence=constraint.evidence,
    )


def _edge_id(expression: str, kind: ConstraintKind) -> str:
    digest = hashlib.sha256(f"{kind.value}\0{expression}".encode()).hexdigest()[:16]
    return f"dep-{digest}"


def _split_implication(expression: str) -> tuple[str | None, str]:
    if "=>" not in expression:
        return None, expression.strip()
    guard, predicate = expression.split("=>", 1)
    return guard.strip(), predicate.strip()


def _classify_relation(
    kind: ConstraintKind,
    guard: str | None,
    predicate: str,
) -> DependencyRelation:
    if kind is ConstraintKind.TYPE:
        return DependencyRelation.TYPE
    if kind is ConstraintKind.RANGE:
        return DependencyRelation.RANGE
    if kind is ConstraintKind.ENUM:
        return DependencyRelation.ENUM
    if kind is ConstraintKind.ENVIRONMENT:
        return DependencyRelation.ENVIRONMENT
    if kind is ConstraintKind.RESOURCE:
        return DependencyRelation.RESOURCE
    try:
        node = ast.parse(predicate, mode="eval").body
    except SyntaxError:
        return DependencyRelation.CONDITIONAL if guard else DependencyRelation.OTHER
    if guard is not None:
        guard_features = {
            name
            for name in _symbols_in_text(guard)
            if _looks_feature_name(name)
        }
        if isinstance(node, ast.Name):
            return DependencyRelation.REQUIRES
        if (
            isinstance(node, ast.UnaryOp)
            and isinstance(node.op, ast.Not)
            and isinstance(node.operand, ast.Name)
        ):
            return DependencyRelation.CONFLICTS
        if guard_features:
            return DependencyRelation.REQUIRES
    if _is_mod_zero(node):
        assert isinstance(node, ast.Compare)
        modulo = node.left
        assert isinstance(modulo, ast.BinOp)
        if isinstance(modulo.right, ast.Constant):
            return DependencyRelation.ALIGNMENT
        return DependencyRelation.DIVISIBILITY
    if isinstance(node, ast.Compare):
        if any(isinstance(op, (ast.In, ast.NotIn)) for op in node.ops):
            return DependencyRelation.ENUM
        if any(isinstance(op, (ast.Eq, ast.NotEq, ast.Is, ast.IsNot)) for op in node.ops):
            return DependencyRelation.EQUALITY
        if _looks_product_limit(node):
            return DependencyRelation.PRODUCT_LIMIT
        return DependencyRelation.BOUND
    if guard is not None:
        return DependencyRelation.REQUIRES
    return DependencyRelation.OTHER


def _infer_direction(
    guard: str | None,
    predicate: str,
    participants: Sequence[str],
    *,
    known_parameters: set[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    participant_set = set(participants)
    try:
        node = ast.parse(predicate, mode="eval").body
    except SyntaxError:
        return (), ()
    predicate_drivers, predicate_dependents = _predicate_direction(
        node,
        participant_set,
    )
    if guard is not None:
        guard_names = _symbols_in_text(guard) & participant_set
        predicate_names = _symbols_in_text(predicate) & participant_set
        if predicate_dependents:
            drivers = predicate_drivers | (guard_names - predicate_dependents)
            return tuple(sorted(drivers)), tuple(sorted(predicate_dependents))
        dependents = predicate_names - guard_names
        if guard_names and dependents:
            return tuple(sorted(guard_names)), tuple(sorted(dependents))
    if predicate_drivers or predicate_dependents:
        return tuple(sorted(predicate_drivers)), tuple(sorted(predicate_dependents))
    parameter_names = tuple(name for name in participants if name in known_parameters)
    if len(parameter_names) == 1:
        return (), parameter_names
    return (), ()


def _predicate_direction(
    node: ast.AST,
    participants: set[str],
) -> tuple[set[str], set[str]]:
    if _is_mod_zero(node) and isinstance(node, ast.Compare):
        modulo = node.left
        assert isinstance(modulo, ast.BinOp)
        drivers = _symbols_in_ast(modulo.right) & participants
        dependents = _symbols_in_ast(modulo.left) & participants
        return drivers, dependents
    if isinstance(node, ast.Name) and node.id in participants:
        return set(), {node.id}
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.Not)
        and isinstance(node.operand, ast.Name)
        and node.operand.id in participants
    ):
        return set(), {node.operand.id}
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and len(node.comparators) == 1:
        left_names = _symbols_in_ast(node.left) & participants
        right_names = _symbols_in_ast(node.comparators[0]) & participants
        if left_names and right_names:
            return right_names, left_names
        if left_names:
            return set(), left_names
        if right_names:
            return set(), right_names
    return set(), set()


def _boolean_parameters(constraint_sets: Iterable[ConstraintSet]) -> set[str]:
    result: set[str] = set()
    for item in constraint_sets:
        for constraint in item.constraints:
            expression = constraint.expression.strip()
            if expression == f"{item.parameter}: boolean":
                result.add(item.parameter)
            guard, predicate = _split_implication(expression)
            for text in (guard, predicate):
                if text is None:
                    continue
                try:
                    node = ast.parse(text, mode="eval").body
                except SyntaxError:
                    continue
                result.update(_bare_boolean_names(node))
    return result


def _bare_boolean_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    if isinstance(node, ast.Name):
        names.add(node.id)
    elif isinstance(node, ast.Attribute):
        path = _attribute_path(node)
        if path is not None:
            names.add(path)
    elif (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.Not)
        and isinstance(node.operand, ast.Name)
    ):
        names.add(node.operand.id)
    elif isinstance(node, ast.BoolOp):
        for value in node.values:
            names.update(_bare_boolean_names(value))
    return names


def _constraint_participants(constraint: Constraint) -> tuple[str, ...]:
    guard, predicate = _split_implication(constraint.expression)
    symbols = _symbols_in_text(predicate)
    if guard is not None:
        symbols.update(_symbols_in_text(guard))
    ordered: list[str] = []
    for name in constraint.parameters:
        if name in symbols:
            ordered.append(name)
            continue
        dotted = sorted(
            symbol
            for symbol in symbols
            if symbol.startswith(f"{name}.")
        )
        if dotted:
            ordered.extend(dotted)
        else:
            ordered.append(name)
    ordered.extend(sorted(symbols - set(ordered)))
    return tuple(dict.fromkeys(ordered))


def _node_kind(
    name: str,
    *,
    boolean_parameters: set[str],
    environment_edge: bool,
) -> DependencyNodeKind:
    if _looks_environment_name(name) or (environment_edge and _looks_environment_name(name)):
        return DependencyNodeKind.ENVIRONMENT
    if _looks_derived_name(name):
        return DependencyNodeKind.DERIVED
    if name in boolean_parameters or _looks_feature_name(name):
        return DependencyNodeKind.FEATURE
    return DependencyNodeKind.PARAMETER


def _looks_environment_name(name: str) -> bool:
    leaf = name.rsplit(".", 1)[-1].lower()
    exact = {
        "available_memory",
        "device_count",
        "local_rank",
        "node_count",
        "num_nodes",
        "rank",
        "world",
        "world_size",
    }
    return leaf in exact or leaf.endswith(("_world_size", "_device_count", "_memory_limit"))


def _looks_derived_name(name: str) -> bool:
    leaf = name.rsplit(".", 1)[-1].lower()
    exact = {
        "data_parallel_degree",
        "effective_context_length",
        "head_dim",
        "layers_per_stage",
        "per_partition_size",
        "tokens_per_global_batch",
    }
    return leaf in exact or leaf.endswith(
        (
            "_per_partition",
            "_per_pipeline_stage",
            "_per_rank",
        )
    )


def _looks_feature_name(name: str) -> bool:
    leaf = name.rsplit(".", 1)[-1].lower()
    return (
        leaf.startswith(("enable_", "disable_", "use_"))
        or leaf.endswith(("_enabled", "_parallel"))
        or leaf in {"sequence_parallel", "overlap_p2p_comm", "tp_comm_overlap"}
    )


def _is_mod_zero(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Compare)
        and len(node.ops) == 1
        and isinstance(node.ops[0], ast.Eq)
        and len(node.comparators) == 1
        and isinstance(node.comparators[0], ast.Constant)
        and node.comparators[0].value == 0
        and isinstance(node.left, ast.BinOp)
        and isinstance(node.left.op, ast.Mod)
    )


def _looks_product_limit(node: ast.Compare) -> bool:
    if len(node.ops) != 1 or len(node.comparators) != 1:
        return False
    if not isinstance(node.ops[0], (ast.Lt, ast.LtE, ast.Gt, ast.GtE)):
        return False
    return _contains_multiplication(node.left) or _contains_multiplication(node.comparators[0])


def _contains_multiplication(node: ast.AST) -> bool:
    return any(isinstance(item, ast.Mult) for item in ast.walk(node))


def _symbols_in_text(expression: str) -> set[str]:
    try:
        return _symbols_in_ast(ast.parse(expression, mode="eval").body)
    except SyntaxError:
        return set()


def _symbols_in_ast(node: ast.AST) -> set[str]:
    collector = _SymbolCollector()
    collector.visit(node)
    return collector.symbols


class _SymbolCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.symbols: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        if node.id not in {
            "False",
            "None",
            "True",
            "abs",
            "bool",
            "float",
            "int",
            "len",
            "max",
            "min",
            "self",
            "str",
        }:
            self.symbols.add(node.id)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        path = _attribute_path(node)
        if path is not None and (
            not path.startswith("self.") or _looks_environment_name(path)
        ):
            self.symbols.add(path)

    def visit_Call(self, node: ast.Call) -> None:
        for argument in node.args:
            self.visit(argument)
        for keyword in node.keywords:
            self.visit(keyword.value)


def _component_from_source(source: str) -> str | None:
    path = Path(source)
    parent = path.parent.as_posix()
    if parent in {"", "."}:
        return path.name or None
    return parent


def _evaluate_text(expression: str, context: Mapping[str, Any]) -> _EvaluationResult:
    try:
        node = ast.parse(expression, mode="eval").body
    except SyntaxError:
        return _EvaluationResult()
    return _evaluate_ast(node, context)


def _evaluate_type_constraint(
    expression: str,
    context: Mapping[str, Any],
) -> _EvaluationResult:
    if ":" not in expression:
        return _evaluate_text(expression, context)
    target, expected = (part.strip() for part in expression.split(":", 1))
    try:
        node = ast.parse(target, mode="eval").body
    except SyntaxError:
        return _EvaluationResult()
    value = _evaluate_ast(node, context)
    if not value.known:
        return value
    checks = {
        "boolean": lambda item: isinstance(item, bool),
        "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
        "number": lambda item: isinstance(item, (int, float)) and not isinstance(item, bool),
        "float": lambda item: isinstance(item, float),
        "string": lambda item: isinstance(item, str),
        "list": lambda item: isinstance(item, list),
    }
    check = checks.get(expected.lower())
    if check is None:
        return _EvaluationResult()
    return _EvaluationResult(value=check(value.value), known=True)


def _evaluate_ast(node: ast.AST, context: Mapping[str, Any]) -> _EvaluationResult:
    if isinstance(node, ast.Constant):
        return _EvaluationResult(value=node.value, known=True)
    if isinstance(node, ast.Name):
        if node.id in context:
            return _EvaluationResult(value=context[node.id], known=True)
        return _EvaluationResult(missing=frozenset({node.id}))
    if isinstance(node, ast.Attribute):
        dotted = _attribute_path(node)
        if dotted is not None and dotted in context:
            return _EvaluationResult(value=context[dotted], known=True)
        base = _evaluate_ast(node.value, context)
        if not base.known:
            leaf = node.attr
            if leaf in context:
                return _EvaluationResult(value=context[leaf], known=True)
            return base
        if isinstance(base.value, Mapping) and node.attr in base.value:
            return _EvaluationResult(value=base.value[node.attr], known=True)
        return _EvaluationResult(missing=frozenset({dotted or node.attr}))
    if isinstance(node, ast.Subscript):
        container = _evaluate_ast(node.value, context)
        key = _evaluate_ast(node.slice, context)
        if not container.known or not key.known:
            return _combine_unknown(container, key)
        try:
            return _EvaluationResult(value=container.value[key.value], known=True)
        except (KeyError, IndexError, TypeError):
            return _EvaluationResult()
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        values = [_evaluate_ast(item, context) for item in node.elts]
        if not all(item.known for item in values):
            return _combine_unknown(*values)
        raw = [item.value for item in values]
        if isinstance(node, ast.Tuple):
            value: Any = tuple(raw)
        elif isinstance(node, ast.Set):
            value = set(raw)
        else:
            value = raw
        return _EvaluationResult(value=value, known=True)
    if isinstance(node, ast.UnaryOp):
        operand = _evaluate_ast(node.operand, context)
        if not operand.known:
            return operand
        operations: dict[type[ast.unaryop], Any] = {
            ast.Not: operator.not_,
            ast.USub: operator.neg,
            ast.UAdd: operator.pos,
        }
        operation = operations.get(type(node.op))
        if operation is None:
            return _EvaluationResult()
        try:
            return _EvaluationResult(value=operation(operand.value), known=True)
        except (TypeError, ValueError):
            return _EvaluationResult()
    if isinstance(node, ast.BoolOp):
        values = [_evaluate_ast(item, context) for item in node.values]
        known_values = [item.value for item in values if item.known]
        missing = frozenset().union(*(item.missing for item in values))
        if isinstance(node.op, ast.And):
            if any(value is False for value in known_values):
                return _EvaluationResult(value=False, known=True, missing=missing)
            if all(item.known for item in values):
                return _EvaluationResult(value=all(bool(item.value) for item in values), known=True)
        elif isinstance(node.op, ast.Or):
            if any(value is True for value in known_values):
                return _EvaluationResult(value=True, known=True, missing=missing)
            if all(item.known for item in values):
                return _EvaluationResult(value=any(bool(item.value) for item in values), known=True)
        return _EvaluationResult(missing=missing)
    if isinstance(node, ast.BinOp):
        left = _evaluate_ast(node.left, context)
        right = _evaluate_ast(node.right, context)
        if not left.known or not right.known:
            return _combine_unknown(left, right)
        operations: dict[type[ast.operator], Any] = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.FloorDiv: operator.floordiv,
            ast.Mod: operator.mod,
            ast.Pow: operator.pow,
        }
        operation = operations.get(type(node.op))
        if operation is None:
            return _EvaluationResult()
        try:
            return _EvaluationResult(value=operation(left.value, right.value), known=True)
        except (ArithmeticError, TypeError, ValueError):
            return _EvaluationResult()
    if isinstance(node, ast.Compare):
        operands = [node.left, *node.comparators]
        values = [_evaluate_ast(item, context) for item in operands]
        if not all(item.known for item in values):
            return _combine_unknown(*values)
        comparisons: dict[type[ast.cmpop], Any] = {
            ast.Eq: operator.eq,
            ast.NotEq: operator.ne,
            ast.Lt: operator.lt,
            ast.LtE: operator.le,
            ast.Gt: operator.gt,
            ast.GtE: operator.ge,
            ast.In: lambda left, right: left in right,
            ast.NotIn: lambda left, right: left not in right,
            ast.Is: operator.is_,
            ast.IsNot: operator.is_not,
        }
        try:
            result = all(
                comparisons[type(op)](values[index].value, values[index + 1].value)
                for index, op in enumerate(node.ops)
            )
        except (KeyError, TypeError, ValueError):
            return _EvaluationResult()
        return _EvaluationResult(value=result, known=True)
    if isinstance(node, ast.IfExp):
        test = _evaluate_ast(node.test, context)
        if not test.known:
            return test
        return _evaluate_ast(node.body if test.value else node.orelse, context)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        functions = {
            "abs": abs,
            "bool": bool,
            "float": float,
            "int": int,
            "len": len,
            "max": max,
            "min": min,
            "str": str,
        }
        function = functions.get(node.func.id)
        if function is None:
            return _EvaluationResult()
        arguments = [_evaluate_ast(item, context) for item in node.args]
        if not all(item.known for item in arguments):
            return _combine_unknown(*arguments)
        try:
            return _EvaluationResult(
                value=function(*(item.value for item in arguments)),
                known=True,
            )
        except (TypeError, ValueError):
            return _EvaluationResult()
    return _EvaluationResult()


def _combine_unknown(*results: _EvaluationResult) -> _EvaluationResult:
    return _EvaluationResult(
        missing=frozenset().union(*(item.missing for item in results))
    )


def _attribute_path(node: ast.Attribute) -> str | None:
    parts: list[str] = [node.attr]
    value: ast.AST = node.value
    while isinstance(value, ast.Attribute):
        parts.append(value.attr)
        value = value.value
    if not isinstance(value, ast.Name):
        return None
    parts.append(value.id)
    return ".".join(reversed(parts))


def _repair_comparison_value(current: Any, boundary: Any, op: ast.cmpop) -> Any:
    if isinstance(op, ast.Eq):
        return boundary
    if not isinstance(boundary, (int, float)):
        return _NO_REPAIR
    if isinstance(op, ast.LtE):
        if current is None or current > boundary:
            return boundary
        return current
    if isinstance(op, ast.Lt):
        candidate = boundary - 1 if isinstance(boundary, int) else math.nextafter(boundary, -math.inf)
        if current is None or current >= boundary:
            return candidate
        return current
    if isinstance(op, ast.GtE):
        if current is None or current < boundary:
            return boundary
        return current
    if isinstance(op, ast.Gt):
        candidate = boundary + 1 if isinstance(boundary, int) else math.nextafter(boundary, math.inf)
        if current is None or current <= boundary:
            return candidate
        return current
    return _NO_REPAIR
