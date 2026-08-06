from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from configfuzz.corpus import ConstraintCorpus, ManualConstraintRule, load_corpus
from configfuzz.experiment import MutationIntent


_TOPOLOGY_LEAVES = (
    "tensor_model_parallel_size",
    "pipeline_model_parallel_size",
    "expert_model_parallel_size",
    "context_parallel_size",
)


@dataclass(frozen=True, slots=True)
class WorkloadSpec:
    workload_id: str
    baseline_id: str
    baseline_config: Path
    family: str
    constraint_ids: tuple[str, ...] = ()
    topology_parameters: tuple[str, ...] = _TOPOLOGY_LEAVES
    metadata: Mapping[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, base_dir: Path) -> "WorkloadSpec":
        baseline = data.get("baseline_config")
        if baseline is None:
            raise ValueError(
                f"workload {data.get('workload_id', '<unknown>')}: baseline_config is not bound"
            )
        baseline_path = Path(str(baseline))
        if not baseline_path.is_absolute():
            baseline_path = (base_dir / baseline_path).resolve()
        if not baseline_path.is_file():
            raise FileNotFoundError(
                f"baseline configuration not found: {baseline_path}"
            )
        return cls(
            workload_id=_required_string(data, "workload_id"),
            baseline_id=_required_string(data, "baseline_id"),
            baseline_config=baseline_path,
            family=_required_string(data, "family"),
            constraint_ids=tuple(str(item) for item in data.get("constraint_ids", ())),
            topology_parameters=tuple(
                str(item) for item in data.get("topology_parameters", _TOPOLOGY_LEAVES)
            ),
            metadata=dict(data.get("metadata", {})),
        )


def load_workloads(
    path: str | Path, *, skip_unbound: bool = False
) -> list[WorkloadSpec]:
    source = Path(path).expanduser().resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("workload registry root must be an object")
    workloads: list[WorkloadSpec] = []
    for item in raw.get("workloads", ()):
        if not isinstance(item, Mapping):
            raise ValueError("workload entry must be an object")
        if item.get("baseline_config") is None and skip_unbound:
            continue
        workloads.append(WorkloadSpec.from_dict(item, base_dir=source.parent))
    seen: set[str] = set()
    for workload in workloads:
        if workload.workload_id in seen:
            raise ValueError(f"duplicate workload id: {workload.workload_id}")
        seen.add(workload.workload_id)
    return workloads


def generate_intents(
    corpus: ConstraintCorpus,
    workloads: Sequence[WorkloadSpec],
    *,
    topology_values: Sequence[int] = (1, 2, 4, 8),
) -> list[MutationIntent]:
    intents: list[MutationIntent] = []
    for workload in workloads:
        baseline = json.loads(workload.baseline_config.read_text(encoding="utf-8"))
        if not isinstance(baseline, Mapping):
            raise ValueError(
                f"workload {workload.workload_id}: baseline configuration must be an object"
            )
        allowed_ids = set(workload.constraint_ids)
        rules = [
            rule for rule in corpus.rules if not allowed_ids or rule.id in allowed_ids
        ]
        workload_intents: list[MutationIntent] = []
        for rule in rules:
            workload_intents.extend(_rule_intents(rule, workload, baseline))
        workload_intents.extend(
            _topology_intents(workload, baseline, topology_values=topology_values)
        )
        intents.extend(_deduplicate(workload_intents))
    return sorted(intents, key=lambda item: item.intent_id)


def generate_intent_payload(
    corpus_path: str | Path,
    workloads_path: str | Path,
    *,
    skip_unbound: bool = False,
    topology_values: Sequence[int] = (1, 2, 4, 8),
) -> dict[str, Any]:
    corpus = load_corpus(corpus_path)
    workloads = load_workloads(workloads_path, skip_unbound=skip_unbound)
    intents = generate_intents(corpus, workloads, topology_values=topology_values)
    counts: dict[str, int] = {}
    for intent in intents:
        counts[intent.workload_id] = counts.get(intent.workload_id, 0) + 1
    return {
        "schema_version": 1,
        "name": "rq2-generated-mutation-intents",
        "metadata": {
            "source_corpus": corpus.name,
            "source_corpus_baseline": corpus.baseline,
            "workload_count": len(workloads),
            "intent_count": len(intents),
            "intents_per_workload": dict(sorted(counts.items())),
            "generation_policy": [
                "enumeration alternatives",
                "numeric window boundaries",
                "repair boundaries and adjacent values",
                "simple guard-enabling transitions",
                "parallel topology values",
            ],
            "warning": (
                "Generated intentions are candidates. Remove examples, verify workload "
                "scope, and freeze the final list before running any method."
            ),
        },
        "intents": [intent.to_dict() for intent in intents],
    }


def _rule_intents(
    rule: ManualConstraintRule,
    workload: WorkloadSpec,
    baseline: Mapping[str, Any],
) -> list[MutationIntent]:
    candidates: list[tuple[str, Any, str, Mapping[str, Any]]] = []
    pool_entry = rule.metadata.get("pool_entry")
    if isinstance(pool_entry, list) and len(rule.parameters) == 1:
        parameter = rule.parameters[0]
        current = _try_get(baseline, parameter)
        if current.found:
            for value in pool_entry:
                if value != current.value:
                    candidates.append(
                        (
                            parameter,
                            value,
                            "enumeration_alternative",
                            {"baseline_value": current.value},
                        )
                    )
    elif isinstance(pool_entry, Mapping) and len(rule.parameters) == 1:
        parameter = rule.parameters[0]
        current = _try_get(baseline, parameter)
        if current.found and _is_number(current.value):
            candidates.extend(
                (
                    parameter,
                    value,
                    intent_class,
                    {"baseline_value": current.value},
                )
                for value, intent_class in _numeric_window_values(
                    current.value, pool_entry
                )
                if value != current.value
            )

    if rule.repair:
        target = str(rule.repair.get("target", ""))
        current = _try_get(baseline, target)
        if target and current.found:
            for value, intent_class, metadata in _repair_values(
                rule, baseline, current.value
            ):
                if value != current.value:
                    candidates.append((target, value, intent_class, metadata))

    if rule.scope.condition:
        for parameter, value in _guard_enabling_values(rule.scope.condition, baseline):
            current = _try_get(baseline, parameter)
            if current.found and value != current.value:
                candidates.append(
                    (
                        parameter,
                        value,
                        "guard_enable_transition",
                        {
                            "guard": rule.scope.condition,
                            "baseline_value": current.value,
                        },
                    )
                )

    intents: list[MutationIntent] = []
    for parameter, value, intent_class, metadata in candidates:
        intents.append(
            _make_intent(
                workload,
                parameter,
                value,
                intent_class,
                source_constraint_ids=(rule.id,),
                metadata={"rule_expression": rule.expression, **dict(metadata)},
            )
        )
    return intents


def _repair_values(
    rule: ManualConstraintRule,
    baseline: Mapping[str, Any],
    current: Any,
) -> list[tuple[Any, str, Mapping[str, Any]]]:
    repair = rule.repair or {}
    strategy = str(repair.get("strategy", ""))
    values: list[tuple[Any, str, Mapping[str, Any]]] = []
    if strategy in {"nearest_divisible", "nearest_lower_divisible"}:
        divisor_expression = str(repair.get("divisor", ""))
        divisor = _evaluate_expression(divisor_expression, baseline)
        if _is_number(current) and _is_number(divisor) and int(divisor) > 0:
            divisor_int = int(divisor)
            current_int = int(current)
            lower = (current_int // divisor_int) * divisor_int
            upper = lower + divisor_int
            for boundary in {max(divisor_int, lower), max(divisor_int, upper)}:
                values.extend(
                    [
                        (
                            boundary,
                            "divisibility_legal_boundary",
                            {"divisor": divisor_int},
                        ),
                        (
                            boundary - 1,
                            "divisibility_adjacent_value",
                            {"divisor": divisor_int},
                        ),
                        (
                            boundary + 1,
                            "divisibility_adjacent_value",
                            {"divisor": divisor_int},
                        ),
                    ]
                )
    elif strategy == "set":
        value = repair.get("value")
        values.append((value, "repair_boundary", {"repair_strategy": strategy}))
        alternative = _adjacent_or_opposite(value)
        if alternative is not None:
            values.append(
                (alternative, "repair_adjacent_value", {"repair_strategy": strategy})
            )
    elif strategy in {"cap", "fraction"}:
        expression = str(repair.get("expression", ""))
        boundary = _evaluate_expression(expression, baseline)
        if _is_number(boundary):
            values.extend(_boundary_triplet(boundary, strategy))
    elif strategy in {"copy", "raise_to", "align_or_fallback"}:
        source = str(repair.get("source", ""))
        source_value = _try_get(baseline, source)
        if source_value.found:
            values.append(
                (
                    source_value.value,
                    "relational_boundary",
                    {"source_parameter": source, "repair_strategy": strategy},
                )
            )
            adjacent = _adjacent_or_opposite(source_value.value)
            if adjacent is not None:
                values.append(
                    (
                        adjacent,
                        "relational_adjacent_value",
                        {"source_parameter": source, "repair_strategy": strategy},
                    )
                )
    elif strategy in {"disable", "set_or_disable"}:
        preferred = repair.get("preferred", repair.get("value", False))
        values.append(
            (preferred, "feature_enable_transition", {"repair_strategy": strategy})
        )
        values.append(
            (False, "feature_disable_transition", {"repair_strategy": strategy})
        )
    return _unique_value_records(values)


def _numeric_window_values(
    current: int | float,
    pool_entry: Mapping[str, Any],
) -> list[tuple[int | float, str]]:
    minimum = pool_entry.get("min_val")
    maximum = pool_entry.get("max_val")
    min_factor = pool_entry.get("min_factor", 1.0)
    max_factor = pool_entry.get("max_factor", 1.0)
    if not all(_is_number(item) for item in (minimum, maximum, min_factor, max_factor)):
        return []
    low = max(float(minimum), float(current) * float(min_factor))
    high = min(float(maximum), float(current) * float(max_factor))
    integer = isinstance(current, int) and not isinstance(current, bool)
    raw = [
        (minimum, "numeric_minimum"),
        (_coerce_number(low, integer), "numeric_local_lower_boundary"),
        (_coerce_number(high, integer), "numeric_local_upper_boundary"),
        (maximum, "numeric_maximum"),
    ]
    return _unique_pairs(raw)


def _topology_intents(
    workload: WorkloadSpec,
    baseline: Mapping[str, Any],
    *,
    topology_values: Sequence[int],
) -> list[MutationIntent]:
    world_size = _try_get(baseline, "world_size")
    maximum = (
        int(world_size.value)
        if world_size.found and _is_number(world_size.value)
        else None
    )
    intents: list[MutationIntent] = []
    for parameter in workload.topology_parameters:
        current = _try_get(baseline, parameter)
        if not current.found:
            continue
        values = {
            int(value)
            for value in topology_values
            if int(value) >= 1 and (maximum is None or int(value) <= maximum)
        }
        if maximum is not None:
            values.update(
                divisor for divisor in range(1, maximum + 1) if maximum % divisor == 0
            )
        for value in sorted(values):
            if value == current.value:
                continue
            intents.append(
                _make_intent(
                    workload,
                    parameter,
                    value,
                    "parallel_topology",
                    metadata={"baseline_value": current.value, "world_size": maximum},
                )
            )
    return intents


def _guard_enabling_values(
    guard: str,
    baseline: Mapping[str, Any],
) -> list[tuple[str, Any]]:
    values: list[tuple[str, Any]] = []
    patterns = (
        (r"([A-Za-z_][A-Za-z0-9_.]*)\s*==\s*true", True),
        (r"([A-Za-z_][A-Za-z0-9_.]*)\s*==\s*false", False),
        (r"([A-Za-z_][A-Za-z0-9_.]*)\s*>\s*1", 2),
        (r"([A-Za-z_][A-Za-z0-9_.]*)\s*>=\s*1", 1),
        (r"([A-Za-z_][A-Za-z0-9_.]*)\s*!=\s*0", 1),
    )
    for pattern, value in patterns:
        for match in re.finditer(pattern, guard, re.I):
            parameter = match.group(1)
            if _try_get(baseline, parameter).found:
                values.append((parameter, value))
    for match in re.finditer(r"([A-Za-z_][A-Za-z0-9_.]*)\s*==\s*(['\"])(.*?)\2", guard):
        parameter, value = match.group(1), match.group(3)
        if _try_get(baseline, parameter).found:
            values.append((parameter, value))
    return list(dict.fromkeys(values))


def _evaluate_expression(expression: str, baseline: Mapping[str, Any]) -> Any | None:
    if not expression:
        return None
    symbols = sorted(
        set(re.findall(r"[A-Za-z_][A-Za-z0-9_.]*", expression)),
        key=len,
        reverse=True,
    )
    rewritten = expression
    values: dict[str, Any] = {}
    for index, symbol in enumerate(symbols):
        if symbol in {"min", "max", "true", "false", "None"}:
            continue
        resolved = _try_get(baseline, symbol)
        if not resolved.found:
            return None
        placeholder = f"v{index}"
        rewritten = re.sub(
            rf"(?<![A-Za-z0-9_.]){re.escape(symbol)}(?![A-Za-z0-9_.])",
            placeholder,
            rewritten,
        )
        values[placeholder] = resolved.value
    try:
        node = ast.parse(rewritten, mode="eval")
        return _eval_ast(node.body, values)
    except (SyntaxError, TypeError, ValueError, ZeroDivisionError):
        return None


def _eval_ast(node: ast.AST, values: Mapping[str, Any]) -> Any:
    if isinstance(node, ast.Constant) and _is_number(node.value):
        return node.value
    if isinstance(node, ast.Name) and node.id in values:
        return values[node.id]
    if isinstance(node, ast.BinOp):
        left = _eval_ast(node.left, values)
        right = _eval_ast(node.right, values)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.FloorDiv):
            return left // right
        if isinstance(node.op, ast.Mod):
            return left % right
    if isinstance(node, ast.UnaryOp):
        value = _eval_ast(node.operand, values)
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return value
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        args = [_eval_ast(item, values) for item in node.args]
        if node.func.id == "min":
            return min(args)
        if node.func.id == "max":
            return max(args)
    raise ValueError("unsupported expression")


@dataclass(frozen=True, slots=True)
class _ResolvedValue:
    found: bool
    value: Any = None


def _try_get(configuration: Mapping[str, Any], parameter: str) -> _ResolvedValue:
    if not parameter:
        return _ResolvedValue(False)
    parts = parameter.split(".")
    current: Any = configuration
    exact = True
    for part in parts:
        if not isinstance(current, Mapping) or part not in current:
            exact = False
            break
        current = current[part]
    if exact:
        return _ResolvedValue(True, current)
    leaf = parts[-1]
    matches = list(_find_leaf(configuration, leaf))
    if len(matches) == 1:
        return _ResolvedValue(True, matches[0])
    return _ResolvedValue(False)


def _find_leaf(value: Any, leaf: str) -> Iterable[Any]:
    if not isinstance(value, Mapping):
        return
    for key, item in value.items():
        if str(key) == leaf:
            yield item
        if isinstance(item, Mapping):
            yield from _find_leaf(item, leaf)


def _make_intent(
    workload: WorkloadSpec,
    parameter: str,
    value: Any,
    intent_class: str,
    *,
    source_constraint_ids: tuple[str, ...] = (),
    metadata: Mapping[str, Any] | None = None,
) -> MutationIntent:
    identity = json.dumps(
        [workload.workload_id, parameter, value, intent_class, source_constraint_ids],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    leaf = parameter.rsplit(".", 1)[-1].replace("_", "-")
    return MutationIntent(
        intent_id=f"{workload.workload_id}-{leaf}-{intent_class}-{digest}",
        workload_id=workload.workload_id,
        baseline_id=workload.baseline_id,
        target_parameter=parameter,
        target_value=value,
        intent_class=intent_class,
        source_constraint_ids=source_constraint_ids,
        metadata={"workload_family": workload.family, **dict(metadata or {})},
    )


def _deduplicate(intents: Sequence[MutationIntent]) -> list[MutationIntent]:
    selected: dict[tuple[str, str, str], MutationIntent] = {}
    for intent in intents:
        value_key = json.dumps(intent.target_value, ensure_ascii=False, sort_keys=True)
        key = (intent.target_parameter, value_key, intent.intent_class)
        existing = selected.get(key)
        if existing is None:
            selected[key] = intent
            continue
        constraint_ids = tuple(
            sorted(
                set(existing.source_constraint_ids) | set(intent.source_constraint_ids)
            )
        )
        metadata = {**dict(existing.metadata), **dict(intent.metadata)}
        selected[key] = MutationIntent(
            intent_id=existing.intent_id,
            workload_id=existing.workload_id,
            baseline_id=existing.baseline_id,
            target_parameter=existing.target_parameter,
            target_value=existing.target_value,
            intent_class=existing.intent_class,
            source_constraint_ids=constraint_ids,
            metadata=metadata,
        )
    return list(selected.values())


def _boundary_triplet(
    boundary: int | float, strategy: str
) -> list[tuple[Any, str, Mapping[str, Any]]]:
    step = (
        1
        if isinstance(boundary, int) and not isinstance(boundary, bool)
        else max(abs(float(boundary)) * 0.01, 1e-6)
    )
    return [
        (boundary - step, "numeric_boundary_below", {"repair_strategy": strategy}),
        (boundary, "numeric_boundary", {"repair_strategy": strategy}),
        (boundary + step, "numeric_boundary_above", {"repair_strategy": strategy}),
    ]


def _adjacent_or_opposite(value: Any) -> Any | None:
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value - 1 if value > 0 else value + 1
    if isinstance(value, float):
        return value + max(abs(value) * 0.01, 1e-6)
    return None


def _coerce_number(value: float, integer: bool) -> int | float:
    return int(round(value)) if integer else value


def _unique_pairs(values: Iterable[tuple[Any, str]]) -> list[tuple[Any, str]]:
    result: list[tuple[Any, str]] = []
    seen: set[tuple[str, str]] = set()
    for value, label in values:
        key = (json.dumps(value, ensure_ascii=False, sort_keys=True), label)
        if key not in seen:
            seen.add(key)
            result.append((value, label))
    return result


def _unique_value_records(
    values: Iterable[tuple[Any, str, Mapping[str, Any]]],
) -> list[tuple[Any, str, Mapping[str, Any]]]:
    result: list[tuple[Any, str, Mapping[str, Any]]] = []
    seen: set[tuple[str, str]] = set()
    for value, label, metadata in values:
        if _is_number(value) and not math.isfinite(float(value)):
            continue
        key = (json.dumps(value, ensure_ascii=False, sort_keys=True), label)
        if key not in seen:
            seen.add(key)
            result.append((value, label, metadata))
    return result


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _required_string(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    if value is None or not str(value).strip():
        raise ValueError(f"{key} must be a non-empty string")
    return str(value).strip()
