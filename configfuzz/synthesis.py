from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

import z3

from configfuzz.model import Constraint, ConstraintKind, ConstraintSet, Evidence, EvidenceKind
from configfuzz.outcomes import OutcomeLabel
from configfuzz.probing import ProbeSample


@dataclass(frozen=True, slots=True)
class _Candidate:
    expression: str
    kind: ConstraintKind
    predicate: Callable[[Any], bool]
    complexity: int
    rank: int


def synthesize_constraints(
    parameter: str,
    samples: Iterable[ProbeSample],
    *,
    context: dict[str, Any] | None = None,
    max_divisor: int = 64,
) -> ConstraintSet:
    samples = [sample for sample in samples if sample.parameter == parameter]
    positives = [sample for sample in samples if sample.outcome.label is OutcomeLabel.VALID]
    negatives = [sample for sample in samples if sample.outcome.label is OutcomeLabel.INVALID]
    ignored = [
        sample
        for sample in samples
        if sample.outcome.label not in {OutcomeLabel.VALID, OutcomeLabel.INVALID}
    ]

    result = ConstraintSet(
        parameter=parameter,
        metadata={
            "method": "z3_minimum_consistent_conjunction",
            "valid_samples": len(positives),
            "invalid_samples": len(negatives),
            "ignored_samples": len(ignored),
        },
    )
    if not positives or not negatives:
        result.metadata["status"] = "insufficient_labeled_samples"
        return result

    positive_values = [sample.value for sample in positives]
    negative_values = [sample.value for sample in negatives]
    candidates = _build_candidates(
        parameter,
        positive_values,
        negative_values,
        context or {},
        max_divisor,
    )
    candidates = [
        candidate
        for candidate in candidates
        if all(_safe_apply(candidate.predicate, value) for value in positive_values)
        and any(not _safe_apply(candidate.predicate, value) for value in negative_values)
    ]

    selected, uncovered = _select_with_z3(candidates, negative_values)
    support = len(positives) + len(negatives)
    confidence = min(0.99, 0.55 + 0.04 * support)
    for candidate in selected:
        rejected = sum(
            not _safe_apply(candidate.predicate, value) for value in negative_values
        )
        result.add(
            Constraint(
                expression=candidate.expression,
                kind=candidate.kind,
                parameters=(parameter,),
                confidence=confidence,
                evidence=(
                    Evidence(
                        kind=EvidenceKind.DYNAMIC,
                        source="runtime-probe-samples",
                        detail=(
                            f"consistent with {len(positive_values)} valid samples; "
                            f"rejects {rejected}/{len(negative_values)} invalid samples"
                        ),
                    ),
                ),
            )
        )

    result.metadata.update(
        {
            "status": "ok" if selected else "no_supported_constraint",
            "candidate_count": len(candidates),
            "uncovered_invalid_samples": [negative_values[index] for index in uncovered],
        }
    )
    return result


def _build_candidates(
    parameter: str,
    positives: list[Any],
    negatives: list[Any],
    context: dict[str, Any],
    max_divisor: int,
) -> list[_Candidate]:
    values = positives + negatives
    rank = 0
    candidates: list[_Candidate] = []

    def add(expression: str, kind: ConstraintKind, predicate: Callable[[Any], bool], complexity: int) -> None:
        nonlocal rank
        candidates.append(_Candidate(expression, kind, predicate, complexity, rank))
        rank += 1

    if all(_is_int(value) for value in values):
        constants = sorted(set(int(value) for value in values))
        for constant in constants:
            add(
                f"{parameter} >= {constant}",
                ConstraintKind.RANGE,
                lambda value, c=constant: _is_int(value) and int(value) >= c,
                1,
            )
            add(
                f"{parameter} > {constant}",
                ConstraintKind.RANGE,
                lambda value, c=constant: _is_int(value) and int(value) > c,
                1,
            )
            add(
                f"{parameter} <= {constant}",
                ConstraintKind.RANGE,
                lambda value, c=constant: _is_int(value) and int(value) <= c,
                1,
            )
            add(
                f"{parameter} < {constant}",
                ConstraintKind.RANGE,
                lambda value, c=constant: _is_int(value) and int(value) < c,
                1,
            )
        for divisor in range(2, max_divisor + 1):
            add(
                f"{parameter} % {divisor} == 0",
                ConstraintKind.RELATION,
                lambda value, d=divisor: _is_int(value) and int(value) % d == 0,
                2,
            )
        for name, raw_value in sorted(context.items()):
            if not _is_int(raw_value):
                continue
            related = int(raw_value)
            add(
                f"{name} % {parameter} == 0",
                ConstraintKind.RELATION,
                lambda value, c=related: _is_int(value)
                and int(value) != 0
                and c % int(value) == 0,
                1,
            )
            add(
                f"{parameter} % {name} == 0",
                ConstraintKind.RELATION,
                lambda value, c=related: _is_int(value)
                and c != 0
                and int(value) % c == 0,
                1,
            )
            add(
                f"{parameter} <= {name}",
                ConstraintKind.RELATION,
                lambda value, c=related: _is_int(value) and int(value) <= c,
                1,
            )

    elif all(_is_number(value) for value in values):
        constants = sorted(set(float(value) for value in values))
        for constant in constants:
            rendered = _render_number(constant)
            add(
                f"{parameter} >= {rendered}",
                ConstraintKind.RANGE,
                lambda value, c=constant: _is_number(value) and float(value) >= c,
                1,
            )
            add(
                f"{parameter} > {rendered}",
                ConstraintKind.RANGE,
                lambda value, c=constant: _is_number(value) and float(value) > c,
                1,
            )
            add(
                f"{parameter} <= {rendered}",
                ConstraintKind.RANGE,
                lambda value, c=constant: _is_number(value) and float(value) <= c,
                1,
            )
            add(
                f"{parameter} < {rendered}",
                ConstraintKind.RANGE,
                lambda value, c=constant: _is_number(value) and float(value) < c,
                1,
            )

    allowed = tuple(dict.fromkeys(positives))
    if allowed and all(_is_scalar(value) for value in values):
        expression = f"{parameter} in {{{', '.join(_render_literal(value) for value in allowed)}}}"
        add(
            expression,
            ConstraintKind.ENUM,
            lambda value, choices=allowed: value in choices,
            max(3, len(allowed) + 1),
        )

    return candidates


def _select_with_z3(
    candidates: list[_Candidate],
    negatives: list[Any],
) -> tuple[list[_Candidate], list[int]]:
    if not candidates:
        return [], list(range(len(negatives)))
    optimizer = z3.Optimize()
    selected = [z3.Bool(f"select_{index}") for index in range(len(candidates))]
    uncovered = [z3.Bool(f"uncovered_{index}") for index in range(len(negatives))]

    for negative_index, value in enumerate(negatives):
        rejectors = [
            selected[index]
            for index, candidate in enumerate(candidates)
            if not _safe_apply(candidate.predicate, value)
        ]
        optimizer.add(z3.Or(*rejectors, uncovered[negative_index]))

    optimizer.minimize(z3.Sum([z3.If(item, 1, 0) for item in uncovered]))
    optimizer.minimize(
        z3.Sum(
            [
                z3.If(item, candidates[index].complexity, 0)
                for index, item in enumerate(selected)
            ]
        )
    )
    optimizer.minimize(z3.Sum([z3.If(item, 1, 0) for item in selected]))
    optimizer.minimize(
        z3.Sum(
            [
                z3.If(item, candidates[index].rank + 1, 0)
                for index, item in enumerate(selected)
            ]
        )
    )

    if optimizer.check() != z3.sat:
        return [], list(range(len(negatives)))
    model = optimizer.model()
    chosen = [
        candidate
        for item, candidate in zip(selected, candidates, strict=True)
        if z3.is_true(model.eval(item, model_completion=True))
    ]
    missing = [
        index
        for index, item in enumerate(uncovered)
        if z3.is_true(model.eval(item, model_completion=True))
    ]
    return sorted(chosen, key=lambda candidate: candidate.rank), missing


def _safe_apply(predicate: Callable[[Any], bool], value: Any) -> bool:
    try:
        return bool(predicate(value))
    except (TypeError, ValueError, ArithmeticError):
        return False


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _render_number(value: float) -> str:
    return str(int(value)) if value.is_integer() else repr(value)


def _render_literal(value: Any) -> str:
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    return str(value)
