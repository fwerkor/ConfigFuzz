from __future__ import annotations

from configfuzz.model import (
    Constraint,
    ConstraintKind,
    ConstraintSet,
    Evidence,
    EvidenceKind,
)


def test_constraint_set_merges_duplicate_evidence() -> None:
    result = ConstraintSet(parameter="x")
    first = Constraint(
        expression="x > 0",
        kind=ConstraintKind.RANGE,
        parameters=("x",),
        confidence=0.8,
        evidence=(Evidence(EvidenceKind.STATIC, "a.py", 1),),
    )
    second = Constraint(
        expression="x > 0",
        kind=ConstraintKind.RANGE,
        parameters=("x",),
        confidence=0.9,
        evidence=(Evidence(EvidenceKind.STATIC, "b.py", 2),),
    )

    result.add(first)
    result.add(second)

    assert len(result.constraints) == 1
    assert result.constraints[0].confidence == 0.9
    assert len(result.constraints[0].evidence) == 2


def test_constraint_set_round_trip() -> None:
    original = ConstraintSet(
        parameter="hidden_size",
        metadata={"extractor": "test"},
    )
    original.add(
        Constraint(
            expression="hidden_size % tensor_model_parallel_size == 0",
            kind=ConstraintKind.RELATION,
            parameters=("hidden_size", "tensor_model_parallel_size"),
            confidence=0.95,
            evidence=(
                Evidence(
                    EvidenceKind.STATIC,
                    "framework/config.py",
                    12,
                    "assertion",
                ),
            ),
        )
    )

    restored = ConstraintSet.from_dict(original.to_dict())

    assert restored.to_dict() == original.to_dict()
