from configfuzz.outcomes import ClassifiedOutcome, OutcomeLabel, ProcessObservation
from configfuzz.probing import ProbeSample
from configfuzz.synthesis import synthesize_constraints


def sample(value, label):
    return ProbeSample(
        parameter="x",
        value=value,
        observation=ProcessObservation(("target",), 0, "", "", 0.1),
        outcome=ClassifiedOutcome(label, "test"),
    )


def expressions(result):
    return {constraint.expression for constraint in result.constraints}


def test_synthesizes_lower_bound():
    result = synthesize_constraints(
        "x",
        [sample(-1, OutcomeLabel.INVALID), sample(0, OutcomeLabel.INVALID),
         sample(1, OutcomeLabel.VALID), sample(2, OutcomeLabel.VALID)],
    )
    assert any(expression in expressions(result) for expression in {"x >= 1", "x > 0"})
    assert result.metadata["uncovered_invalid_samples"] == []


def test_synthesizes_divisibility_without_learning_bug_as_invalid():
    result = synthesize_constraints(
        "x",
        [sample(1, OutcomeLabel.VALID), sample(2, OutcomeLabel.VALID),
         sample(4, OutcomeLabel.VALID), sample(3, OutcomeLabel.INVALID),
         sample(5, OutcomeLabel.INVALID), sample(16, OutcomeLabel.POTENTIAL_BUG)],
        context={"hidden_size": 32},
    )
    assert "hidden_size % x == 0" in expressions(result)
    assert result.metadata["ignored_samples"] == 1
    assert 16 not in result.metadata["uncovered_invalid_samples"]
    assert result.constraints


def test_synthesizes_string_enum():
    result = synthesize_constraints(
        "x",
        [sample("fp16", OutcomeLabel.VALID), sample("bf16", OutcomeLabel.VALID),
         sample("int8", OutcomeLabel.INVALID)],
    )
    assert "x in {'fp16', 'bf16'}" in expressions(result)


def test_prefers_symbolic_context_over_hard_coded_divisor():
    result = synthesize_constraints(
        "x",
        [sample(0, OutcomeLabel.VALID), sample(4, OutcomeLabel.VALID),
         sample(8, OutcomeLabel.VALID), sample(1, OutcomeLabel.INVALID),
         sample(2, OutcomeLabel.INVALID), sample(5, OutcomeLabel.INVALID)],
        context={"tensor_parallel_size": 4},
    )
    assert "x % tensor_parallel_size == 0" in expressions(result)
    assert "x % 4 == 0" not in expressions(result)


def test_requires_both_positive_and_negative_samples():
    result = synthesize_constraints("x", [sample(1, OutcomeLabel.VALID)])
    assert result.constraints == []
    assert result.metadata["status"] == "insufficient_labeled_samples"
