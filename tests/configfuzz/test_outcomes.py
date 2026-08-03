from configfuzz.outcomes import (
    ClassificationPolicy,
    OutcomeLabel,
    ProcessObservation,
    classify_observation,
)


def observation(returncode=0, stdout="", stderr="", timed_out=False):
    return ProcessObservation(("target",), returncode, stdout, stderr, 0.1, timed_out)


def test_explicit_invalid_takes_precedence():
    policy = ClassificationPolicy(invalid_patterns=("CONFIG_INVALID",))
    result = classify_observation(observation(2, "CONFIG_INVALID"), policy)
    assert result.label is OutcomeLabel.INVALID


def test_infrastructure_failure_is_unknown():
    policy = ClassificationPolicy(infrastructure_patterns=("connection reset",))
    result = classify_observation(observation(1, stderr="Connection reset"), policy)
    assert result.label is OutcomeLabel.UNKNOWN


def test_success_requires_configured_milestone():
    policy = ClassificationPolicy(milestone_patterns=("READY",))
    result = classify_observation(observation(0, "started"), policy)
    assert result.label is OutcomeLabel.UNKNOWN


def test_failure_after_milestone_is_potential_bug():
    policy = ClassificationPolicy(milestone_patterns=("READY",))
    result = classify_observation(observation(1, "READY\ncrash"), policy)
    assert result.label is OutcomeLabel.POTENTIAL_BUG


def test_timeout_is_unknown():
    result = classify_observation(observation(None, timed_out=True), ClassificationPolicy())
    assert result.label is OutcomeLabel.UNKNOWN
