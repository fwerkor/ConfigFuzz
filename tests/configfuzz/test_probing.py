import sys
from pathlib import Path

from configfuzz.outcomes import ClassificationPolicy, OutcomeLabel
from configfuzz.probing import ProbeManifest, generate_probe_values, run_manifest


TOY = Path(__file__).parents[2] / "examples" / "toy_target.py"


def manifest(values=()):
    return ProbeManifest(
        parameter="parallel_size",
        parameter_type="integer",
        baseline=1,
        values=tuple(values),
        command=(sys.executable, str(TOY), "--parallel-size", "{value}"),
        timeout_seconds=3,
        classification=ClassificationPolicy(
            invalid_patterns=("CONFIG_INVALID:",),
            bug_patterns=("BUG_ORACLE:",),
            milestone_patterns=("MILESTONE:",),
        ),
    )


def test_integer_boundary_generation_is_bounded_and_deduplicated():
    values = generate_probe_values(manifest())
    assert values[0] == 1
    assert 0 in values
    assert 2 in values
    assert len(values) == len({(type(value), repr(value)) for value in values})
    assert len(values) <= 32


def test_manifest_execution_preserves_bug_trigger():
    samples = run_manifest(manifest((0, 1, 3, 16)))
    labels = {sample.value: sample.outcome.label for sample in samples}
    assert labels[0] is OutcomeLabel.INVALID
    assert labels[1] is OutcomeLabel.VALID
    assert labels[3] is OutcomeLabel.INVALID
    assert labels[16] is OutcomeLabel.POTENTIAL_BUG
