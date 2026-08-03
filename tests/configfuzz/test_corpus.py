from __future__ import annotations

from pathlib import Path

import pytest

from configfuzz.corpus import ConstraintCorpus, ManualConstraintRule, load_corpus


ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "corpus/lmsv/manual_constraints.yaml"


def test_committed_lmsv_corpus_is_valid_and_classified() -> None:
    corpus = load_corpus(CORPUS)

    assert len(corpus.rules) == 93
    assert len({rule.id for rule in corpus.rules}) == len(corpus.rules)
    assert {rule.strength.value for rule in corpus.rules} >= {
        "framework_requirement",
        "lmsv_policy",
        "environment_limit",
        "workaround",
    }
    assert all(rule.status.value == "reviewed" for rule in corpus.rules)


def test_known_rule_preserves_semantics_and_repair_metadata() -> None:
    corpus = load_corpus(CORPUS)
    rule = next(
        item for item in corpus.rules
        if item.id == "lmsv.task1.hidden-size-tp-divisibility"
    )

    assert rule.expression == "model.hidden_size % parallel.tensor_model_parallel_size == 0"
    assert rule.strength.value == "framework_requirement"
    assert rule.enforcement.value == "repair"
    assert rule.repair == {
        "target": "model.hidden_size",
        "strategy": "nearest_divisible",
        "divisor": "parallel.tensor_model_parallel_size",
    }


def test_committed_sources_and_line_ranges_exist() -> None:
    corpus = load_corpus(CORPUS)

    for rule in corpus.rules:
        for source in rule.sources:
            path = ROOT / source.file
            assert path.is_file(), (rule.id, source.file)
            if source.lines is not None:
                line_count = len(path.read_text(encoding="utf-8").splitlines())
                assert source.lines[1] <= line_count, (rule.id, source.lines, line_count)


def test_duplicate_rule_ids_are_rejected() -> None:
    corpus = load_corpus(CORPUS)
    raw = corpus.to_dict()
    raw["rules"].append(raw["rules"][0])

    with pytest.raises(ValueError, match="duplicate rule id"):
        ConstraintCorpus.from_dict(raw)


def test_repair_rule_requires_repair_metadata() -> None:
    corpus = load_corpus(CORPUS)
    raw = corpus.rules[0].to_dict()
    raw.pop("repair")

    with pytest.raises(ValueError, match="requires repair metadata"):
        ManualConstraintRule.from_dict(raw)
