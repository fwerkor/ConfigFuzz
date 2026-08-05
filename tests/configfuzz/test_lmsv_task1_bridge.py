from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
BRIDGE_PATH = (
    ROOT
    / "lmsv_rec"
    / "utils"
    / "runtime"
    / "configfuzz_bridge.py"
)
spec = importlib.util.spec_from_file_location("configfuzz_lmsv_bridge", BRIDGE_PATH)
assert spec is not None and spec.loader is not None
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)
apply_configfuzz_assignments = bridge.apply_configfuzz_assignments
apply_configfuzz_assignments_file = bridge.apply_configfuzz_assignments_file
ensure_configfuzz_assignments_preserved = bridge.ensure_configfuzz_assignments_preserved
find_configfuzz_assignment_mismatches = bridge.find_configfuzz_assignment_mismatches
load_configfuzz_assignments = bridge.load_configfuzz_assignments


def baseline() -> dict[str, object]:
    return {
        "model": {
            "hidden_size": 2048,
            "num_attention_heads": 16,
        },
        "parallel": {
            "tensor_model_parallel_size": 1,
            "pipeline_model_parallel_size": 1,
        },
        "training": {
            "micro_batch_size": 1,
            "global_batch_size": 32,
        },
        "moe": {},
        "mla": {},
    }


def test_applies_nested_configfuzz_configuration() -> None:
    updated, applied = apply_configfuzz_assignments(
        baseline(),
        {
            "model": {"hidden_size": 2112},
            "parallel": {"tensor_model_parallel_size": 3},
            "training": {"global_batch_size": 24},
        },
    )

    assert updated["model"]["hidden_size"] == 2112
    assert updated["parallel"]["tensor_model_parallel_size"] == 3
    assert updated["training"]["global_batch_size"] == 24
    assert applied == {
        "model.hidden_size": 2112,
        "parallel.tensor_model_parallel_size": 3,
        "training.global_batch_size": 24,
    }


def test_applies_flat_names_and_aliases_to_expected_sections() -> None:
    updated, applied = apply_configfuzz_assignments(
        baseline(),
        {
            "hidden_size": 2304,
            "tensor_parallel_size": 2,
            "num_experts": 8,
        },
    )

    assert updated["model"]["hidden_size"] == 2304
    assert updated["parallel"]["tensor_model_parallel_size"] == 2
    assert updated["moe"]["num_experts"] == 8
    assert applied == {
        "model.hidden_size": 2304,
        "parallel.tensor_model_parallel_size": 2,
        "moe.num_experts": 8,
    }


def test_prefers_semantic_section_for_duplicate_leaf() -> None:
    config = baseline()
    config["model"]["tensor_model_parallel_size"] = 1

    updated, applied = apply_configfuzz_assignments(
        config,
        {"tensor_model_parallel_size": 4},
    )

    assert updated["parallel"]["tensor_model_parallel_size"] == 4
    assert updated["model"]["tensor_model_parallel_size"] == 1
    assert applied == {"parallel.tensor_model_parallel_size": 4}


def test_rejects_ambiguous_unknown_leaf() -> None:
    config = baseline()
    config["model"]["size"] = 1
    config["parallel"]["size"] = 2

    with pytest.raises(ValueError, match="ambiguous"):
        apply_configfuzz_assignments(config, {"size": 3})


def test_detects_validator_repairs_to_configfuzz_values() -> None:
    config = baseline()
    updated, applied = apply_configfuzz_assignments(
        config,
        {"hidden_size": 2112, "tensor_model_parallel_size": 3},
    )
    updated["model"]["hidden_size"] = 2115

    assert find_configfuzz_assignment_mismatches(updated, applied) == {
        "model.hidden_size": {"expected": 2112, "actual": 2115}
    }
    with pytest.raises(ValueError, match="assignments were repaired"):
        ensure_configfuzz_assignments_preserved(updated, applied)


def test_accepts_assignments_preserved_by_validator() -> None:
    updated, applied = apply_configfuzz_assignments(
        baseline(),
        {"hidden_size": 2112, "tensor_model_parallel_size": 3},
    )

    ensure_configfuzz_assignments_preserved(updated, applied)


def test_real_lmsv_validator_repair_is_detected() -> None:
    validator_path = (
        ROOT
        / "lmsv_rec"
        / "utils"
        / "runtime"
        / "mutate_and_forward"
        / "parallel_mutate"
        / "config_validator_moe.py"
    )
    validator_spec = importlib.util.spec_from_file_location(
        "configfuzz_test_validator", validator_path
    )
    assert validator_spec is not None and validator_spec.loader is not None
    validator_module = importlib.util.module_from_spec(validator_spec)
    validator_spec.loader.exec_module(validator_module)
    config = json.loads(
        (ROOT / "experiments" / "lmsv_validator_baseline.json").read_text()
    )
    assigned, applied = apply_configfuzz_assignments(
        config,
        {"hidden_size": 15},
    )

    validated, _, _, _ = validator_module.EnhancedMegatronConfigValidator(
        assigned
    ).validate_and_fix()

    assert validated["model"]["hidden_size"] == 16
    with pytest.raises(ValueError, match="model.hidden_size: 15 -> 16"):
        ensure_configfuzz_assignments_preserved(validated, applied)


def test_loads_wrapped_assignment_file(tmp_path: Path) -> None:
    path = tmp_path / "assignments.json"
    path.write_text(
        json.dumps({"configuration": {"hidden_size": 3072}}),
        encoding="utf-8",
    )

    assert load_configfuzz_assignments(path) == {"hidden_size": 3072}
    updated, applied = apply_configfuzz_assignments_file(baseline(), path)
    assert updated["model"]["hidden_size"] == 3072
    assert applied == {"model.hidden_size": 3072}
