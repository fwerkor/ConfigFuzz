from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROBE_PATH = ROOT / "experiments" / "lmsv_task1_probe.py"
spec = importlib.util.spec_from_file_location("configfuzz_lmsv_task1_probe", PROBE_PATH)
assert spec is not None and spec.loader is not None
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def base_task_config() -> dict[str, object]:
    return {
        "task_type": 6,
        "PTA_NAME": "mindspeed",
        "PTA_PATH": "/opt/pta",
        "MSA_NAME": "msadapter",
        "MSA_PATH": "/opt/msa",
        "tasks": {
            "1": {
                "MODEL_NAME": "qwen3",
                "TOTAL_ITER": 10,
                "MUTNM": 2,
                "SAVE_STEPS": 3,
                "LOAD_STEPS": 4,
                "COMPARE_MODE": "pta_msa",
            }
        },
    }


def test_builds_single_iteration_task1_config(tmp_path: Path) -> None:
    assignments = tmp_path / "case.json"
    assignments.write_text('{"model":{"hidden_size":2112}}\n', encoding="utf-8")

    config = probe.build_task1_config(
        base_task_config(),
        assignments,
        compare_mode="pta_pta",
    )

    task1 = config["tasks"]["1"]
    assert config["task_type"] == 1
    assert task1["TOTAL_ITER"] == 1
    assert task1["MUTNM"] == 0
    assert task1["SAVE_STEPS"] == 3
    assert task1["LOAD_STEPS"] == 4
    assert task1["COMPARE_MODE"] == "pta_pta"
    assert task1["CONFIGFUZZ_ASSIGNMENTS_PATH"] == str(assignments.resolve())


def test_prepare_only_cli_writes_task_config(tmp_path: Path) -> None:
    assignments = tmp_path / "case.json"
    assignments.write_text('{"parallel":{"tensor_model_parallel_size":2}}\n')
    task_config = tmp_path / "base.json"
    task_config.write_text(json.dumps(base_task_config()), encoding="utf-8")
    output = tmp_path / "prepared.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(PROBE_PATH),
            "--config",
            str(assignments),
            "--task-config",
            str(task_config),
            "--prepared-config-output",
            str(output),
            "--prepare-only",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "MILESTONE: ConfigFuzz Task1 configuration prepared" in completed.stdout
    prepared = json.loads(output.read_text())
    assert prepared["task_type"] == 1
    assert prepared["tasks"]["1"]["TOTAL_ITER"] == 1
    assert prepared["tasks"]["1"]["CONFIGFUZZ_ASSIGNMENTS_PATH"] == str(
        assignments.resolve()
    )


def test_parallel_mutate_and_task1_sources_expose_assignment_path() -> None:
    parallel_source = (
        ROOT
        / "lmsv_rec"
        / "utils"
        / "runtime"
        / "mutate_and_forward"
        / "parallel_mutate"
        / "main.py"
    ).read_text(encoding="utf-8")
    task1_source = (
        ROOT / "lmsv_rec" / "utils" / "task" / "task1.py"
    ).read_text(encoding="utf-8")

    assert 'parser.add_argument("--configfuzz-assignments"' in parallel_source
    assert "apply_configfuzz_assignments_file" in parallel_source
    assert "mutator.mutate_parallel_parameters()" in parallel_source
    assert "if configfuzz_assignments:" in parallel_source
    assert "--configfuzz-assignments" in task1_source
    assert "CONFIGFUZZ_ASSIGNMENTS_PATH" in task1_source
    assert "LMSV_MUTATE_SKIP_FORWARD=1" in task1_source

    mutator_source = (
        ROOT
        / "lmsv_rec"
        / "utils"
        / "runtime"
        / "core"
        / "withnum_mutation_system.py"
    ).read_text(encoding="utf-8")
    assert "if mutation_num <= 0:" in mutator_source
    assert "保留模型模板配置" in mutator_source


def test_do_py_accepts_isolated_config_path() -> None:
    source = (ROOT / "lmsv_rec" / "do.py").read_text(encoding="utf-8")
    assert 'os.environ.get("LMSV_CONFIG_PATH"' in source
    assert "指定的 LMSV_CONFIG_PATH 不存在" in source
