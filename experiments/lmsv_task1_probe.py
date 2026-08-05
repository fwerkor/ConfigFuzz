#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
LMSV_ROOT = ROOT / "lmsv_rec"
TASK1_MILESTONE = "MILESTONE: lm-sv Task1 completed with ConfigFuzz assignments"


def build_task1_config(
    base_config: Mapping[str, Any],
    assignments_path: str | Path,
    *,
    compare_mode: str | None = None,
) -> dict[str, Any]:
    config = copy.deepcopy(dict(base_config))
    config["task_type"] = 1
    tasks = config.setdefault("tasks", {})
    if not isinstance(tasks, dict):
        raise ValueError("task configuration 'tasks' must be an object")
    task1 = tasks.setdefault("1", {})
    if not isinstance(task1, dict):
        raise ValueError("task configuration 'tasks.1' must be an object")

    task1["TOTAL_ITER"] = 1
    task1["MUTNM"] = 0
    task1["SAVE_STEPS"] = max(1, int(task1.get("SAVE_STEPS", 1) or 1))
    task1["LOAD_STEPS"] = max(1, int(task1.get("LOAD_STEPS", 1) or 1))
    task1["CONFIGFUZZ_ASSIGNMENTS_PATH"] = str(
        Path(assignments_path).expanduser().resolve()
    )
    if compare_mode is not None:
        task1["COMPARE_MODE"] = compare_mode
    return config


def prepare_task1_config(
    base_config_path: Path,
    assignments_path: Path,
    output_path: Path,
    *,
    compare_mode: str | None = None,
) -> dict[str, Any]:
    base = json.loads(base_config_path.read_text(encoding="utf-8"))
    if not isinstance(base, Mapping):
        raise ValueError("lm-sv task configuration must be a JSON object")
    config = build_task1_config(
        base,
        assignments_path,
        compare_mode=compare_mode,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return config


def run_task1(
    task_config_path: Path,
    *,
    timeout_seconds: float,
) -> int:
    env = os.environ.copy()
    env["LMSV_CONFIG_PATH"] = str(task_config_path.resolve())
    try:
        completed = subprocess.run(
            [sys.executable, "do.py"],
            cwd=LMSV_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = _timeout_text(exc.stdout)
        stderr = _timeout_text(exc.stderr)
        if stdout:
            print(stdout, end="" if stdout.endswith("\n") else "\n")
        if stderr:
            print(stderr, file=sys.stderr, end="" if stderr.endswith("\n") else "\n")
        print("INFRASTRUCTURE_FAILURE: lm-sv Task1 execution timed out")
        return 124

    if completed.stdout:
        print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
    if completed.stderr:
        print(
            completed.stderr,
            file=sys.stderr,
            end="" if completed.stderr.endswith("\n") else "\n",
        )
    if completed.returncode == 0:
        print(TASK1_MILESTONE)
    else:
        print(f"TASK1_FAILURE: lm-sv Task1 exited with code {completed.returncode}")
    return int(completed.returncode)


def _timeout_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one lm-sv Task1 iteration with deterministic ConfigFuzz assignments."
    )
    parser.add_argument("--config", type=Path, required=True, help="case assignment JSON")
    parser.add_argument(
        "--task-config",
        type=Path,
        required=True,
        help="existing lm-sv config.json with real environment paths",
    )
    parser.add_argument(
        "--prepared-config-output",
        type=Path,
        help="retain the generated one-iteration lm-sv configuration",
    )
    parser.add_argument("--compare-mode", help="override Task1 comparison mode")
    parser.add_argument("--timeout-seconds", type=float, default=7200.0)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="only construct the Task1 configuration without executing do.py",
    )
    args = parser.parse_args()

    assignments_path = args.config.expanduser().resolve()
    base_config_path = args.task_config.expanduser().resolve()
    if not assignments_path.is_file():
        raise FileNotFoundError(f"ConfigFuzz assignment file not found: {assignments_path}")
    if not base_config_path.is_file():
        raise FileNotFoundError(f"lm-sv task configuration not found: {base_config_path}")
    if args.timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    if args.prepared_config_output is not None:
        output_path = args.prepared_config_output.expanduser().resolve()
        prepare_task1_config(
            base_config_path,
            assignments_path,
            output_path,
            compare_mode=args.compare_mode,
        )
        print(f"MILESTONE: ConfigFuzz Task1 configuration prepared at {output_path}")
        if args.prepare_only:
            return 0
        return run_task1(output_path, timeout_seconds=args.timeout_seconds)

    with tempfile.TemporaryDirectory(prefix="configfuzz-task1-") as raw_temp:
        output_path = Path(raw_temp) / "config.json"
        prepare_task1_config(
            base_config_path,
            assignments_path,
            output_path,
            compare_mode=args.compare_mode,
        )
        if args.prepare_only:
            print("MILESTONE: ConfigFuzz Task1 configuration prepared")
            print(json.dumps(json.loads(output_path.read_text()), ensure_ascii=False))
            return 0
        return run_task1(output_path, timeout_seconds=args.timeout_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
