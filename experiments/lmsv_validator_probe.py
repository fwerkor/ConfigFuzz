#!/usr/bin/env python3
"""Expose the lm-sv validator as a lightweight ConfigFuzz runtime oracle."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import random
from pathlib import Path
from typing import Any

from configfuzz.intervention_runner import (
    get_configuration_value,
    resolve_configuration_path,
)


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = (
    ROOT
    / "lmsv_rec"
    / "utils"
    / "runtime"
    / "mutate_and_forward"
    / "parallel_mutate"
    / "config_validator_moe.py"
)


def load_validator_class():
    spec = importlib.util.spec_from_file_location("configfuzz_lmsv_validator", VALIDATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load validator from {VALIDATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.EnhancedMegatronConfigValidator


def set_path(config: dict[str, Any], dotted_path: str, value: Any) -> None:
    parts = dotted_path.split(".")
    current: dict[str, Any] = config
    for part in parts[:-1]:
        child = current.setdefault(part, {})
        if not isinstance(child, dict):
            raise TypeError(f"{part!r} is not a mapping in {dotted_path!r}")
        current = child
    current[parts[-1]] = value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--parameter")
    parser.add_argument("--value", help="JSON scalar")
    parser.add_argument(
        "--tracked-parameters",
        default="[]",
        help="JSON array of fields whose validator repairs imply rejection",
    )
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    tracked_raw = json.loads(args.tracked_parameters)
    if not isinstance(tracked_raw, list):
        raise ValueError("--tracked-parameters must be a JSON array")
    tracked = [resolve_configuration_path(config, str(item)) for item in tracked_raw]
    if args.parameter is not None:
        if args.value is None:
            parser.error("--value is required with --parameter")
        parameter_path = resolve_configuration_path(config, args.parameter)
        set_path(config, parameter_path, json.loads(args.value))
        if parameter_path not in tracked:
            tracked.append(parameter_path)
    elif args.value is not None:
        parser.error("--parameter is required with --value")
    if not tracked:
        parser.error("provide --parameter or --tracked-parameters")
    before = copy.deepcopy(config)

    try:
        random.seed(0)
        validator_class = load_validator_class()
        validated, issues, warnings, fixes = validator_class(config).validate_and_fix()
    except Exception as exc:  # The validator should reject or repair, not crash.
        print(f"BUG_ORACLE: validator raised {type(exc).__name__}: {exc}")
        return 3

    repaired = []
    for path in tracked:
        original_value = get_configuration_value(before, path)
        validated_value = get_configuration_value(validated, path)
        if validated_value != original_value:
            repaired.append((path, original_value, validated_value))
    if repaired:
        for path, original_value, validated_value in repaired:
            print(
                "CONFIG_INVALID: "
                f"{path} was repaired from {original_value!r} to {validated_value!r}"
            )
        for issue in issues:
            if any(path.split(".")[-1] in issue for path, _, _ in repaired):
                print(issue)
        print(f"PROVENANCE: {VALIDATOR_PATH}")
        return 2

    legacy_mode = args.parameter is not None
    milestone = (
        "MILESTONE: lm-sv validator accepted target parameter unchanged"
        if legacy_mode
        else "MILESTONE: lm-sv validator accepted tracked parameters unchanged"
    )
    print(milestone)
    result = {
        "tracked_parameters": tracked,
        "values": {
            path: get_configuration_value(validated, path) for path in tracked
        },
        "fixes_elsewhere": fixes,
        "warnings": warnings,
    }
    if legacy_mode:
        parameter_path = resolve_configuration_path(validated, args.parameter)
        result["parameter"] = args.parameter
        result["value"] = get_configuration_value(validated, parameter_path)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
