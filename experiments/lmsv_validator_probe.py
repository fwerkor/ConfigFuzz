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


def get_path(config: dict[str, Any], dotted_path: str) -> Any:
    current: Any = config
    for part in dotted_path.split("."):
        current = current[part]
    return current


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
    parser.add_argument("--parameter", required=True)
    parser.add_argument("--value", required=True, help="JSON scalar")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    value = json.loads(args.value)
    set_path(config, args.parameter, value)
    before = copy.deepcopy(config)

    try:
        random.seed(0)
        validator_class = load_validator_class()
        validated, issues, warnings, fixes = validator_class(config).validate_and_fix()
    except Exception as exc:  # The validator should reject or repair, not crash.
        print(f"BUG_ORACLE: validator raised {type(exc).__name__}: {exc}")
        return 3

    original_value = get_path(before, args.parameter)
    validated_value = get_path(validated, args.parameter)
    if validated_value != original_value:
        print(
            "CONFIG_INVALID: "
            f"{args.parameter} was repaired from {original_value!r} to {validated_value!r}"
        )
        for issue in issues:
            if args.parameter.split(".")[-1] in issue:
                print(issue)
        return 2

    print("MILESTONE: lm-sv validator accepted target parameter unchanged")
    print(
        json.dumps(
            {
                "parameter": args.parameter,
                "value": validated_value,
                "fixes_elsewhere": fixes,
                "warnings": warnings,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
