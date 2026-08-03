#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from configfuzz.corpus import ConstraintCorpus, dump_corpus, load_corpus

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "corpus/lmsv/manual_constraints.yaml"
TASK1_RULES = ROOT / "corpus/lmsv/sources/task1_validator_rules.yaml"
POOL = ROOT / "lmsv_rec/mutable_params_pool.yaml"


def _render(value: Any) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return "null"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _pool_rules() -> list[dict[str, Any]]:
    data = yaml.safe_load(POOL.read_text(encoding="utf-8"))
    rules: list[dict[str, Any]] = []
    common_scope = {
        "tasks": ["6"],
        "backends": ["PTA", "MSA"],
        "stage": "mutation-generation",
    }
    source = [{
        "file": str(POOL.relative_to(ROOT)),
        "source_type": "parameter_pool",
    }]

    for name, bounds in sorted(data.get("numeric_params", {}).items()):
        expression = (
            f"max({_render(bounds['min_val'])}, baseline({name}) * {_render(bounds['min_factor'])}) "
            f"<= {name} <= "
            f"min({_render(bounds['max_val'])}, baseline({name}) * {_render(bounds['max_factor'])})"
        )
        rules.append({
            "id": f"lmsv.task6.pool.{name}.numeric-window",
            "expression": expression,
            "kind": "range",
            "parameters": [name],
            "enforcement": "sample",
            "strength": "lmsv_policy",
            "status": "reviewed",
            "scope": common_scope,
            "sources": source,
            "rationale": (
                "The Task6 mutator samples this parameter inside the intersection "
                "of an absolute range and a baseline-relative factor window."
            ),
            "metadata": {"pool_entry": bounds},
        })

    for name, values in sorted(data.get("enum_params", {}).items()):
        rendered = ", ".join(_render(value) for value in values)
        rules.append({
            "id": f"lmsv.task6.pool.{name}.enum",
            "expression": f"{name} in {{{rendered}}}",
            "kind": "enum",
            "parameters": [name],
            "enforcement": "sample",
            "strength": "lmsv_policy",
            "status": "reviewed",
            "scope": common_scope,
            "sources": source,
            "rationale": (
                "The Task6 mutator restricts generated values to this explicit candidate set."
            ),
            "metadata": {"pool_entry": values},
        })
    return rules


def build_corpus() -> ConstraintCorpus:
    task1 = load_corpus(TASK1_RULES)
    raw = {
        "schema_version": 1,
        "name": "lmsv-manual-constraints",
        "baseline": {
            "repository": "fwerkor/lm-sv",
            "commit": "e73ba3d35",
            "role": "manual-rule corpus and evaluation reference; not framework source input",
            "coverage": {
                "included": [
                    "Task1 EnhancedMegatronConfigValidator repair rules",
                    "Task6 primary mutable_params_pool.yaml sampling rules",
                ],
                "pending": [
                    "Task1 ParallelParameterMutator sampling and repair rules",
                    "Task2-Task5 task-specific validators and mutators",
                    "Task6 inline fallback parameter pool",
                    "shell-script and framework-template constraints",
                ],
            },
        },
        "rules": sorted(
            [*(rule.to_dict() for rule in task1.rules), *_pool_rules()],
            key=lambda item: item["id"],
        ),
    }
    return ConstraintCorpus.from_dict(raw)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the normalized lm-sv manual constraint corpus."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if the committed corpus is stale",
    )
    args = parser.parse_args()

    corpus = build_corpus()
    if args.check:
        committed = load_corpus(args.output)
        if committed.to_dict() != corpus.to_dict():
            raise SystemExit(f"stale corpus: regenerate {args.output}")
        print(
            f"validated {len(corpus.rules)} normalized rules in "
            f"{args.output.relative_to(ROOT)}"
        )
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    dump_corpus(corpus, args.output)
    print(
        f"wrote {len(corpus.rules)} normalized rules to "
        f"{args.output.relative_to(ROOT)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
