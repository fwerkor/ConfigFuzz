#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from configfuzz.active_validation import run_active_validation
from configfuzz.dependencies import DependencyGraph
from configfuzz.feedback import apply_probe_feedback
from configfuzz.graph_solver import solve_graph_mutation
from configfuzz.intervention_runner import InterventionExecutionManifest
from configfuzz.probing import load_samples


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATIC_GRAPH = ROOT / "artifacts" / "lmsv_static_inventory.json"
DEFAULT_SAMPLES = ROOT / "artifacts" / "lmsv_hidden_size_samples.json"
DEFAULT_BASELINE = ROOT / "experiments" / "lmsv_validator_baseline.json"
DEFAULT_MANIFEST = (
    ROOT / "experiments" / "manifests" / "lmsv_hidden_size_intervention.json"
)
DEFAULT_FEEDBACK_OUTPUT = ROOT / "artifacts" / "lmsv_static_inventory_feedback.json"
DEFAULT_ACTIVE_OUTPUT = ROOT / "artifacts" / "lmsv_active_validation.json"
DEFAULT_SOLVER_OUTPUT = ROOT / "artifacts" / "lmsv_hidden_size_solver_validation.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Apply existing lm-sv samples, run bounded active validation, and "
            "check solver decisions for representative hidden-size values."
        )
    )
    parser.add_argument("--static-graph", type=Path, default=DEFAULT_STATIC_GRAPH)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--rounds", type=int, default=25)
    parser.add_argument("--solver-timeout-ms", type=int, default=1000)
    parser.add_argument("--feedback-output", type=Path, default=DEFAULT_FEEDBACK_OUTPUT)
    parser.add_argument("--active-output", type=Path, default=DEFAULT_ACTIVE_OUTPUT)
    parser.add_argument("--solver-output", type=Path, default=DEFAULT_SOLVER_OUTPUT)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    graph_payload = _read_object(args.static_graph)
    graph = DependencyGraph.from_dict(graph_payload)
    baseline = _read_object(args.baseline)
    _, samples = load_samples(args.samples)

    feedback = apply_probe_feedback(graph, baseline, samples)
    feedback_payload = {
        "schema_version": 1,
        "experiment": "lmsv_existing_probe_feedback",
        "inputs": {
            "static_graph": _relative(args.static_graph),
            "samples": _relative(args.samples),
            "baseline": _relative(args.baseline),
        },
        "feedback": feedback.to_dict(),
        "dependency_graph": graph.to_dict(),
    }
    _write_json(args.feedback_output, feedback_payload)

    manifest = InterventionExecutionManifest.from_path(args.manifest)
    active = run_active_validation(
        graph,
        manifest,
        max_rounds=args.rounds,
        solver_timeout_ms=args.solver_timeout_ms,
    )
    active_payload = {
        "schema_version": 1,
        "experiment": "lmsv_active_constraint_validation",
        "inputs": {
            "feedback_graph": _relative(args.feedback_output),
            "manifest": _relative(args.manifest),
            "baseline": _relative(args.baseline),
        },
        "active_validation": active.to_dict(),
    }
    _write_json(args.active_output, active_payload)

    observed = {
        sample.value: sample.outcome.label.value
        for sample in samples
        if sample.parameter == "hidden_size"
    }
    validation_cases = []
    for value in (4, 5):
        plan = solve_graph_mutation(active.graph, baseline, "hidden_size", value)
        validation_cases.append(
            {
                "value": value,
                "observed_label": observed.get(value),
                "solver_plan": plan.to_dict(),
            }
        )
    solver_payload = {
        "schema_version": 1,
        "experiment": "lmsv_hidden_size_feedback_active_validation_and_solver",
        "inputs": {
            "active_graph": _relative(args.active_output),
            "samples": _relative(args.samples),
            "baseline": _relative(args.baseline),
        },
        "cases": validation_cases,
    }
    _write_json(args.solver_output, solver_payload)

    return {
        "feedback": feedback.to_dict(),
        "active_validation": active.to_dict()["summary"],
        "solver_cases": validation_cases,
    }


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _relative(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run(args)
    print(f"wrote {_relative(args.feedback_output)}")
    print(f"wrote {_relative(args.active_output)}")
    print(f"wrote {_relative(args.solver_output)}")
    active = summary["active_validation"]
    print(
        f"rounds={active['rounds_executed']} "
        f"stop={active['stop_reason']} "
        f"statuses={active['edge_statuses']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
