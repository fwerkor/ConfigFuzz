from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from configfuzz.active_validation import run_active_validation
from configfuzz.corpus import load_corpus
from configfuzz.dependencies import DependencyGraph
from configfuzz.extractors import scan_source_paths_multi
from configfuzz.feedback import apply_probe_feedback
from configfuzz.graph_solver import design_edge_intervention, solve_graph_mutation
from configfuzz.intervention_runner import (
    InterventionExecutionManifest,
    intervention_samples_payload,
    run_intervention,
)
from configfuzz.probing import (
    ProbeManifest,
    load_samples,
    run_manifest,
    samples_payload,
)
from configfuzz.selection import select_interventions
from configfuzz.synthesis import synthesize_constraints


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="configfuzz",
        description="Infer candidate configuration constraints from source context.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan = subparsers.add_parser(
        "scan",
        help="scan Python code and configuration declarations for constraints",
    )
    scan.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Python, YAML, or source directories to scan",
    )
    scan.add_argument(
        "--parameter",
        action="append",
        required=True,
        dest="parameters",
        help="parameter name to analyze; repeat for multiple parameters",
    )
    scan.add_argument(
        "--output",
        type=Path,
        help="write JSON to this path instead of stdout",
    )
    scan.add_argument(
        "--broad",
        action="store_true",
        help="retain unsupported expressions for exploratory recall-oriented scans",
    )
    scan.add_argument(
        "--jobs",
        type=int,
        default=0,
        help="parallel source-file workers; 0 selects up to four automatically",
    )
    scan.set_defaults(handler=_run_scan)

    graph = subparsers.add_parser(
        "graph",
        help="build a dependency hypergraph from a saved static scan",
    )
    graph.add_argument("scan", type=Path, help="JSON scan or framework artifact")
    graph.add_argument("--framework", help="override framework scope")
    graph.add_argument("--version", help="override framework version scope")
    graph.add_argument("--output", type=Path, help="write the dependency graph")
    graph.set_defaults(handler=_run_graph)

    plan = subparsers.add_parser(
        "plan-mutation",
        help="plan a constraint-aware joint mutation from a dependency graph",
    )
    plan.add_argument("graph", type=Path, help="dependency graph JSON")
    plan.add_argument("baseline", type=Path, help="baseline configuration JSON")
    plan.add_argument("--parameter", required=True, help="parameter to mutate")
    plan.add_argument(
        "--value",
        required=True,
        type=_parse_json_value,
        help="requested value encoded as JSON, for example 8, true, or \"bf16\"",
    )
    plan.add_argument("--output", type=Path, help="write the mutation plan")
    plan.set_defaults(handler=_run_plan_mutation)

    solve = subparsers.add_parser(
        "solve-mutation",
        help="solve a joint mutation with Z3 over supported dependency edges",
    )
    solve.add_argument("graph", type=Path, help="dependency graph JSON")
    solve.add_argument("baseline", type=Path, help="baseline configuration JSON")
    solve.add_argument("--parameter", required=True, help="parameter to mutate")
    solve.add_argument(
        "--value",
        required=True,
        type=_parse_json_value,
        help="requested value encoded as JSON",
    )
    solve.add_argument(
        "--static-hard",
        action="store_true",
        help="treat unvalidated static candidates as hard instead of weighted soft constraints",
    )
    solve.add_argument("--output", type=Path, help="write the solver plan")
    solve.set_defaults(handler=_run_solve_mutation)

    intervention = subparsers.add_parser(
        "design-intervention",
        help="design satisfying, violating, and repaired configurations for one edge",
    )
    intervention.add_argument("graph", type=Path, help="dependency graph JSON")
    intervention.add_argument("baseline", type=Path, help="baseline configuration JSON")
    intervention.add_argument("--edge", required=True, help="target dependency edge ID")
    intervention.add_argument(
        "--no-repair",
        action="store_true",
        help="omit the repaired counterpart",
    )
    intervention.add_argument("--output", type=Path, help="write the intervention plan")
    intervention.set_defaults(handler=_run_design_intervention)

    selection = subparsers.add_parser(
        "select-interventions",
        help="rank executable candidate edges for active validation",
    )
    selection.add_argument("graph", type=Path, help="dependency graph JSON")
    selection.add_argument("baseline", type=Path, help="baseline configuration JSON")
    selection.add_argument(
        "--limit",
        type=int,
        default=10,
        help="maximum number of ranked intervention plans",
    )
    selection.add_argument("--output", type=Path, help="write the ranked queue")
    selection.set_defaults(handler=_run_select_interventions)

    active = subparsers.add_parser(
        "active-validate",
        help="iteratively select, execute, and learn from candidate edges",
    )
    active.add_argument("graph", type=Path, help="dependency graph JSON")
    active.add_argument(
        "manifest",
        type=Path,
        help="intervention execution manifest JSON",
    )
    active.add_argument(
        "--rounds",
        type=int,
        default=10,
        help="maximum active-validation rounds",
    )
    active.add_argument("--output", type=Path, help="write rounds and updated graph")
    active.set_defaults(handler=_run_active_validation)

    run_intervention_parser = subparsers.add_parser(
        "run-intervention",
        help="execute designed intervention cases and emit feedback-ready samples",
    )
    run_intervention_parser.add_argument("plan", type=Path, help="intervention plan JSON")
    run_intervention_parser.add_argument(
        "manifest",
        type=Path,
        help="intervention execution manifest JSON",
    )
    run_intervention_parser.add_argument(
        "--candidate-index",
        type=int,
        default=0,
        help="zero-based candidate index when the input is a selection queue",
    )
    run_intervention_parser.add_argument(
        "--output",
        type=Path,
        help="write feedback-ready samples",
    )
    run_intervention_parser.set_defaults(handler=_run_intervention)

    feedback = subparsers.add_parser(
        "apply-feedback",
        help="update dependency-edge status and confidence from runtime samples",
    )
    feedback.add_argument("graph", type=Path, help="dependency graph JSON")
    feedback.add_argument("samples", type=Path, help="probe sample JSON")
    feedback.add_argument("baseline", type=Path, help="baseline configuration JSON")
    feedback.add_argument("--output", type=Path, help="write the updated graph")
    feedback.set_defaults(handler=_run_apply_feedback)

    probe = subparsers.add_parser(
        "probe",
        help="execute a runtime probing manifest and classify outcomes",
    )
    probe.add_argument("manifest", type=Path, help="JSON probe manifest")
    probe.add_argument("--output", type=Path, help="write samples to this JSON file")
    probe.set_defaults(handler=_run_probe)

    synthesize = subparsers.add_parser(
        "synthesize",
        help="synthesize a constraint set from labeled probe samples",
    )
    synthesize.add_argument("samples", type=Path, help="JSON file produced by probe")
    synthesize.add_argument(
        "--parameter",
        help="parameter to synthesize; defaults to the manifest parameter",
    )
    synthesize.add_argument("--output", type=Path, help="write the inferred specification")
    synthesize.set_defaults(handler=_run_synthesize)

    infer = subparsers.add_parser(
        "infer",
        help="run probing and synthesize a constraint set in one command",
    )
    infer.add_argument("manifest", type=Path, help="JSON probe manifest")
    infer.add_argument("--samples-output", type=Path, help="retain labeled runtime samples")
    infer.add_argument("--output", type=Path, help="write the inferred specification")
    infer.set_defaults(handler=_run_infer)

    validate_corpus = subparsers.add_parser(
        "validate-corpus",
        help="validate a normalized manual-constraint corpus",
    )
    validate_corpus.add_argument("corpus", type=Path, help="YAML corpus to validate")
    validate_corpus.add_argument("--output", type=Path, help="write a JSON summary")
    validate_corpus.set_defaults(handler=_run_validate_corpus)
    return parser


def _run_scan(args: argparse.Namespace) -> int:
    scanned = scan_source_paths_multi(
        args.paths,
        args.parameters,
        strict=not args.broad,
        jobs=args.jobs,
    )
    results = [scanned[parameter].to_dict() for parameter in args.parameters]
    graph = DependencyGraph.from_constraint_sets(scanned.values())
    payload = {
        "schema_version": 1,
        "results": results,
        "dependency_graph": graph.to_dict(),
    }
    _write_json(payload, args.output)
    return 0


def _run_graph(args: argparse.Namespace) -> int:
    payload = _read_json_object(args.scan)
    scope = {
        key: value
        for key, value in {
            "framework": args.framework,
            "version": args.version,
        }.items()
        if value is not None
    }
    graph = DependencyGraph.from_scan_payload(payload, scope=scope)
    _write_json(
        {"schema_version": 1, "dependency_graph": graph.to_dict()},
        args.output,
    )
    return 0


def _run_plan_mutation(args: argparse.Namespace) -> int:
    graph = DependencyGraph.from_dict(_read_json_object(args.graph))
    baseline_payload = _read_json_object(args.baseline)
    nested = baseline_payload.get("config")
    baseline = nested if isinstance(nested, dict) else baseline_payload
    plan = graph.plan_joint_mutation(args.parameter, args.value, baseline)
    _write_json({"schema_version": 1, "plan": plan.to_dict()}, args.output)
    return 0


def _run_solve_mutation(args: argparse.Namespace) -> int:
    graph = DependencyGraph.from_dict(_read_json_object(args.graph))
    baseline_payload = _read_json_object(args.baseline)
    nested = baseline_payload.get("config")
    baseline = nested if isinstance(nested, dict) else baseline_payload
    plan = solve_graph_mutation(
        graph,
        baseline,
        args.parameter,
        args.value,
        static_as_hard=args.static_hard,
    )
    _write_json({"schema_version": 1, "plan": plan.to_dict()}, args.output)
    return 0


def _run_design_intervention(args: argparse.Namespace) -> int:
    graph = DependencyGraph.from_dict(_read_json_object(args.graph))
    baseline_payload = _read_json_object(args.baseline)
    nested = baseline_payload.get("config")
    baseline = nested if isinstance(nested, dict) else baseline_payload
    plan = design_edge_intervention(
        graph,
        baseline,
        args.edge,
        include_repair=not args.no_repair,
    )
    _write_json({"schema_version": 1, "intervention": plan.to_dict()}, args.output)
    return 0


def _run_select_interventions(args: argparse.Namespace) -> int:
    graph = DependencyGraph.from_dict(_read_json_object(args.graph))
    baseline_payload = _read_json_object(args.baseline)
    nested = baseline_payload.get("config")
    baseline = nested if isinstance(nested, dict) else baseline_payload
    queue = select_interventions(graph, baseline, limit=args.limit)
    _write_json({"schema_version": 1, "selection": queue.to_dict()}, args.output)
    return 0


def _run_active_validation(args: argparse.Namespace) -> int:
    graph = DependencyGraph.from_dict(_read_json_object(args.graph))
    manifest = InterventionExecutionManifest.from_path(args.manifest)
    result = run_active_validation(graph, manifest, max_rounds=args.rounds)
    _write_json(
        {"schema_version": 1, "active_validation": result.to_dict()},
        args.output,
    )
    return 0


def _run_intervention(args: argparse.Namespace) -> int:
    plan_payload = _read_json_object(args.plan)
    manifest = InterventionExecutionManifest.from_path(args.manifest)
    samples = run_intervention(
        plan_payload,
        manifest,
        candidate_index=args.candidate_index,
    )
    _write_json(
        intervention_samples_payload(
            plan_payload,
            manifest,
            samples,
            candidate_index=args.candidate_index,
        ),
        args.output,
    )
    return 0


def _run_apply_feedback(args: argparse.Namespace) -> int:
    graph = DependencyGraph.from_dict(_read_json_object(args.graph))
    _, samples = load_samples(args.samples)
    baseline_payload = _read_json_object(args.baseline)
    nested = baseline_payload.get("config")
    baseline = nested if isinstance(nested, dict) else baseline_payload
    report = apply_probe_feedback(graph, baseline, samples)
    _write_json(
        {
            "schema_version": 1,
            "feedback": report.to_dict(),
            "dependency_graph": graph.to_dict(),
        },
        args.output,
    )
    return 0


def _run_probe(args: argparse.Namespace) -> int:
    manifest = ProbeManifest.from_path(args.manifest)
    samples = run_manifest(manifest)
    _write_json(samples_payload(manifest, samples), args.output)
    return 0


def _run_synthesize(args: argparse.Namespace) -> int:
    payload, samples = load_samples(args.samples)
    manifest = payload.get("manifest", {})
    parameter = args.parameter or manifest.get("parameter")
    if not parameter:
        raise ValueError("parameter is required when the sample file has no manifest parameter")
    context = manifest.get("context", {}) if isinstance(manifest, dict) else {}
    result = synthesize_constraints(str(parameter), samples, context=context)
    output = {
        "schema_version": 1,
        "result": result.to_dict(),
    }
    _write_json(output, args.output)
    return 0


def _run_infer(args: argparse.Namespace) -> int:
    manifest = ProbeManifest.from_path(args.manifest)
    samples = run_manifest(manifest)
    sample_payload = samples_payload(manifest, samples)
    if args.samples_output is not None:
        _write_json(sample_payload, args.samples_output)
    result = synthesize_constraints(
        manifest.parameter,
        samples,
        context=dict(manifest.context),
    )
    output = {
        "schema_version": 1,
        "result": result.to_dict(),
    }
    _write_json(output, args.output)
    return 0


def _run_validate_corpus(args: argparse.Namespace) -> int:
    corpus = load_corpus(args.corpus)
    strength_counts: dict[str, int] = {}
    enforcement_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    for rule in corpus.rules:
        strength_counts[rule.strength.value] = strength_counts.get(rule.strength.value, 0) + 1
        enforcement_counts[rule.enforcement.value] = enforcement_counts.get(rule.enforcement.value, 0) + 1
        status_counts[rule.status.value] = status_counts.get(rule.status.value, 0) + 1
    payload = {
        "schema_version": corpus.schema_version,
        "name": corpus.name,
        "rules": len(corpus.rules),
        "strengths": dict(sorted(strength_counts.items())),
        "enforcements": dict(sorted(enforcement_counts.items())),
        "statuses": dict(sorted(status_counts.items())),
    }
    _write_json(payload, args.output)
    return 0


def _write_json(payload: object, path: Path | None) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if path is None:
        print(text, end="")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _read_json_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def _parse_json_value(text: str) -> object:
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(
            "mutation value must be valid JSON; quote string values"
        ) from exc


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
