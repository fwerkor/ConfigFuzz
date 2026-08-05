from __future__ import annotations

import sys
from pathlib import Path

from configfuzz.active_validation import run_active_validation
from configfuzz.dependencies import DependencyGraph, DependencyStatus
from configfuzz.intervention_runner import InterventionExecutionManifest
from configfuzz.model import Constraint, ConstraintKind, ConstraintSet, Evidence, EvidenceKind
from configfuzz.outcomes import ClassificationPolicy


def make_graph(*constraints: Constraint) -> DependencyGraph:
    sets: dict[str, ConstraintSet] = {}
    for constraint in constraints:
        for parameter in constraint.parameters:
            sets.setdefault(parameter, ConstraintSet(parameter=parameter)).add(constraint)
    return DependencyGraph.from_constraint_sets(sets.values())


def parity_constraint(name: str) -> Constraint:
    return Constraint(
        expression=f"{name} % 2 == 0",
        kind=ConstraintKind.RELATION,
        parameters=(name,),
        confidence=0.7,
        evidence=(
            Evidence(
                kind=EvidenceKind.STATIC,
                source="framework/config.py",
                line=7,
            ),
        ),
    )


def manifest_for(tmp_path: Path) -> InterventionExecutionManifest:
    baseline = tmp_path / "baseline.json"
    baseline.write_text('{"x": 2, "y": 2}\n', encoding="utf-8")
    oracle = (
        "import json,sys; "
        "c=json.load(open(sys.argv[1])); "
        "ok=c['x']%2==0 and c['y']%2==0; "
        "print('MILESTONE: accepted' if ok else "
        "'CONFIG_INVALID: odd value\\nPROVENANCE: framework/config.py'); "
        "raise SystemExit(0 if ok else 2)"
    )
    return InterventionExecutionManifest(
        baseline_config=baseline,
        command=(sys.executable, "-c", oracle, "{config}"),
        cwd=tmp_path,
        classification=ClassificationPolicy(
            invalid_patterns=("CONFIG_INVALID:",),
            milestone_patterns=("MILESTONE: accepted",),
        ),
        provenance_patterns=(r"PROVENANCE: framework/config\.py",),
    )


def test_active_validation_confirms_multiple_edges(tmp_path: Path) -> None:
    graph = make_graph(parity_constraint("x"), parity_constraint("y"))

    result = run_active_validation(graph, manifest_for(tmp_path), max_rounds=2)

    assert result.stop_reason == "budget_exhausted"
    assert len(result.rounds) == 2
    assert len(set(result.attempted_edges)) == 2
    assert all(round_.feedback.paired_interventions == 1 for round_ in result.rounds)
    assert all(
        edge.status is DependencyStatus.CONFIRMED for edge in graph.edges.values()
    )
    assert result.to_dict()["summary"]["edge_statuses"] == {"confirmed": 2}


def test_active_validation_stops_when_no_executable_edge(tmp_path: Path) -> None:
    graph = make_graph(
        Constraint(
            expression="x: integer",
            kind=ConstraintKind.TYPE,
            parameters=("x",),
        )
    )

    result = run_active_validation(graph, manifest_for(tmp_path), max_rounds=3)

    assert result.stop_reason == "no_executable_candidates"
    assert not result.rounds
    assert not result.attempted_edges


def test_attempted_unconfirmed_edge_is_not_retried(tmp_path: Path) -> None:
    graph = make_graph(parity_constraint("x"))
    manifest = manifest_for(tmp_path)
    manifest = InterventionExecutionManifest(
        baseline_config=manifest.baseline_config,
        command=manifest.command,
        cwd=manifest.cwd,
        classification=manifest.classification,
        provenance_patterns=(r"does-not-match",),
    )

    result = run_active_validation(graph, manifest, max_rounds=3)

    assert len(result.rounds) == 1
    assert result.stop_reason == "no_executable_candidates"
    assert next(iter(graph.edges.values())).status is DependencyStatus.DYNAMICALLY_SUPPORTED
