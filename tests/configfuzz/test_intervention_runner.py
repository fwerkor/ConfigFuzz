from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from configfuzz.dependencies import DependencyGraph, DependencyStatus
from configfuzz.feedback import apply_probe_feedback
from configfuzz.graph_solver import design_edge_intervention
from configfuzz.intervention_runner import (
    InterventionExecutionManifest,
    apply_configuration_updates,
    run_intervention,
)
from configfuzz.model import Constraint, ConstraintKind, ConstraintSet, Evidence, EvidenceKind
from configfuzz.outcomes import ClassificationPolicy, OutcomeLabel


def graph_for(expression: str, parameters: tuple[str, ...]) -> DependencyGraph:
    constraint = Constraint(
        expression=expression,
        kind=ConstraintKind.RELATION,
        parameters=parameters,
        confidence=0.8,
        evidence=(
            Evidence(
                kind=EvidenceKind.STATIC,
                source="framework/config.py",
                line=7,
            ),
        ),
    )
    sets = []
    for parameter in parameters:
        item = ConstraintSet(parameter=parameter)
        item.add(constraint)
        sets.append(item)
    return DependencyGraph.from_constraint_sets(sets)


def test_runs_pair_and_matches_rejection_provenance(tmp_path: Path) -> None:
    graph = graph_for("x % 2 == 0", ("x",))
    edge = next(iter(graph.edges.values()))
    plan = {"intervention": design_edge_intervention(graph, {"x": 2}, edge.id).to_dict()}
    baseline = tmp_path / "baseline.json"
    baseline.write_text('{"x": 2}\n', encoding="utf-8")
    oracle = (
        "import json,sys; "
        "x=json.load(open(sys.argv[1]))['x']; "
        "ok=x%2==0; "
        "print('MILESTONE: accepted' if ok else "
        "'CONFIG_INVALID: odd value\\nPROVENANCE: framework/config.py'); "
        "raise SystemExit(0 if ok else 2)"
    )
    manifest = InterventionExecutionManifest(
        baseline_config=baseline,
        command=(sys.executable, "-c", oracle, "{config}"),
        cwd=tmp_path,
        classification=ClassificationPolicy(
            invalid_patterns=("CONFIG_INVALID:",),
            milestone_patterns=("MILESTONE: accepted",),
        ),
        provenance_patterns=(r"PROVENANCE: framework/config\.py",),
    )

    samples = run_intervention(plan, manifest)

    by_role = {sample.intervention_role: sample for sample in samples}
    assert by_role["satisfying"].outcome.label is OutcomeLabel.VALID
    assert by_role["violating"].outcome.label is OutcomeLabel.INVALID
    assert by_role["violating"].provenance_matched
    assert by_role["repaired"].outcome.label is OutcomeLabel.VALID

    report = apply_probe_feedback(graph, {"x": 2}, samples)
    assert report.paired_interventions == 1
    assert graph.edges[edge.id].status is DependencyStatus.CONFIRMED


def test_applies_flat_updates_to_unique_nested_fields() -> None:
    configuration = {
        "model": {"hidden_size": 16},
        "parallel": {"tensor_model_parallel_size": 4},
    }

    resolved = apply_configuration_updates(
        configuration,
        {"hidden_size": 15, "tensor_model_parallel_size": 3},
    )

    assert resolved == {
        "hidden_size": "model.hidden_size",
        "tensor_model_parallel_size": "parallel.tensor_model_parallel_size",
    }
    assert configuration["model"]["hidden_size"] == 15
    assert configuration["parallel"]["tensor_model_parallel_size"] == 3


def test_rejects_ambiguous_leaf_updates() -> None:
    configuration = {"model": {"size": 1}, "parallel": {"size": 2}}

    with pytest.raises(ValueError, match="ambiguous"):
        apply_configuration_updates(configuration, {"size": 3})


def test_execution_manifest_resolves_relative_paths(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifests" / "run.json"
    manifest_path.parent.mkdir()
    manifest_path.write_text(
        json.dumps(
            {
                "baseline_config": "../baseline.json",
                "command": ["python", "oracle.py", "{config}"],
                "cwd": "..",
            }
        ),
        encoding="utf-8",
    )

    manifest = InterventionExecutionManifest.from_path(manifest_path)

    assert manifest.baseline_config == (tmp_path / "baseline.json").resolve()
    assert manifest.cwd == tmp_path.resolve()
