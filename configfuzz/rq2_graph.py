from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from configfuzz.corpus import ConstraintCorpus, RuleStatus, load_corpus
from configfuzz.dependencies import DependencyGraph, DependencyStatus
from configfuzz.model import Constraint, ConstraintSet, Evidence, EvidenceKind


def build_reviewed_manual_graph(corpus: ConstraintCorpus) -> DependencyGraph:
    rules = [
        rule
        for rule in corpus.rules
        if rule.status in {RuleStatus.REVIEWED, RuleStatus.VALIDATED}
    ]
    sets: list[ConstraintSet] = []
    for rule in rules:
        source = rule.sources[0]
        evidence = Evidence(
            kind=EvidenceKind.MANUAL,
            source=source.file,
            line=source.lines[0] if source.lines else None,
            detail=f"reviewed manual rule {rule.id}",
        )
        constraint = Constraint(
            expression=rule.expression,
            kind=rule.kind,
            parameters=rule.parameters,
            evidence=(evidence,),
            confidence=1.0,
        )
        sets.append(ConstraintSet(parameter=rule.parameters[0], constraints=[constraint]))

    raw = DependencyGraph.from_constraint_sets(
        sets,
        scope={"corpus": corpus.name, "rule_status": "reviewed_or_validated"},
    )
    rule_by_expression = {rule.expression: rule for rule in rules}
    edges = {}
    for edge in raw.edges.values():
        rule = rule_by_expression.get(edge.expression)
        if rule is None:
            raise ValueError(f"no manual rule found for graph edge {edge.expression!r}")
        status = (
            edge.status
            if edge.status is DependencyStatus.ENVIRONMENT_SPECIFIC
            else DependencyStatus.CONFIRMED
        )
        edges[rule.id] = replace(
            edge,
            id=rule.id,
            status=status,
            confidence=1.0,
        )
    return DependencyGraph(
        nodes=raw.nodes,
        edges=edges,
        metadata={
            **raw.metadata,
            "source": "reviewed_manual_constraint_corpus",
            "rule_count": len(rules),
        },
    )


def build_reviewed_manual_graph_from_path(path: str | Path) -> DependencyGraph:
    return build_reviewed_manual_graph(load_corpus(path))


def materialize_effective_campaign_baseline(
    candidate: Mapping[str, Any],
    corpus: ConstraintCorpus,
) -> dict[str, Any]:
    effective = candidate.get("effective_config")
    if not isinstance(effective, Mapping):
        raise ValueError("candidate workload must contain an effective_config object")

    # Keep leaf names as well because some lm-sv rules/intents are intentionally
    # unnamespaced. Add namespaced aliases only when the authoritative effective
    # configuration supplies the corresponding leaf value.
    baseline: dict[str, Any] = copy.deepcopy(dict(effective))
    for rule in corpus.rules:
        for parameter in rule.parameters:
            leaf = parameter.rsplit(".", 1)[-1]
            if leaf not in effective:
                continue
            _set_nested(baseline, parameter, copy.deepcopy(effective[leaf]))
    return baseline


def materialize_effective_campaign_baseline_from_paths(
    candidate_path: str | Path,
    corpus_path: str | Path,
) -> dict[str, Any]:
    import json

    candidate = json.loads(Path(candidate_path).read_text(encoding="utf-8"))
    if not isinstance(candidate, Mapping):
        raise ValueError("candidate workload root must be an object")
    return materialize_effective_campaign_baseline(candidate, load_corpus(corpus_path))


def _set_nested(root: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    if len(parts) == 1:
        root[parts[0]] = value
        return
    current = root
    for part in parts[:-1]:
        child = current.get(part)
        if child is None:
            child = {}
            current[part] = child
        elif not isinstance(child, dict):
            # A top-level scalar and namespaced object with the same prefix would
            # be ambiguous; fail instead of silently changing baseline semantics.
            raise ValueError(f"cannot materialize {path!r}: prefix {part!r} is scalar")
        current = child
    current[parts[-1]] = value
