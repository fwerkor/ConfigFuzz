#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from configfuzz.corpus import load_corpus
from configfuzz.rq2_graph import (
    build_reviewed_manual_graph,
    materialize_effective_campaign_baseline,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare reviewed manual dependency graph and effective baseline for RQ2."
    )
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--graph-output", type=Path, required=True)
    parser.add_argument("--baseline-output", type=Path, required=True)
    args = parser.parse_args()

    corpus = load_corpus(args.corpus)
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    graph = build_reviewed_manual_graph(corpus)
    baseline = materialize_effective_campaign_baseline(candidate, corpus)

    args.graph_output.parent.mkdir(parents=True, exist_ok=True)
    args.baseline_output.parent.mkdir(parents=True, exist_ok=True)
    args.graph_output.write_text(
        json.dumps(graph.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.baseline_output.write_text(
        json.dumps(baseline, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "graph_output": str(args.graph_output),
                "baseline_output": str(args.baseline_output),
                "rule_count": len(graph.edges),
                "node_count": len(graph.nodes),
                "baseline_leaf_count": len(candidate.get("effective_config", {})),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
