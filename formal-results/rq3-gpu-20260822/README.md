# RQ3 GPU results (2026-08-22)

This directory is the archival package for the completed GPU portion of RQ3. It contains the final current-version campaign, differential historical-bug confirmation, and the triage products used by the paper.

## Current-version campaign

The campaign executed 21,600 paired cases across DeepSpeed, Transformers/Accelerate, Megatron-Core, and PyTorch/CUDA with seed 2026. The six methods receive the same frozen mutation intents within each supported framework/workload pair. The pinned GPU stack is PyTorch 2.13.0, DeepSpeed 0.19.1, Megatron-Core 0.18.2, Transformers 5.9.0, and Accelerate 1.14.0.

| Outcome | Count |
| --- | ---: |
| valid | 14,878 |
| expected rejection | 3,387 |
| unexplained failure | 3,210 |
| unknown | 125 |
| infrastructure failure | 0 |
| resource failure | 0 |

`unexplained_failure` is a triage label, not a bug oracle. Deduplicating these rows across methods by framework, workload, target parameter, and target value produces 754 candidate configurations. Normalizing the last concrete framework exception produces 87 diagnostic signature clusters. Neither number is treated as an independent root-cause count.

Manual/minimal-reproducer triage yielded five public upstream reports from this campaign: three primary semantic--implementation mismatches (DeepSpeed #8276, DeepSpeed #8279, and Megatron-LM #6656) and two validation gaps (Transformers #48101 and #48169). ConfigFuzz triggers all three primary defects. The GQA validation gap in #48169 is exposed by raw mutation and native-validator-guided mutation because constraint-preserving methods retain the head-divisibility relation.

Pure rendezvous `EADDRINUSE` failures were preserved in `raw/infrastructure-failure-evidence.jsonl.gz` and rerun with the same case, seed, framework revision, and devices. The final primary JSONL streams contain zero infrastructure failures.

## Historical differential confirmation

Historical candidates are dynamically admitted: the attributed failure must reproduce on the buggy revision and disappear on the fixed revision. Five independent historical defects satisfy this rule. Across them, all 15 buggy runs reproduce the target oracle and all 15 fixed-control runs complete without it. DeepSpeed #7650 is retained under `historical/rejected-deepspeed-7650/` because its target failure did not reproduce on the buggy checkout.

The historical branch is differential confirmation of independently fixed bugs; the six-method comparison is the current-version campaign. No time-to-first or Kaplan--Meier statistic is inferred from the historical confirmation runs.

## Files

- `summary.json`: machine-readable campaign totals and source hashes.
- `current-version-by-framework.csv`: executor outcomes by framework.
- `current-version-by-method.csv`: executor outcomes by method.
- `unexplained-candidates.csv`: deduplicated configuration-level triage queue.
- `unexplained-signature-clusters.csv`: normalized diagnostic clusters.
- `current-version-defects.json`: five minimized/publicly reported root causes.
- `historical-benchmark.json`: admitted/rejected historical benchmark records.
- `raw/*.jsonl.gz`: canonical final result streams and preserved infrastructure evidence.
- `plans/*.json.gz`: frozen campaign plans.
- `historical/`: buggy/fixed replay evidence.
- `SHA256SUMS`: hashes for the archived package.

The raw streams retain absolute execution paths as provenance from the original runner. Compressed artifacts are included so the published summaries can be recomputed without the original machine.
