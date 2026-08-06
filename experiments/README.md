# ConfigFuzz Experiments

The evaluation follows one argument chain:

1. **RQ1** establishes that configuration constraints are complex, incompletely validated, and costly when violated.
2. **RQ2** tests whether constraint-guided coordinated mutation converts the same frozen mutation intents into more deep, valid, and diverse executions.
3. **RQ3** tests whether that additional deep execution improves historical bug replay and current-version bug discovery.

`protocol.yaml` is the machine-readable source of truth for methods, milestones, outcomes, controls, and metrics.

## Current non-accelerator results

- The RQ1 audit contains 93 constraints: 52 unary, 27 binary, 12 ternary, one four-parameter, and one five-parameter constraint; 26 have guards.
- Static mining over pinned MindSpeed-LLM, MindSpeed, and Megatron-LM trees extracted 733 raw native-validation candidates. After excluding test, example, and documentation evidence, 70 audited constraints have at least one implementation candidate and 46 have a candidate mentioning every participant. These are review leads, not coverage labels.
- Full Git history mining found 263 configuration-related fix candidates. The balanced RQ3 review shortlist contains 40 entries, split evenly between MindSpeed and MindSpeed-LLM.
- RQ2 intent generation is implemented, but the final intent list is deliberately not materialized until stable baseline configurations are bound for all three workload families.

## Pinned framework sources

The checked-in candidate artifacts were generated from:

- MindSpeed-LLM `79afbee24168aae3c30dedbec9ca04504a3204e4`;
- MindSpeed `8659565fc2dc7de3caba13acd97d4e8814f1df7e`;
- Megatron-LM `core_v0.12.1` at `a845aa7e12b3a117e24c2352b9e3e60bad2e3a17`.

## RQ1: audit and native coverage review

```bash
# Bootstrap the audit table. Coverage fields remain unreviewed.
configfuzz-experiment init-rq1-audit \
  corpus/lmsv/manual_constraints.yaml \
  experiments/rq1/constraint_audit.yaml

# Mine portable static candidates and build the per-constraint review queue.
python scripts/mine_rq1_native_candidates.py \
  --corpus corpus/lmsv/manual_constraints.yaml \
  --audit experiments/rq1/constraint_audit.yaml \
  --source MindSpeed-LLM=/path/to/MindSpeed-LLM \
  --source MindSpeed=/path/to/MindSpeed \
  --source Megatron-LM=/path/to/Megatron-LM \
  --output artifacts/rq1_mindspeed_static_candidates.json \
  --queue-output experiments/rq1/native_validation_candidates.json \
  --jobs 4 --limit 8

# Validate edits and produce descriptive statistics before execution.
configfuzz-experiment validate-rq1-audit experiments/rq1/constraint_audit.yaml
configfuzz-experiment summarize rq1 experiments/rq1/constraint_audit.yaml \
  --output experiments/rq1/bootstrap_summary.json

# Once satisfying/violating pairs have run, add failure-stage and resource-cost metrics.
configfuzz-experiment summarize rq1 experiments/rq1/constraint_audit.yaml \
  --runs experiments/results/rq1.jsonl \
  --output experiments/results/rq1-summary.json
```

`native_validation` and `first_affected_milestone` must be supported by source or execution evidence. Bootstrap category, semantic-class, and software-layer labels are deterministic annotation aids and also require review. A static candidate never automatically implies full or partial coverage.

## RQ2: workload binding and frozen intents

`rq2/workloads.yaml` declares the dense Transformer, GQA/long-sequence/FlashAttention, and MoE workload families. Bind each `baseline_config` to a small configuration that reliably completes training and checkpoint save/load before generating intents.

```bash
python scripts/generate_rq2_intents.py \
  --corpus corpus/lmsv/manual_constraints.yaml \
  --workloads experiments/rq2/workloads.yaml \
  --output experiments/rq2/intents.yaml \
  --frozen-output experiments/rq2/intents.frozen.yaml
```

The generator covers enum alternatives, numeric windows, divisibility boundaries and adjacent values, simple guard-enabling transitions, and TP/PP/EP/CP topology values. Review workload scope and retain at least 300 intents per workload before freezing. All comparison methods must consume the same frozen file.

After each workload also binds its audited dependency graph and native-validator manifest, expand every frozen intent into the five comparison cases:

```bash
python scripts/plan_rq2_campaign.py \
  --workloads experiments/rq2/workloads.yaml \
  --intents experiments/rq2/intents.frozen.yaml \
  --output experiments/rq2/campaign-plan.json
```

The plan records the exact target assignment, preflight mode, filter result, solver status, coordinated parameters, hard/unsupported constraints, and repair scope. It verifies the frozen intent hash and rejects baseline-ID mismatches before execution.

## RQ3: historical bug benchmark construction

```bash
# Mine fix-like commits that touch audited parameters and relevant framework files.
python scripts/mine_rq3_fix_candidates.py \
  --corpus corpus/lmsv/manual_constraints.yaml \
  --repository MindSpeed-LLM=/path/to/MindSpeed-LLM \
  --repository MindSpeed=/path/to/MindSpeed \
  --output artifacts/rq3_historical_fix_candidates.json

# Filter and diversify the manual source-review queue.
python scripts/build_rq3_triage_shortlist.py \
  artifacts/rq3_historical_fix_candidates.json \
  experiments/rq3/triage_shortlist.yaml \
  --limit 40 \
  --max-per-primary-parameter 5 \
  --max-patch-lines 2000

# Validate only after verified entries are copied into the benchmark registry.
configfuzz-experiment validate-bugs experiments/rq3/historical_bugs.yaml
```

A shortlist entry is not a benchmark bug. Admission requires a concrete configuration trigger, executable workload, non-performance failure oracle, at least three failures on the parent commit, passage on the fix commit, root-cause agreement, and a minimized configuration. Exact historical reproducers are reserved for final confirmation and are not supplied to the search methods.

## Campaign records and summaries

Each attempted configuration is one JSON object per line. Records include method, workload, frozen intent, seed, generation success, target-value preservation, coordinated parameters, solver cost, deepest milestone, outcome, wall time, GPU-seconds, peak memory, and diversity coverage. RQ1 records additionally identify the target constraint, satisfying/violating role, first failure, failure mode, and message quality. RQ3 records can include cumulative campaign position and buggy/fixed oracle evidence for exact first-reproducer costs.

```bash
configfuzz-experiment validate-runs experiments/results/rq2.jsonl
configfuzz-experiment summarize rq2 experiments/results/rq2.jsonl \
  --target-milestone optimizer_step \
  --output experiments/results/rq2-summary.json

configfuzz-experiment summarize rq3 experiments/results/rq3.jsonl \
  --bugs experiments/rq3/historical_bugs.yaml \
  --bug-split evaluation \
  --output experiments/results/rq3-summary.json

configfuzz-experiment fingerprint \
  --repository . \
  --repository /path/to/MindSpeed-LLM \
  --repository /path/to/MindSpeed \
  --repository /path/to/Megatron-LM \
  --output experiments/environment.json
```

## Dataset discipline

- Preserve every target assignment; filtering or repair must never silently replace the requested target value.
- Freeze mutation intents before running any method.
- Keep resource and infrastructure failures separate from validity evidence.
- Count RQ3 failures by independent root cause, not by failing configuration.
- Keep older verified bugs in the development split and reserve newer verified bugs for final evaluation.
