# ConfigFuzz Experiments

The evaluation follows one argument chain:

1. **RQ1** establishes that configuration constraints are complex, incompletely validated, and costly when violated.
2. **RQ2** tests whether constraint-guided coordinated mutation converts the same frozen mutation intents into more deep, valid, and diverse executions.
3. **RQ3** tests whether that additional deep execution improves historical bug replay and current-version bug discovery.

`protocol.yaml` is the machine-readable source of truth for methods, milestones, outcomes, controls, and metrics.

## Current non-accelerator results

- The RQ1 audit contains 93 constraints: 52 unary, 27 binary, 12 ternary, one four-parameter, and one five-parameter constraint; 26 have guards.
- Static mining over pinned MindSpeed-LLM, MindSpeed, and Megatron-LM trees extracted 733 raw native-validation candidates. After excluding test, example, and documentation evidence, 70 audited constraints have at least one implementation candidate and 46 have a candidate mentioning every participant. These are review leads, not coverage labels.
- Primary source adjudication covers all 93 RQ1 records. For the 39 framework-legality constraints, it currently labels 13 full-explicit, 11 partial, eight implicit/delayed, and seven uncovered. This remains a primary review and requires independent confirmation before paper use.
- RQ2 source candidates are pinned to lm-sv `e73ba3d355152a5711e2f80c1fb0d166f2ba1496`. Command-template overrides and derived parallel values are merged into an auditable `effective_config`.
- The RQ2 generator produced 477 dense-Qwen2, 453 GQA/long-sequence ChatGLM3, and 516 Mixtral-MoE unique candidate intents. Deterministic parameter-balanced selection freezes 300 per primary workload, for a 900-intent candidate set with SHA-256 `1dd5071213f8be7cc5daac691651806eed370c33474b7ba83f3c146b07b21d5d`.
- Full Git history mining found 263 configuration-related fix candidates. Source review retained 23 of the balanced 40-entry shortlist for buggy/fixed execution, deferred eight, and excluded nine.
- RQ3 replay planning identifies three harnesses present on both revisions, five fixed-side tests that can be backported as test code only, 13 candidates requiring a minimal harness, and two candidates requiring root-cause separation. No candidate is yet counted as a verified historical bug.

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

# Apply the checked-in primary adjudication without overwriting the untouched bootstrap table.
python scripts/apply_rq1_adjudication.py \
  experiments/rq1/constraint_audit.yaml \
  experiments/rq1/primary_adjudication.yaml \
  experiments/rq1/constraint_audit.primary.yaml \
  --require-complete

python scripts/validate_rq1_evidence.py \
  experiments/rq1/constraint_audit.primary.yaml \
  --repository-root . \
  --source-root MindSpeed-LLM=/path/to/MindSpeed-LLM \
  --source-root MindSpeed=/path/to/MindSpeed \
  --source-root Megatron-LM=/path/to/Megatron-LM

# Build a secondary-review packet with all primary labels and interpretations hidden.
python scripts/prepare_rq1_secondary_review.py \
  --audit experiments/rq1/constraint_audit.yaml \
  --primary-adjudication experiments/rq1/primary_adjudication.yaml \
  --packet-output experiments/rq1/secondary_review_packet.yaml \
  --template-output experiments/rq1/secondary_review_template.yaml

# If a second independent audit is performed, fill the template and compute
# agreement plus the entries that require adjudication. No second audit is
# assumed by the experiment protocol itself.
python scripts/compare_rq1_reviews.py \
  --primary-adjudication experiments/rq1/primary_adjudication.yaml \
  --secondary-review experiments/rq1/secondary_review.completed.yaml \
  --output experiments/rq1/review_agreement.json

# Once satisfying/violating pairs have run, add failure-stage and resource-cost metrics.
configfuzz-experiment summarize rq1 experiments/rq1/constraint_audit.yaml \
  --runs experiments/results/rq1.jsonl \
  --recovered-model artifacts/lmsv_active_validation.json \
  --output experiments/results/rq1-summary.json
```

`native_validation` and `first_affected_milestone` must be supported by source or execution evidence. Bootstrap category, semantic-class, and software-layer labels are deterministic annotation aids and also require review. A static candidate never automatically implies full or partial coverage.

## RQ2: workload binding and frozen intents

`rq2/workload_candidates.source.yaml` pins candidate lm-sv model configurations and command templates for Qwen2, Llama2, ChatGLM3, Mixtral, DeepSeek-V3, InternVL3, and CogVideoX. Candidate snapshots remain unverified until each one completes an optimizer step, repeated training, and checkpoint save/load on the target stack.

```bash
# Materialize command-aware candidate baselines from the pinned lm-sv revision.
python scripts/prepare_rq2_workload_candidates.py \
  --source-spec experiments/rq2/workload_candidates.source.yaml \
  --source-root /path/to/lm-sv \
  --output-dir experiments/rq2/candidates \
  --registry-output experiments/rq2/candidate_workloads.yaml

# Generate a larger unique candidate pool.
python scripts/generate_rq2_intents.py \
  --workloads experiments/rq2/candidate_workloads.yaml \
  --output experiments/rq2/candidate_intents.yaml

# Select exactly 150 method-independent intents per primary workload and freeze the review candidate set.
python scripts/select_rq2_intents.py \
  --candidates experiments/rq2/candidate_intents.yaml \
  --workloads experiments/rq2/candidate_workloads.yaml \
  --intent-pool method_independent \
  --output experiments/rq2/selected_candidate_intents.yaml \
  --frozen-output experiments/rq2/selected_candidate_intents.frozen.yaml
```

The generator emits two explicitly labeled pools. The primary `method_independent` pool is built only from scalar parameters exposed by the qualified baseline, generic numeric/Boolean boundary grids, and TP/PP/EP/CP topology values; recovered constraints and the legacy rule corpus do not choose its target values. The separate `constraint_challenge` pool contains relation-derived boundaries, guard transitions, and other constraint-focused stress cases. Select and freeze the primary pool for the main RQ2 comparison, and report the challenge pool separately if it is used. The checked-in frozen file is a review candidate only; regenerate the final frozen set after accelerator validation. All comparison methods consume the same final frozen file.

Generate the optional challenge pool with `--include-constraint-challenge --corpus corpus/lmsv/manual_constraints.yaml`; this flag never changes the method-independent intents.

After each workload also binds both the pre-validation static graph and the execution-validated dependency graph, together with its native-validator manifest, expand every frozen intent into the six comparison cases:

```bash
python scripts/plan_rq2_campaign.py \
  --workloads experiments/rq2/workloads.yaml \
  --intents experiments/rq2/intents.frozen.yaml \
  --output experiments/rq2/campaign-plan.json
```

The plan records the exact target assignment, preflight mode, filter result, solver status, coordinated parameters, hard/unsupported constraints, repair scope, and constraint-treatment policy. `static_hard_configfuzz` hardens every static candidate and ignores validation status; normal `configfuzz` and `global_repair` hard-enforce only confirmed/environment-specific relations and use unresolved candidates as confidence-tiered soft guidance. The plan verifies the frozen intent hash and rejects baseline-ID mismatches before execution.

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

# Validate the 40-entry source review and its 23-entry execution queue.
python scripts/validate_rq3_source_review.py \
  experiments/rq3/source_review.yaml \
  --execution-queue experiments/rq3/execution_queue.yaml

# Verify historical commits and harness blobs, then build unexecuted replay specifications.
python scripts/build_rq3_replay_specs.py \
  --execution-queue experiments/rq3/execution_queue.yaml \
  --source-plan experiments/rq3/replay_plan.source.yaml \
  --repository-root MindSpeed-LLM=/path/to/MindSpeed-LLM \
  --repository-root MindSpeed=/path/to/MindSpeed \
  --output experiments/rq3/replay_specs.yaml

# Validate only after verified entries are copied into the benchmark registry.
configfuzz-experiment validate-bugs experiments/rq3/historical_bugs.yaml
```

A shortlist entry is not a benchmark bug. Admission requires a concrete configuration trigger, executable workload, non-performance failure oracle, at least three failures on the parent commit, passage on the fix commit, root-cause agreement, and a minimized configuration. Exact historical reproducers are reserved for final confirmation and are not supplied to the search methods.

## Campaign records and summaries

Each attempted configuration is one JSON object per line. Records include method, workload, frozen intent, seed, generation success, target-value preservation, coordinated parameters, exact solver modifications, affected region, active constraint IDs, constraint status before/after execution, provenance IDs, refined constraints, solver cost, deepest milestone, outcome, wall time, GPU-seconds, peak memory, stable runtime behavior IDs, and the behavior signature. RQ1 records additionally identify the target constraint, satisfying/violating role, first failure, failure mode, and message quality. RQ3 records can include cumulative campaign position, an independent root-cause ID, and buggy/fixed oracle evidence for exact first-reproducer costs.

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
