# ConfigFuzz Experiments

The evaluation follows one argument chain:

1. **RQ1** evaluates whether ConfigFuzz recovers properly scoped cross-layer configuration relations and whether execution evidence corrects incomplete or overly broad static candidates.
2. **RQ2** tests whether constraint-guided coordinated mutation converts the same frozen mutation intents into more deep, valid, and diverse executions.
3. **RQ3** tests whether that additional deep execution improves historical bug replay and current-version bug discovery.

`protocol.yaml` is the machine-readable source of truth for methods, milestones, outcomes, controls, and metrics.
`frameworks.yaml` records the execution stacks and their accelerator/backend
scope. The retained Ascend subjects are PTA and MSA; GPU coverage adds PyTorch
Native/CUDA, DeepSpeed, Megatron-Core, and Transformers/Accelerate. Method
comparisons remain paired within one framework/workload/environment, and
cross-framework aggregates are reported only after per-framework results.

## Current non-accelerator results

- The RQ1 audit contains 93 constraints: 52 unary, 27 binary, 12 ternary, one four-parameter, and one five-parameter constraint; 26 have guards.
- Static mining over pinned MindSpeed-LLM, MindSpeed, and Megatron-LM trees extracted 733 raw native-validation candidates. After excluding test, example, and documentation evidence, 70 audited constraints have at least one implementation candidate and 46 have a candidate mentioning every participant. These are review leads, not coverage labels.
- Primary source adjudication covers all 93 RQ1 records. For the 39 framework-legality constraints, it currently labels 13 full-explicit, 11 partial, eight implicit/delayed, and seven uncovered. This remains a primary review and requires independent confirmation before paper use.
- RQ2 now has canonical reduced profiles for Qwen2, Llama2, ChatGLM3, Mixtral, DeepSeek-V3, InternVL3, and CogVideoX. All seven pass accelerator-free architecture construction and CPU forward/backward smoke checks at the NPU-aligned 4-layer/hidden-512 scale.
- The primary prequalified RQ2 set contains exactly 1,050 method-independent intents (150 per workload). The framework--workload matrix contains 38 formal pairs; unsupported Megatron native model families are excluded rather than replaced by surrogates.
- RQ3 historical selection is frozen to fixes dated from 2025-08-01 through 2026-07-31. The source-admission queues contain 109 candidates without framework balancing or per-parameter caps: 67 GPU issues with an explicit fix link and 42 NPU fix commits mined from MindSpeed, MindSpeed-LLM, and MindSpeed-MM.
- Every frozen source candidate is attempted for buggy/fixed admission. A case enters the final historical benchmark only when its trigger can be expressed through configuration changes without changing model code or test input, the buggy revision fails three times, the fixed revision passes, and the observed failure matches the fix root cause. The previous 40-entry diversified shortlist is retained only as a legacy construction artifact and is not the formal RQ3 selection set.

## Completed accelerator results

- The completed Ascend RQ1 replication contains 10 relation targets and 20 satisfying/violating executions on two Ascend 910B3 devices with MindSpeed-LLM v26.1.0, MindSpeed 26.1.0_core_r0.12.1, Megatron-Core v0.12.1, and CANN 8.5.1. Seven relations are paired-confirmed, three are scope-disputed by valid violating counterexamples (`vocab-divisibility`, `ffn-hidden-size-cap`, and `moe-topk-one-pre-softmax`), and none remain unresolved. Raw records and the normalized summary are pinned under `npu/rq1/`.
- The final GPU RQ1 relation-validation campaign was expanded and rerun on August 19 using the NPU-aligned 4-layer/hidden-512 baselines. Across PyTorch Native, DeepSpeed, Transformers/Accelerate, and Megatron-Core, the frozen set contains 24 relation targets and 72 satisfying/violating/repaired executions. Sixteen targets are paired-confirmed, five are scope-disputed by valid counterexamples, and three remain unresolved. Megatron-Core contributes 11 topology-qualified targets: six are paired-confirmed, four are scope-disputed, and the `num_layers > 0` target remains unresolved because the violating case fails through a different construction path than the recovered provenance. DeepSpeed retains the reproducible `reduce_bucket_size=0` execution defect as unresolved, and the Transformers/Accelerate FP8 target remains hardware-limited. The normalized result is pinned in `gpu/formal_results_summary.yaml`.
- The completed primary-seed GPU RQ2 package contains 21,600 records across the four GPU stacks and is published under `rq2/results/gpu-primary-20260819/`. The final promotion uses the 24-target GPU RQ1 state plus model-stack relation recovery. The corrected Megatron-Core promotion includes the workload-level model relations and the framework-default `kv_channels=None` planning context while keeping the same 450 frozen intents and requested target values. Because its launched runtime runner had changed to `runtime-events-v1`, prior Megatron runtime evidence was not reused; all 578 unique corrected Megatron configurations were executed again with zero infrastructure failures. Across 3,600 framework-local frozen intents, Raw Mutation reaches IPDE for 2,440 cases (67.78%) and ConfigFuzz for 2,624 (72.89%). ConfigFuzz generates 98.72% of assigned cases versus 90.67% for Static-Hard, preserves every generated target assignment, and yields 184 paired Raw-failure-to-ConfigFuzz-deep rescues with zero paired regressions.

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

`rq2/baselines/canonical-v1/` defines seven architecture-faithful reduced workloads at the common Ascend/GPU campaign scale (4 layers, hidden size 512, FFN size 1024, sequence length 128). Family-specific GQA, MoE, MLA, vision-language, and CogVideoX components remain active. `scripts/preflight_rq2_models.py` validates all seven without accelerator access by constructing the real framework model class and completing a small CPU forward/backward pass.

`rq2/framework_workload_matrix.yaml` freezes the compatibility matrix. PTA, MSA, PyTorch, DeepSpeed, and Transformers/Accelerate cover all seven workload families; Megatron-Core enters formal RQ2 only for Qwen2, Llama2, and Mixtral, producing 38 formal framework--workload pairs. All 24 GPU pairs have completed accelerator qualification and are promoted under `rq2/promoted/gpu/`; the 14 PTA/MSA pairs remain pending Ascend qualification.

Generate the method-independent intent set from the canonical profiles and freeze exactly 150 intents per workload:

```bash
python scripts/generate_rq2_intents.py   --workloads experiments/rq2/canonical_workloads.prequalified.yaml   --output /tmp/configfuzz-rq2-candidates.yaml

python scripts/select_rq2_intents.py   --candidates /tmp/configfuzz-rq2-candidates.yaml   --workloads experiments/rq2/canonical_workloads.prequalified.yaml   --intent-pool method_independent   --output /tmp/configfuzz-rq2-selected.yaml   --frozen-output experiments/rq2/intents.prequalified.frozen.yaml
```

The primary set contains 1,050 intents and does not use recovered constraints to choose target values. Relation-derived divisibility boundaries, guard transitions, and cross-component stress cases remain in the separately reported `constraint_challenge` pool.

The primary statistical unit is one frozen intent. Because the six configured transformations are deterministic after intent freezing, each framework/workload/method pair uses seed 2026 once for the main comparison. A fixed SHA-256-selected 20% subset is additionally repeated with seeds 17, 42, 101, 2026, and 4099 as a separately reported execution-sensitivity analysis. Build the schedule with:

```bash
python scripts/build_experiment_schedule.py   --intents experiments/rq2/intents.prequalified.frozen.yaml   --matrix experiments/rq2/framework_workload_matrix.yaml   --output /tmp/configfuzz-rq2-schedule.json
```

The current prepared schedule contains 58,560 records across the six framework subjects. This is not an accelerator-launch count: planner-side `FILTERED` and `UNSAT` records do not launch a framework process.

`rq2/runtime_subjects.prequalified.yaml` retains the common prepared launchers, world sizes, timeouts, and supported workloads. The qualified GPU bindings, per-workload static/validated graphs, and native-validator settings are pinned under `rq2/promoted/gpu/` and are ready for formal RQ2 execution. PTA/MSA remain on the prequalified bindings until their accelerator qualification completes. If qualification changes a baseline schema, the 1,050 method-independent intents must be refrozen before the corresponding formal schedule is generated.

## RQ3: historical bug benchmark construction

```bash
# Mine the complete NPU fix-like set inside the frozen historical window.
python scripts/mine_rq3_fix_candidates.py \
  --corpus corpus/lmsv/manual_constraints.yaml \
  --repository MindSpeed-LLM=/path/to/MindSpeed-LLM \
  --repository MindSpeed=/path/to/MindSpeed \
  --repository MindSpeed-MM=/path/to/MindSpeed-MM \
  --since 2025-08-01 \
  --until 2026-07-31 \
  --output artifacts/rq3_historical_fix_candidates_2025-08_2026-07.json

# The formal source membership is frozen in:
#   rq3/npu_historical_source_queue.frozen.yaml  (42 candidates)
#   rq3/gpu_historical_source_queue.frozen.yaml  (67 candidates)
# and governed by rq3/historical_selection_policy.frozen.yaml.
# The old triage_shortlist/source_review files are legacy artifacts and do not
# determine benchmark membership.

# Attempt buggy/fixed admission for every frozen source candidate. Only verified
# cases are copied into the benchmark registry; the older replay_specs and
# execution_queue files describe the superseded capped-shortlist workflow.
configfuzz-experiment validate-bugs experiments/rq3/historical_bugs.yaml
```

A frozen source-queue entry is not yet a benchmark bug. Admission requires a concrete configuration trigger expressible without changing model code or test input, an executable workload, a non-performance failure oracle, at least three failures on the buggy revision, passage on the fixed revision, and root-cause agreement. Ordinary OOM, resource, and infrastructure failures do not admit a case. Exact historical reproducers are reserved for final confirmation and are not supplied to the search methods.

## Campaign records and summaries

Each attempted configuration is one JSON object per line. Records include method, workload, frozen intent, seed, generation success, target-value preservation, coordinated parameters, exact solver modifications, affected region, active constraint IDs, constraint status before/after execution, provenance IDs, refined constraints, solver cost, deepest milestone, outcome, wall time, accelerator-seconds, peak memory, stable runtime behavior IDs, and the behavior signature. RQ1 records additionally identify the target constraint, satisfying/violating role, first failure, failure mode, and message quality. RQ3 records can include cumulative campaign position, an independent root-cause ID, and buggy/fixed oracle evidence for exact first-reproducer costs.

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
