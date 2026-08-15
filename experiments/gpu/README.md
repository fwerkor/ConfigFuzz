# GPU Framework Evaluation

This directory contains the GPU-side qualification harness used before the
cross-framework ConfigFuzz campaign. Qualification is deliberately separate
from the frozen RQ2 comparison: a framework--workload pair must first reach
argument parsing, distributed initialization, model construction, forward,
backward, optimizer step, repeated training, and checkpoint save/load without a
mutation.

## Pinned subjects

The current qualification revisions and observed status are recorded in
`qualification/qualified_subjects.yaml`. The evaluated GPU stacks are:

- PyTorch Native/CUDA;
- DeepSpeed;
- Megatron-Core;
- Transformers/Accelerate.

All comparisons must remain paired within one framework, workload, software
revision, and hardware environment. GPU results are reported separately from
the Ascend PTA/MSA campaign before any aggregate is computed.

## Qualification workload

`qualification/baseline.json` defines a small Llama-like workload used to
exercise the complete execution lifecycle without downloading pretrained
weights. Framework-specific runners instantiate the equivalent workload and
emit stable milestone markers of the form
`CONFIGFUZZ_MILESTONE:<milestone>`.

Megatron-Core uses `baselines/megatron-core.json` with a Llama-like RMSNorm
configuration and TP=2. Its qualification runner invokes Megatron's native
training-argument validation before model construction and uses the framework's
`Float16Module` path for FP16/BF16 cases. The isolated Megatron environment also
installs `requirements-megatron-validation.txt` so the native training argument
stack can load its YAML configuration support.

## Static recovery

Versioned framework scans are generated with the built-in profiles:

```bash
python scripts/run_framework_static_scan.py /path/to/pytorch \
  --profile pytorch-cuda --jobs 4 \
  --output artifacts/frameworks/pytorch_v2.13.0.json

python scripts/run_framework_static_scan.py /path/to/DeepSpeed \
  --profile deepspeed --jobs 4 \
  --output artifacts/frameworks/deepspeed_v0.19.1.json

python scripts/run_framework_static_scan.py /path/to/Megatron-LM \
  --profile megatron-core --jobs 4 \
  --output artifacts/frameworks/megatron_core_v0.18.2.json

python scripts/run_framework_static_scan.py /path/to/transformers /path/to/accelerate \
  --profile transformers-accelerate --jobs 4 \
  --output artifacts/frameworks/transformers_v5.9.0_accelerate_v1.14.0.json
```

These artifacts are static candidates. They are not promoted to semantic hard
constraints solely because the scanned implementation contains a rejecting
branch.

## Execution-guided validation

All four qualified GPU stacks are connected to the same active-validation
loop. Each subject has a dedicated baseline, launcher, and execution manifest;
the launchers keep machine-specific Python/source paths configurable through
environment variables while the manifests remain repository-relative.

Run one subject with the unified wrapper:

```bash
bash experiments/gpu/run_active_validation.sh pytorch-native 3
bash experiments/gpu/run_active_validation.sh deepspeed 3
bash experiments/gpu/run_active_validation.sh transformers-accelerate 3
bash experiments/gpu/run_active_validation.sh megatron-core 3
```

The wrapper dispatches to the equivalent direct command, for example:

```bash
python -m configfuzz active-validate \
  artifacts/frameworks/megatron_core_v0.18.2.json \
  experiments/gpu/manifests/megatron-core.json \
  --rounds 3 \
  --output artifacts/gpu/results/megatron-active-validation.json
```

Each round selects an executable recovered edge, constructs satisfying,
violating, and repaired configurations, executes them on the qualified runtime,
and updates only the evidence supported by the observations. Preliminary
qualification/active-validation runs must not be mixed with the final frozen
RQ2 campaign.

The framework baselines expose only configuration fields that are actually
consumed by the corresponding runner. In particular, the PyTorch runner drives
the recovered distributed backend and subgroup-size paths, while the DeepSpeed
runner passes recovered batch-size and ZeRO bucket settings into the native
DeepSpeed configuration object. This keeps execution feedback tied to the
scanned framework semantics rather than to a synthetic validation shim.

## Frozen cross-framework target set

The formal GPU generalization campaign uses `validation_targets.frozen.yaml`.
Targets are ranked and frozen from each versioned static graph plus its qualified
effective baseline before formal execution feedback is considered. Candidates
with an explicit execution-stage scope that does not match the qualified workload
are excluded before ranking. This prevents later runtime outcomes from changing
the evaluated target set. The current frozen set contains 19 targets across the
four GPU stacks and records hashes for each source artifact, baseline, execution
manifest, launch script, qualification runner, and shared runner source.

Regenerate and verify the frozen set with:

```bash
python scripts/freeze_gpu_validation_targets.py \
  --output experiments/gpu/validation_targets.frozen.yaml \
  --limit-per-subject 6 --solver-timeout-ms 3000
```

Run one framework against the exact frozen intervention plans with:

```bash
python scripts/run_frozen_gpu_validation.py \
  experiments/gpu/validation_targets.frozen.yaml deepspeed \
  --output artifacts/gpu/frozen-results/deepspeed.json
```

The runner rejects changed baselines, manifests, or static artifacts by SHA-256
before executing a target. Frozen interventions are executed independently and
runtime feedback is aggregated only after all targets finish, so confirmation
counts do not depend on target order. Preliminary active-validation files are
kept separate from these frozen-campaign outputs.

The normalized result summary for the frozen campaign is recorded in
`formal_results_summary.yaml`; raw process logs remain outside the repository
because they contain machine-local execution paths. The summary records the
ConfigFuzz runner revision and SHA-256 of every raw subject result. Rebuild it
from the raw outputs with `scripts/summarize_frozen_gpu_validation.py`. The
current `df3ecc39...` campaign contains 19 targets and 57 executions: 15 targets
form provenance-matched satisfying/violating/repaired evidence, one target is
scope-disputed, and three remain unresolved under the qualified environments.
