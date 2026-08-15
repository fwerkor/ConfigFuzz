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

Megatron-Core uses `baselines/megatron-core.json` because its qualified local
attention path currently uses FP32 and TP=2. Mixed-precision variants remain
runtime test targets instead of qualification prerequisites.

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
