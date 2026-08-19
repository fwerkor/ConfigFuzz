# RQ2 GPU primary-seed results (2026-08-19)

This directory publishes the completed GPU primary-seed RQ2 records used for the current paper analysis. The frozen intent set uses seed 2026 and treats reaching `optimizer_step` while retaining the requested target value as Intent-Preserving Deep Execution (IPDE).

The package contains 21,600 records across PyTorch Native/CUDA, DeepSpeed, Transformers/Accelerate, and Megatron-Core. Each `*.jsonl.gz` file is a deterministic gzip of the final JSONL records; `manifest.json` pins both compressed and uncompressed SHA-256 hashes, while `summary.json` contains per-framework and aggregate metrics.

Megatron-Core required a selective rerun after the final RQ1 refinement changed its validated graphs. Raw Mutation, Native Validator, and Static-Hard reuse the original campaign because they do not consume the changed validated graph (Static-Hard uses the unchanged static graph). Constraint Filter, ConfigFuzz, and Global Repair were rerun against the current promotion in `experiments/rq2/promoted/gpu/`. The exact 1,350-case rerun plan is included as `megatron-v3-selective-plan.json.gz`.

Current headline numbers across 3,600 framework-local frozen intents are: Raw Mutation IPDE 67.78% (2,440/3,600), ConfigFuzz IPDE 64.75% (2,331/3,600), and ConfigFuzz generation 95.03% versus Static-Hard 90.56%. The paired Raw-vs-ConfigFuzz comparison contains 2,329 both-deep cases, 1,158 neither-deep cases, 111 Raw-only cases, and 2 ConfigFuzz-only rescues. Both rescues occur in Megatron-Core `fp16` intents, where ConfigFuzz coordinates `bf16` and advances an otherwise rejected mutation to completed execution.

The current GPU `behavior_signature` field is not used as evidence for the paper's runtime-diversity claim: this executor version derives it from milestone, outcome, and final error text rather than the branch/backend/topology behavior identifiers specified by the evaluation design.
