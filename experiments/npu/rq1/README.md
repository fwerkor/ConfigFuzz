# Ascend RQ1 results

This directory pins the completed two-NPU RQ1 relation-validation campaign run on 2026-08-21. The campaign exercises 10 frozen recovered relations on two Ascend 910B3 devices (HCCL, world size 2) using MindSpeed-LLM v26.1.0 / MindSpeed 26.1.0_core_r0.12.1 / Megatron-Core v0.12.1 on CANN 8.5.1.

Each relation has a satisfying and a violating arm. A valid satisfying arm plus a provenance-matched rejection of the violating arm is recorded as `paired_confirmed`. If the violating arm completes successfully, the recovered relation is `scope_disputed` and must not be hardened as a universal requirement.

The completed campaign contains 20 executions over 10 relation targets: 7 paired-confirmed, 3 scope-disputed, and 0 unresolved. The three scope-disputed relations are `vocab-divisibility`, `ffn-hidden-size-cap`, and `moe-topk-one-pre-softmax`. The raw JSONL is retained alongside the normalized summary for traceability.
