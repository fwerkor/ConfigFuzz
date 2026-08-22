# Ascend RQ1 results

This directory pins the completed two-NPU RQ1 relation-validation campaigns for both Ascend execution paths. The same 10 frozen recovered relations are exercised independently through MindSpeed-LLM/PTA and MindSpeed-LLM/MSA on two Ascend 910B3 devices (HCCL, world size 2). Each relation has a satisfying and a violating arm. A valid satisfying arm plus a provenance-matched rejection of the violating arm is recorded as `paired_confirmed`; a valid violating arm is `scope_disputed`.

Both backends produce the same evidence-state split: 7 paired-confirmed, 3 scope-disputed, and 0 unresolved. Across the two backend campaigns this gives 20 backend-specific targets and 40 executions, with 26 valid executions and 14 expected configuration rejections. The three scope-disputed relations are `vocab-divisibility`, `ffn-hidden-size-cap`, and `moe-topk-one-pre-softmax`. Backend observations are retained separately and are not treated as independent semantic ground truth.

PTA uses MindSpeed-LLM v26.1.0 / MindSpeed 26.1.0_core_r0.12.1 / Megatron-Core v0.12.1 on CANN 8.5.1. MSA uses MindSpeed-LLM v26.1.0 with MindSpore 2.10.0, MSAdapter 0.7.0, and CANN 9.1.0. Ascend MindSpeed/MindSpeed-LLM source provenance is pinned to GitCode; MSAdapter uses the upstream OpenI source specified by the integration stack. The MSA compatibility patches checked in under `msa-compat-patches/` adapt runtime integration only and do not change the tested relation predicates or participant assignments.

Raw records are stored in `rq1.26.1.dual-npu.jsonl` (PTA) and `rq1.msa.26.1.dual-npu.final.jsonl` (MSA). `formal_results_summary.yaml` records per-backend software provenance, SHA-256 hashes, target-level classifications, and the combined backend-specific aggregate.
