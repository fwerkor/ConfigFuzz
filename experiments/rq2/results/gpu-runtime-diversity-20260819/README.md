# RQ2 GPU runtime-diversity subset (corrected 2026-08-20)

This package contains the deterministic 20% runtime-instrumented subset of the frozen RQ2 intents: 720 framework-local intents and 4,320 method records. The Megatron-Core portion uses the same 90 frozen intent IDs as the August 19 subset, with corrected workload/model-stack planning and `kv_channels=None` context.

Raw Mutation reaches IPDE for 520/720 (72.22%), while ConfigFuzz reaches 548/720 (76.11%), with 28 ConfigFuzz-only rescues and zero Raw-only regressions. Both methods cover 26 executed-path behavior IDs and 39 behavior signatures. Signature entropy is 4.575 bits for ConfigFuzz and 4.654 bits for Raw Mutation.

The subset uses `runtime-events-v1` rank-zero instrumentation. Exact materialized-configuration accounting yields 854 unique runtime configurations across the four framework-local subset plans. The final package contains zero infrastructure-failure records.
