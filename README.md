# ConfigFuzz

ConfigFuzz is a research prototype for automatically inferring configuration-parameter constraints and using them to guide mutation-based testing of large-model systems.

The repository starts from a clean snapshot of `fwerkor/lm-sv` at commit `e73ba3d35` (`dev_0.1.0`). The original testing system remains available as the experimental baseline, while new constraint-inference components live in the top-level `configfuzz/` package.

## Research problem

The baseline system mutates configuration parameters, but each parameter is only meaningful inside a valid domain. Those domains currently come from repeated manual testing and are encoded across parameter pools, validators, model-specific branches, scripts, and environment assumptions.

ConfigFuzz studies whether a new parameter and its software context are sufficient to automatically recover a useful constraint specification:

```text
new parameter + source context
            |
            v
static candidate extraction
            |
            v
active runtime probing
            |
            v
constraint synthesis and validation
            |
            v
constraint-aware mutation
```

The intended output is not limited to a numeric interval. A parameter may have type, range, enumeration, divisibility, relational, conditional, model-specific, environment, and resource constraints.

## Current status

The initial repository provides:

- the lm-sv implementation as a baseline and source of manually encoded constraints;
- a normalized manual-rule corpus that separates framework requirements, lm-sv policies, environment limits, and workarounds;
- a machine-readable constraint and evidence model;
- a strict-by-default Python AST extractor with scoped symbolic propagation,
  conditional normalization, and false-positive filtering;
- a parallel CLI for scanning a framework source tree for one or more parameters;
- an isolated subprocess harness driven by JSON manifests;
- runtime outcome classification into `VALID`, `INVALID`, `UNKNOWN`, and
  `POTENTIAL_BUG`;
- boundary-oriented probe generation with model/environment context;
- a Z3-backed synthesizer for ranges, enums, divisibility, and contextual
  relations;
- a script for building an initial static inventory from the baseline;
- a lightweight adapter that exposes the real lm-sv validator as a runtime
  oracle;
- unit tests and a minimal GitHub Actions workflow;
- a research plan and a proposed constraint DSL.

The current prototype performs one batch of static or dynamic inference. A
full counterexample-guided loop, multi-parameter synthesis, resource modeling,
and mutation integration remain research stages.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest tests/configfuzz
```

Scan framework source for constraints on `hidden_size`:

```bash
python -m configfuzz scan \
  --parameter hidden_size \
  --jobs 0 \
  /path/to/framework
```

The lm-sv validator can still be used as a regression fixture, but it is not
the formal source input for ConfigFuzz evaluation.

Scan several parameters and save JSON:

```bash
python -m configfuzz scan \
  --parameter tensor_model_parallel_size \
  --parameter num_attention_heads \
  --parameter moe_router_topk \
  --jobs 0 \
  --output artifacts/example_constraints.json \
  /path/to/framework
```

Strict mode is the default. Use `--broad` only for exploratory, recall-oriented
scans whose unsupported expressions will be reviewed manually.

Build the first baseline inventory:

```bash
python scripts/build_lmsv_constraint_inventory.py
```

Validate the normalized manual-rule corpus:

```bash
python scripts/build_lmsv_manual_corpus.py --check
python -m configfuzz validate-corpus corpus/lmsv/manual_constraints.yaml
```

Run the complete runtime pipeline on a deterministic demonstration target:

```bash
python -m configfuzz infer examples/toy_parallel_size.json \
  --samples-output artifacts/toy_samples.json \
  --output artifacts/toy_spec.json
```

The inferred result is:

```text
hidden_size % parallel_size == 0
```

Run the same pipeline against the actual lm-sv configuration validator:

```bash
python -m configfuzz infer experiments/manifests/lmsv_hidden_size.json \
  --samples-output artifacts/lmsv_hidden_size_samples.json \
  --output artifacts/lmsv_hidden_size_dynamic_spec.json
```

For the included baseline and probe set, ConfigFuzz recovers:

```text
hidden_size % tensor_model_parallel_size == 0
```

The validator accepts `hidden_size = 0` unchanged, so the dynamic result does
not invent a positivity constraint that is absent from the observed oracle.
This distinction is useful when separating recovered implementation behavior
from intended framework semantics.

## Repository layout

```text
configfuzz/                 new constraint-inference prototype
  extractors/               static candidate extractors
  outcomes.py               runtime outcome oracle
  probing.py                manifest, generator, and subprocess harness
  synthesis.py              Z3-backed constraint synthesis
scripts/                    experiment and inventory utilities
tests/configfuzz/           focused prototype tests
artifacts/                  generated research inventories
corpus/lmsv/                normalized lm-sv manual constraints
experiments/                lm-sv runtime adapter and manifests
examples/                   deterministic end-to-end demonstration
docs/RESEARCH_PLAN.md       research questions and evaluation plan
docs/CONSTRAINT_DSL.md      supported constraint representation
docs/STATIC_SCANNER.md      scanner semantics, normalization, and regression data
docs/RUNTIME_INFERENCE.md   manifest and runtime pipeline reference
docs/MANUAL_CONSTRAINT_CORPUS.md
                            corpus schema, semantics, and review workflow
docs/LMSV_BASELINE.md       original lm-sv README
lmsv_rec/, mm-new/, ...     baseline implementation
```

## Design principle

A failed run is not automatically evidence that a parameter value is invalid. Runtime outcomes must distinguish:

- `VALID`: the configured execution reaches the selected validation stage;
- `INVALID`: an explicit configuration oracle rejects the value or combination;
- `UNKNOWN`: infrastructure, timeout, dependency, or unrelated failure;
- `POTENTIAL_BUG`: the input satisfies current constraints but still exposes a system failure or inconsistency.

Without this distinction, a learner can incorrectly turn bug-triggering inputs into exclusion constraints and suppress the defects that fuzzing is meant to find.

## Scope of the first prototype

The first implementation targets Python configuration code and focuses on constraints that can be represented by a bounded DSL. It does not claim to recover complete parameter semantics from finite executions. The research objective is to infer constraints that are accurate enough to substantially improve valid-input rate and bug-finding effectiveness under a fixed testing budget.
