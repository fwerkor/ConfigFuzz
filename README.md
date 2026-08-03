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
- a machine-readable constraint and evidence model;
- a Python AST extractor for explicit guards, assertions, and lm-sv-style `_apply_fix` checks;
- a CLI for scanning a file or source tree for one or more parameters;
- a script for building an initial static inventory from the baseline;
- unit tests and a minimal GitHub Actions workflow;
- a research plan and a proposed constraint DSL.

Dynamic probing, oracle classification, counterexample-guided synthesis, and mutation integration are intentionally left as the next research stages.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,research]'
pytest tests/configfuzz
```

Scan the baseline validator for constraints on `hidden_size`:

```bash
python -m configfuzz scan \
  --parameter hidden_size \
  lmsv_rec/utils/runtime/mutate_and_forward/parallel_mutate/config_validator_moe.py
```

Scan several parameters and save JSON:

```bash
python -m configfuzz scan \
  --parameter tensor_model_parallel_size \
  --parameter num_attention_heads \
  --parameter moe_router_topk \
  --output artifacts/example_constraints.json \
  lmsv_rec
```

Build the first baseline inventory:

```bash
python scripts/build_lmsv_constraint_inventory.py
```

## Repository layout

```text
configfuzz/                 new constraint-inference prototype
  extractors/               static candidate extractors
scripts/                    experiment and inventory utilities
tests/configfuzz/           focused prototype tests
artifacts/                  generated research inventories
experiments/                experiment manifests and notes
docs/RESEARCH_PLAN.md       research questions and evaluation plan
docs/CONSTRAINT_DSL.md      supported constraint representation
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
