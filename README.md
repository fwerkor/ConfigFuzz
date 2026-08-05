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
  conditional normalization, helper-function summaries, and false-positive filtering;
- declaration extractors for argparse, dataclass, `Literal`, field metadata,
  and schema-style YAML;
- a parallel CLI for scanning Python/YAML framework trees for one or more parameters;
- an explicit dependency hypergraph with deduplicated multi-parameter edges,
  direction, scope, status, connectivity queries, and active-constraint evaluation;
- a bounded joint-mutation planner for divisibility, alignment, equality,
  simple bounds, and Boolean dependencies;
- a Z3 joint solver that treats confirmed and environment-scoped edges as hard
  constraints and unconfirmed evidence as weighted soft constraints;
- an intervention designer that generates minimally different satisfying,
  violating, and repaired configurations for a selected dependency edge;
- an intervention execution adapter that patches nested configurations, runs
  each designed case, classifies the outcome, and matches rejection provenance;
- a bounded active-validation loop that repeatedly selects an unattempted edge,
  executes its intervention, updates graph evidence, and reranks the remainder;
- runtime feedback that separates consistency, isolated violations,
  provenance-matched paired interventions, and valid scope counterexamples;
- an isolated subprocess harness driven by JSON manifests;
- runtime outcome classification into valid, explicit invalid, resource,
  infrastructure, unexplained-failure, potential-bug, and unknown outcomes;
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

Every scan now embeds a dependency graph. It can also be rebuilt independently:

```bash
python -m configfuzz graph artifacts/example_constraints.json \
  --framework Megatron-LM \
  --version 42460a7 \
  --output artifacts/example_dependency_graph.json
```

Plan a joint mutation from a baseline configuration:

```bash
python -m configfuzz plan-mutation \
  artifacts/example_dependency_graph.json \
  examples/framework_static/baseline.json \
  --parameter tensor_model_parallel_size \
  --value 6 \
  --output artifacts/example_mutation_plan.json
```

Solve the same mutation over the supported graph expressions with Z3:

```bash
python -m configfuzz solve-mutation \
  artifacts/example_dependency_graph.json \
  examples/framework_static/baseline.json \
  --parameter tensor_model_parallel_size \
  --value 6 \
  --output artifacts/example_solver_plan.json
```

Apply labeled runtime samples to the graph before solving:

```bash
python -m configfuzz apply-feedback \
  artifacts/example_dependency_graph.json \
  examples/framework_static/feedback_samples.json \
  examples/framework_static/feedback_baseline.json \
  --output artifacts/example_feedback_graph.json
```

Design a provenance-carrying paired intervention for one candidate edge:

```bash
python -m configfuzz design-intervention \
  artifacts/example_dependency_graph.json \
  examples/framework_static/baseline.json \
  --edge dep-c47258679d480492 \
  --output artifacts/example_intervention.json
```

The output contains satisfying and violating configurations plus an optional
repaired counterpart. For guarded edges, both sides activate the guard so that
disabling the feature cannot masquerade as positive evidence.

Rank the next executable edges automatically instead of choosing an edge ID by
hand:

```bash
python -m configfuzz select-interventions \
  artifacts/lmsv_static_inventory.json \
  experiments/lmsv_validator_baseline.json \
  --limit 5 \
  --output artifacts/lmsv_intervention_queue.json
```

Each candidate exposes an explainable score decomposition covering validation
status, relation type, uncertainty, interaction degree, guard activation, graph
centrality, provenance, pair cost, and unsupported-expression cost. Edges for
which either polarity is unsatisfiable are excluded from the executable queue.
The queue can be executed directly:

```bash
python -m configfuzz run-intervention \
  artifacts/lmsv_intervention_queue.json \
  experiments/manifests/lmsv_hidden_size_intervention.json \
  --candidate-index 0 \
  --output artifacts/lmsv_selected_intervention_samples.json
```

The included lm-sv validator adapter can execute a complete confirmation loop:

```bash
python -m configfuzz design-intervention \
  artifacts/lmsv_static_inventory.json \
  experiments/lmsv_validator_baseline.json \
  --edge dep-c47258679d480492 \
  --output artifacts/lmsv_intervention.json

python -m configfuzz run-intervention \
  artifacts/lmsv_intervention.json \
  experiments/manifests/lmsv_hidden_size_intervention.json \
  --output artifacts/lmsv_intervention_samples.json

python -m configfuzz apply-feedback \
  artifacts/lmsv_static_inventory.json \
  artifacts/lmsv_intervention_samples.json \
  experiments/lmsv_validator_baseline.json \
  --output artifacts/lmsv_confirmed_graph.json
```

The execution manifest controls the command, baseline configuration, milestone
and failure oracles, and provenance patterns. Joint assignments are written to
a temporary configuration using exact paths or unambiguous leaf-name matching.

Run the adaptive loop for several rounds:

```bash
python -m configfuzz active-validate \
  artifacts/lmsv_static_inventory.json \
  experiments/manifests/lmsv_hidden_size_intervention.json \
  --rounds 10 \
  --output artifacts/lmsv_active_validation.json
```

Each round reranks the updated graph, executes the highest-ranked unattempted
edge, applies feedback, and carries the revised statuses into the next round.
The loop stops when its round budget is exhausted or no executable candidate
remains. Its output includes every selected intervention, runtime observation,
feedback report, attempted-edge list, stop reason, and final dependency graph.

Run a versioned scan against an external framework checkout:

```bash
python scripts/run_framework_static_scan.py /path/to/Megatron-LM \
  --source-subdir megatron \
  --name Megatron-LM \
  --parameter tensor_model_parallel_size \
  --parameter pipeline_model_parallel_size \
  --parameter hidden_size \
  --parameter num_attention_heads \
  --jobs 4 \
  --output artifacts/frameworks/megatron_lm_scan.json
```

The repository includes a result generated from Megatron-LM commit
`42460a7af821366e7115a162cb4410106bea93f0` at
`artifacts/frameworks/megatron_lm_42460a7.json`.

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
  active_validation.py     multi-round select/execute/feedback loop
  dependencies.py          dependency hypergraph and joint-mutation planner
  feedback.py              runtime evidence attribution and edge-state updates
  graph_solver.py          joint solver and paired-intervention designer
  intervention_runner.py   joint-config execution and provenance matching
  selection.py             adaptive executable-edge ranking
  extractors/               static candidate extractors
  outcomes.py               runtime outcome oracle
  probing.py                manifest, generator, and subprocess harness
  synthesis.py              Z3-backed constraint synthesis
scripts/                    experiment and inventory utilities
tests/configfuzz/           focused prototype tests
artifacts/                  generated research inventories and framework scans
corpus/lmsv/                normalized lm-sv manual constraints
experiments/                lm-sv runtime adapter and manifests
examples/                   deterministic runtime and framework-static fixtures
docs/RESEARCH_PLAN.md       research questions and evaluation plan
docs/CONSTRAINT_DSL.md      supported constraint representation
docs/STATIC_SCANNER.md      scanner semantics, normalization, and regression data
docs/DEPENDENCY_GRAPH.md    hypergraph, direction, queries, and mutation planning
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
- `RESOURCE_FAILURE`: resource exhaustion or capacity failure;
- `INFRASTRUCTURE_FAILURE`: cluster, launcher, network, or dependency failure;
- `UNEXPLAINED_FAILURE`: failure not yet attributable to validity or a defect;
- `UNKNOWN`: an otherwise inconclusive observation;
- `POTENTIAL_BUG`: the input satisfies current constraints but still exposes a system failure or inconsistency.

Without this distinction, a learner can incorrectly turn bug-triggering inputs into exclusion constraints and suppress the defects that fuzzing is meant to find.

## Scope of the first prototype

The first implementation targets Python configuration code and focuses on constraints that can be represented by a bounded DSL. It does not claim to recover complete parameter semantics from finite executions. The research objective is to infer constraints that are accurate enough to substantially improve valid-input rate and bug-finding effectiveness under a fixed testing budget.
