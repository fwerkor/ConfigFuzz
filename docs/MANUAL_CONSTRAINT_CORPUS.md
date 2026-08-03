# Normalized lm-sv Manual Constraint Corpus

## Purpose

The existing lm-sv rules are useful research material, but they are distributed across validators, mutation pools, model-specific branches, repair logic, and environment assumptions. ConfigFuzz normalizes them into one machine-readable corpus:

```text
corpus/lmsv/manual_constraints.yaml
```

This corpus is **not** the source input used to infer constraints from a framework. Its roles are:

1. a reviewed inventory of the rules currently enforced by lm-sv;
2. an evaluation reference for measuring whether ConfigFuzz recovers equivalent rules from framework source and runtime behavior;
3. a baseline for comparing inferred constraints with manual mutation guidance;
4. a migration format for eventually replacing scattered handwritten rules.

## Why rule strength is explicit

A rule present in lm-sv is not automatically a universal framework requirement. Each record is classified as one of:

- `framework_requirement`: believed to follow from the framework or model implementation;
- `lmsv_policy`: a deliberate mutation-space restriction chosen by lm-sv;
- `environment_limit`: valid only for a particular tested environment or backend;
- `resource_limit`: derived from available memory, devices, or runtime budget;
- `workaround`: compensates for a known unsupported or defective implementation path;
- `empirical`: observed experimentally but not yet justified semantically;
- `unknown`: not yet classified.

This distinction prevents evaluation leakage. ConfigFuzz should be rewarded for recovering framework requirements, but it should not be expected to infer arbitrary lm-sv budget choices such as a manually selected batch-size cap.

## Record format

```yaml
- id: lmsv.task1.hidden-size-tp-divisibility
  expression: model.hidden_size % parallel.tensor_model_parallel_size == 0
  kind: relation
  parameters:
    - model.hidden_size
    - parallel.tensor_model_parallel_size
  enforcement: repair
  strength: framework_requirement
  status: reviewed
  scope:
    tasks: ["1"]
    backends: [PTA]
    frameworks: [MindSpeed-LLM, Megatron-LM]
    stage: configuration-validation
  sources:
    - file: lmsv_rec/utils/runtime/mutate_and_forward/parallel_mutate/config_validator_moe.py
      lines: [198, 207]
      symbol: EnhancedMegatronConfigValidator._check_hidden_size
      source_type: manual_validator
  rationale: Tensor parallel sharding requires the hidden dimension to divide evenly across TP ranks.
  repair:
    target: model.hidden_size
    strategy: nearest_divisible
    divisor: parallel.tensor_model_parallel_size
```

The schema is defined in:

```text
schemas/manual_constraint.schema.json
```

## Core fields

| Field | Meaning |
| --- | --- |
| `id` | Stable rule identifier used in experiments and annotations |
| `expression` | Normalized constraint DSL expression |
| `kind` | Type, range, enum, relation, conditional, environment, resource, or other |
| `parameters` | All configuration or environment symbols referenced by the rule |
| `enforcement` | Whether lm-sv rejects, repairs, warns, samples, or supplies a default |
| `strength` | Semantic origin of the rule rather than its implementation behavior |
| `status` | Candidate, reviewed, validated, contradicted, or deprecated |
| `scope` | Task, model, backend, framework, hardware, stage, and activation condition |
| `sources` | Exact implementation or pool provenance |
| `repair` | Machine-readable repair strategy when enforcement is `repair` |
| `rationale` | Human-reviewed explanation of why the rule exists |

## Current corpus

The first version contains two groups in the same schema:

- Task1 validator rules from `config_validator_moe.py`;
- Task6 mutation-window and enumeration rules from `mutable_params_pool.yaml`.

The build script regenerates the committed corpus deterministically:

```bash
python scripts/build_lmsv_manual_corpus.py
```

Check that the committed corpus still matches the baseline sources:

```bash
python scripts/build_lmsv_manual_corpus.py --check
python -m configfuzz validate-corpus corpus/lmsv/manual_constraints.yaml
```

## Review workflow

A rule should move through these states:

```text
candidate -> reviewed -> validated
                   \-> contradicted
                   \-> deprecated
```

`reviewed` means the normalized record faithfully represents the current lm-sv code. It does not mean the rule has been independently confirmed as a framework requirement. `validated` should only be assigned after framework-source evidence, framework documentation, or controlled runtime experiments support it.

## Evaluation use

For leave-one-rule-out evaluation:

1. select a `framework_requirement` record;
2. hide the corresponding lm-sv validator code and corpus entry from the inference process;
3. run ConfigFuzz on the actual framework source and runtime adapter;
4. normalize the inferred rule into the same DSL;
5. compare semantic agreement over generated configurations, not only string equality.

Rules marked `lmsv_policy`, `environment_limit`, or `workaround` should be evaluated separately because their expected inference evidence differs.
