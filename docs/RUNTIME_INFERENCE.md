# Runtime inference

ConfigFuzz executes a target through a JSON manifest, classifies every result,
and synthesizes the smallest supported conjunction that accepts all observed
valid values while rejecting as many explicit invalid values as possible.

## Manifest

```json
{
  "parameter": "parallel_size",
  "parameter_type": "integer",
  "baseline": 1,
  "values": [0, 1, 2, 3, 4, 8],
  "context": {
    "hidden_size": 32
  },
  "command": [
    "python",
    "target.py",
    "--parallel-size",
    "{value}"
  ],
  "timeout_seconds": 10,
  "classification": {
    "invalid_patterns": ["CONFIG_INVALID:"],
    "infrastructure_patterns": ["connection reset"],
    "bug_patterns": ["BUG_ORACLE:"],
    "milestone_patterns": ["MILESTONE: configuration accepted"]
  }
}
```

`command` is an argument array and is executed without a shell. `{value}` and
`{parameter}` are substituted per probe. The subprocess also receives
`CONFIGFUZZ_VALUE` and `CONFIGFUZZ_PARAMETER` environment variables.

When `values` is omitted, ConfigFuzz generates a bounded probe set from the
baseline, common numeric boundaries, powers of two, user-provided seed values,
and integer context values and divisors.

## Classification order

The classifier applies the following precedence:

1. infrastructure pattern → `INFRASTRUCTURE_FAILURE`;
2. resource pattern → `RESOURCE_FAILURE`;
3. explicit configuration rejection → `INVALID`;
4. explicit bug oracle → `POTENTIAL_BUG`;
5. timeout → `UNKNOWN`;
6. successful exit with the required milestone → `VALID`;
7. unexpected non-zero exit after the milestone → `UNEXPLAINED_FAILURE`;
8. otherwise → `UNKNOWN`.

Validity feedback uses `VALID` and explicit `INVALID` observations. Resource,
infrastructure, unexplained, unknown, and potential-bug outcomes remain in the
sample corpus but do not become invalid-domain evidence.

## Synthesis

The current template library includes:

- open and closed numeric bounds;
- finite scalar sets;
- constant divisibility, such as `x % 8 == 0`;
- contextual divisibility, such as `hidden_size % tp == 0`;
- contextual order, such as `batch_size <= device_count`.

Every candidate must accept all observed valid samples. Z3 then minimizes, in
order:

1. uncovered invalid samples;
2. total constraint complexity;
3. number of selected constraints;
4. deterministic candidate rank.

Contextual formulas are ranked above equivalent hard-coded constants because
they are more likely to transfer across models and environments.

## Commands

Run probes only:

```bash
python -m configfuzz probe manifest.json --output samples.json
```

Synthesize from an existing sample file:

```bash
python -m configfuzz synthesize samples.json --output spec.json
```

Run both stages:

```bash
python -m configfuzz infer manifest.json \
  --samples-output samples.json \
  --output spec.json
```

## Executing paired interventions

After `design-intervention` has produced satisfying, violating, and repaired
configurations, `run-intervention` applies each joint assignment to a baseline
configuration and executes it through a separate manifest:

```json
{
  "baseline_config": "../lmsv_validator_baseline.json",
  "command": [
    "python",
    "experiments/lmsv_validator_probe.py",
    "--config",
    "{config}",
    "--tracked-parameters",
    "{tracked_parameters}"
  ],
  "cwd": "../..",
  "timeout_seconds": 10,
  "classification": {
    "invalid_patterns": ["CONFIG_INVALID:"],
    "bug_patterns": ["BUG_ORACLE:"],
    "milestone_patterns": ["MILESTONE: lm-sv validator accepted tracked parameters unchanged"]
  },
  "provenance_patterns": ["PROVENANCE:.*config_validator_moe\\.py"]
}
```

Supported command substitutions are `{config}`, `{role}`, `{intervention_id}`,
`{edge_id}`, `{primary_parameter}`, `{primary_value}`, `{assignments}`, and
`{tracked_parameters}`. The runner writes each case to an isolated temporary
JSON file. Flat graph names are resolved to exact dotted paths or to a unique
leaf in the nested baseline; ambiguous fields are rejected rather than guessed.

```bash
python -m configfuzz run-intervention \
  intervention-plan.json intervention-manifest.json \
  --output intervention-samples.json
```

The output is directly accepted by `apply-feedback`. An invalid case receives
`provenance_matched=true` only when its output matches the configured rejection
signature or, when no explicit pattern is supplied, the target edge's recorded
source path. This enables paired confirmation without treating an unrelated
validation failure as evidence for the selected edge.

## Current limitations

- One target parameter is varied per manifest.
- Context values are fixed during a run.
- Candidate generation is template bounded.
- The current `infer` command performs one probe batch rather than a full CEGIS
  loop.
- A validator repair is only an invalid oracle when the adapter explicitly
  reports it as such.
- The generic intervention runner currently patches JSON configurations; shell
  launch arguments and framework-native configuration objects require adapters.
