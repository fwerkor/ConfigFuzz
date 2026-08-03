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

1. infrastructure pattern → `UNKNOWN`;
2. explicit configuration rejection → `INVALID`;
3. explicit bug oracle → `POTENTIAL_BUG`;
4. timeout → `UNKNOWN`;
5. successful exit with the required milestone → `VALID`;
6. unexpected non-zero exit after the milestone → `POTENTIAL_BUG`;
7. otherwise → `UNKNOWN`.

Only `VALID` and `INVALID` samples are used to learn hard constraints.
`UNKNOWN` and `POTENTIAL_BUG` remain in the sample corpus but are excluded from
negative learning so that infrastructure failures and defect-triggering inputs
do not become exclusion rules.

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

## Current limitations

- One target parameter is varied per manifest.
- Context values are fixed during a run.
- Candidate generation is template bounded.
- The current `infer` command performs one probe batch rather than a full CEGIS
  loop.
- A validator repair is only an invalid oracle when the adapter explicitly
  reports it as such.
