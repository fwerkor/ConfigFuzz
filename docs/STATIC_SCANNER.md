# Static Constraint Scanner

The static scanner extracts candidate configuration constraints from Python framework code. The formal analysis target is the framework under test, such as Megatron-LM, MindSpeed-LLM, MindFormers, or MindSpeed-MM. The lm-sv snapshot in this repository is used only as a regression fixture and evaluation corpus.

## Modes

`configfuzz scan` uses strict mode by default. Strict mode accepts only expressions that fit the bounded constraint DSL and rejects unsupported object, path, tensor-shape, and arbitrary-call predicates.

```bash
python -m configfuzz scan \
  --parameter hidden_size \
  --parameter tensor_model_parallel_size \
  --jobs 0 \
  /path/to/framework
```

`--jobs 0` selects up to four source-file workers automatically. Pass an explicit positive number to control parallelism.

The JSON result contains both the parameter-indexed candidate lists and a
deduplicated `dependency_graph`. See `docs/DEPENDENCY_GRAPH.md` for graph
semantics and mutation planning.

Broad mode retains unsupported expressions for recall-oriented exploration:

```bash
python -m configfuzz scan --broad --parameter hidden_size /path/to/framework
```

Broad-mode output must not be treated as a validated constraint set.

## Extraction rules

The scanner currently recognizes:

- assertions that reference a target parameter;
- guards that directly raise an exception;
- guards that directly call `_apply_fix` or `apply_fix`;
- constraints implemented in helper functions and instantiated at local or
  cross-file call sites;
- direct configuration access through attributes, subscripts, and `dict.get`;
- exact local aliases and simple arithmetic derived from configuration values;
- conditions inherited from enclosing branches;
- argparse types, choices, Boolean actions, and required options;
- dataclass annotations, `Literal` choices, and `field`/`Field` metadata;
- JSON-schema-style YAML plus the existing numeric/enum pool layout;
- integer range, enumeration, divisibility, relational, type, and conditional predicates.

The scanner deliberately ignores:

- arbitrary path and filesystem checks;
- tensor-shape checks unrelated to configuration validity;
- branch conditions where both outcomes select different repairs;
- `new_value != old_value` checks used only to decide whether a repair should be written;
- tautologies and unsupported free-form expressions in strict mode.

## Normalization

Before emitting a candidate, the scanner:

1. resolves unambiguous configuration aliases;
2. expands local symbolic helper variables such as alignment divisors;
3. merges lexical branch and exception-flow bindings conservatively;
4. converts rejecting guards into valid-domain predicates;
5. rewrites suitable guarded failures as implications;
6. orients Boolean requirements toward the enabled feature;
7. removes redundant `max(1, positive_product)` wrappers;
8. canonicalizes floor-division lower bounds;
9. instantiates bounded function summaries with actual call arguments;
10. canonicalizes fields inside Config, Arguments, and Validator classes;
11. deduplicates equivalent normalized expressions.

For example:

```python
if num_experts > 0 and num_experts % expert_parallel_size != 0:
    raise ValueError(...)
```

becomes:

```text
num_experts > 0 => num_experts % expert_model_parallel_size == 0
```

A validator pattern such as:

```python
if topk == 1 and not pre_softmax:
    apply_fix(...)
```

becomes:

```text
moe_router_topk == 1 => moe_router_pre_softmax
```

## Current regression inventory

On the included lm-sv regression fixture, the optimized strict scanner processes 65 parameter names over 90 Python files and currently emits 61 parameter-indexed candidates for 17 parameters. These correspond to 40 unique normalized expressions because a cross-parameter rule is indexed under every participating parameter.

The corresponding dependency graph contains 35 nodes, 40 unique hyperedges,
and four connected components.

Compared with the original name-matching scanner:

| Indicator | Original | Optimized strict mode |
| --- | ---: | ---: |
| Total parameter-indexed candidates | 214 | 61 |
| Parameters with candidates | 22 | 17 |
| Self-equality tautologies | 5 | 0 |
| Tensor-shape false positives | 7 | 0 |
| Unnormalized `not (...)` expressions | 93 | 0 |
| `other`-class expressions | 83 | 2 |

The current inventory build takes about 11--12 seconds in the repository environment with automatic four-worker scanning. These counts are regression indicators, not precision or recall measurements. Formal accuracy must be measured against the reviewed manual corpus and independently verified framework/runtime behavior.

## Framework-side validation

The scanner has also been run against a clean Megatron-LM checkout rather
than the lm-sv baseline:

```text
repository: https://github.com/NVIDIA/Megatron-LM.git
commit:     42460a7af821366e7115a162cb4410106bea93f0
source:     megatron/
mode:       strict
```

For the first six parameters, the current scanner emits 67 candidates:

| Parameter | Candidates |
| --- | ---: |
| `tensor_model_parallel_size` | 16 |
| `pipeline_model_parallel_size` | 24 |
| `hidden_size` | 6 |
| `num_attention_heads` | 6 |
| `micro_batch_size` | 7 |
| `sequence_parallel` | 8 |

The scan includes dataclass types, positivity constraints, tensor/pipeline
divisibility, feature dependencies, and constraints reached through helper
functions. On the repository machine it takes about 20--22 seconds with four
workers. The versioned result is stored in
`artifacts/frameworks/megatron_lm_42460a7.json`.

After removing copies indexed under multiple parameters, this artifact contains
60 dependency nodes, 62 unique hyperedges, and two connected components.

Reproduce it with:

```bash
python scripts/run_framework_static_scan.py /path/to/Megatron-LM \
  --source-subdir megatron \
  --name Megatron-LM \
  --parameter tensor_model_parallel_size \
  --parameter pipeline_model_parallel_size \
  --parameter hidden_size \
  --parameter num_attention_heads \
  --parameter micro_batch_size \
  --parameter sequence_parallel \
  --jobs 4 \
  --output artifacts/frameworks/megatron_lm_42460a7.json
```

These 67 expressions are candidates, not a claimed gold set. Some apply only
to a specific module or execution path. Scope classification and dynamic
validation must run before they are used to constrain mutation.

## Remaining limitations

The scanner is not yet a complete program analyzer. Important remaining work includes:

- resolving return-value, object-field, and higher-order data flow across calls;
- resolving environment-derived constants into explicit scope symbols;
- extracting shell launch constraints and non-Python schema formats;
- ranking candidates against the reviewed corpus;
- attaching model/module/version scope to each inferred candidate;
- distinguishing framework requirements from conservative repair policy using framework-side evidence;
- validating candidates dynamically before they guide mutation.
