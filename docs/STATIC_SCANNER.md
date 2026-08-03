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
- direct configuration access through attributes, subscripts, and `dict.get`;
- exact local aliases and simple arithmetic derived from configuration values;
- conditions inherited from enclosing branches;
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
9. deduplicates equivalent normalized expressions.

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

On the included lm-sv regression fixture, the optimized strict scanner processes 65 parameter names over 90 Python files and currently emits 42 candidates for 17 parameters.

Compared with the original name-matching scanner:

| Indicator | Original | Optimized strict mode |
| --- | ---: | ---: |
| Total candidates | 214 | 42 |
| Parameters with candidates | 22 | 17 |
| Self-equality tautologies | 5 | 0 |
| Tensor-shape false positives | 7 | 0 |
| Unnormalized `not (...)` expressions | 93 | 0 |
| `other`-class expressions | 83 | 1 |

The current inventory build takes about 16 seconds in the repository environment with automatic four-worker scanning. These counts are regression indicators, not precision or recall measurements. Formal accuracy must be measured against the reviewed manual corpus and independently verified framework/runtime behavior.

## Remaining limitations

The scanner is not yet a complete interprocedural analyzer. Important remaining work includes:

- propagating symbolic values through calls across modules;
- resolving environment-derived constants into explicit scope symbols;
- extracting argparse, dataclass, schema, YAML, and shell constraints;
- ranking candidates against the reviewed corpus;
- distinguishing framework requirements from conservative repair policy using framework-side evidence;
- validating candidates dynamically before they guide mutation.
