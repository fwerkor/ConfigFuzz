# Constraint DSL

ConfigFuzz uses a bounded, evidence-backed representation rather than unrestricted natural-language rules. The DSL is intended to be expressive enough for the dominant constraints in the baseline while remaining synthesizable and testable.

## Record format

```json
{
  "parameter": "tensor_model_parallel_size",
  "constraints": [
    {
      "expression": "tensor_model_parallel_size >= 1",
      "kind": "range",
      "parameters": ["tensor_model_parallel_size"],
      "confidence": 1.0,
      "scope": {
        "model": "*",
        "backend": "*"
      },
      "evidence": [
        {
          "kind": "static",
          "source": "validator.py",
          "line": 120,
          "detail": "rejecting guard"
        }
      ]
    }
  ]
}
```

Inferred constraints use the compact model in `configfuzz/model.py`. Reviewed
manual rules use the richer corpus model in `configfuzz/corpus.py`, which also
records scope, enforcement behavior, semantic strength, status, source
locations, and repair metadata.

## Core grammar

```text
constraint := predicate
            | condition "=>" predicate

predicate  := value relation value
            | value "in" set
            | value "%" value "==" 0
            | "is_power_of_two(" value ")"
            | "supports(" capability ")"
            | "estimated_memory(" config ")" "<=" available_memory

relation   := "==" | "!=" | "<" | "<=" | ">" | ">="
value      := parameter | constant | environment | arithmetic_expression
```

## Constraint classes

### Type

```text
batch_size: integer
use_flash_attention: boolean
dtype: string
```

### Range

```text
batch_size >= 1
0.0 <= dropout <= 1.0
```

### Enumeration

```text
dtype in {"fp16", "bf16"}
recompute_method in {"block", "uniform"}
```

### Arithmetic and relational

```text
hidden_size % tensor_model_parallel_size == 0
num_attention_heads % (tensor_model_parallel_size * context_parallel_size) == 0
micro_batch_size <= global_batch_size
tensor_model_parallel_size * pipeline_model_parallel_size <= world_size
```

### Conditional

```text
position_embedding_type == "alibi" => context_parallel_size == 1
moe_router_topk == 1 => moe_router_pre_softmax == true
```

### Environment and resource

```text
tensor_model_parallel_size <= visible_device_count
backend == "ascend" => dtype in {"fp16", "bf16"}
estimated_memory(config) <= available_memory
```

## Semantics

Each constraint also needs:

- **scope**: model, backend, software version, hardware, and execution stage;
- **strength**: hard semantic, implementation, environment, resource, or empirical;
- **provenance**: static location, documentation, runtime samples, or manual annotation;
- **confidence**: calibrated probability or ranking score;
- **status**: candidate, validated, contradicted, or deprecated.

A constraint is never considered universal merely because it held in one environment.
