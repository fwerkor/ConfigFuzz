from __future__ import annotations

from configfuzz.workload_candidates import (
    _build_effective_config,
    _extract_command_overrides,
)


def test_command_template_overrides_are_resolved_and_aliased() -> None:
    command = r"""
TP=2
PP=4
SEQ_LEN=8192
GPT_ARGS="
  --tensor-model-parallel-size ${TP} \
  --pipeline-model-parallel-size ${PP} \
  --seq-length ${SEQ_LEN} \
  --disable-bias-linear \
  --swiglu \
  --use-flash-attn \
  --no-masked-softmax-fusion
"
"""

    overrides = _extract_command_overrides(command)

    assert overrides["tensor_model_parallel_size"] == 2
    assert overrides["pipeline_model_parallel_size"] == 4
    assert overrides["seq_length"] == 8192
    assert overrides["seq_len"] == 8192
    assert overrides["disable_bias_linear"] is True
    assert overrides["add_bias_linear"] is False
    assert overrides["swiglu"] is True
    assert overrides["gated_linear_unit"] is True
    assert overrides["use_flash_attention"] is True
    assert overrides["masked_softmax_fusion"] is False


def test_effective_config_prefers_command_and_derives_parallel_values() -> None:
    model = {
        "TransformerConfig": {
            "hidden_size": 1024,
            "tensor_model_parallel_size": 1,
            "pipeline_model_parallel_size": 1,
            "num_attention_heads": 16,
            "num_query_groups": 4,
        },
        "extra_config": {"seq_length": 4096},
    }
    overrides = {
        "hidden_size": 2048,
        "tensor_model_parallel_size": 2,
        "pipeline_model_parallel_size": 2,
        "context_parallel_size": 2,
        "seq_length": 8192,
    }

    effective, sources = _build_effective_config(model, overrides, world_size=16)

    assert effective["hidden_size"] == 2048
    assert effective["seq_length"] == 8192
    assert effective["data_parallel_size"] == 2
    assert effective["group_query_attention"] is True
    assert sources["hidden_size"] == "command_template"
    assert sources["data_parallel_size"] == "derived:world_size/(tp*pp*cp)"
