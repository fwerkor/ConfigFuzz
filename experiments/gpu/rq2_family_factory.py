from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping


def load_profile(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("workload profile must be a JSON object")
    return payload


def build_model(profile: Mapping[str, Any], *, empty_weights: bool = False):
    """Build the reduced architecture for a formal RQ2 workload.

    No pretrained weights are downloaded. ``empty_weights`` is used by the
    accelerator-free preflight to validate configuration/model construction
    without allocating the full parameter tensors.
    """

    family = str(profile["family"])
    context = _empty_weight_context() if empty_weights else nullcontext()
    with context:
        if family == "qwen2_dense_gqa":
            return _build_qwen2(profile)
        if family == "llama2_dense_rope":
            return _build_llama(profile)
        if family == "chatglm3_gqa_long_sequence":
            return _build_glm(profile)
        if family == "mixtral_moe":
            return _build_mixtral(profile)
        if family == "deepseekv3_mla_moe":
            return _build_deepseek_v3(profile)
        if family == "internvl3_vision_text":
            return _build_internvl(profile)
        if family == "cogvideox_video_text":
            return _build_cogvideox(profile)
    raise ValueError(f"unsupported RQ2 workload family: {family}")


def model_parameter_count(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _text_common(profile: Mapping[str, Any]) -> dict[str, int | float | bool]:
    model = _mapping(profile, "model")
    return {
        "vocab_size": int(model["vocab_size"]),
        "hidden_size": int(model["hidden_size"]),
        "intermediate_size": int(model["ffn_hidden_size"]),
        "num_hidden_layers": int(model["num_layers"]),
        "num_attention_heads": int(model["num_attention_heads"]),
        "num_key_value_heads": int(model["num_query_groups"]),
        "max_position_embeddings": int(model["max_position_embeddings"]),
        "attention_dropout": float(model.get("attention_dropout", 0.0)),
        "use_cache": False,
    }


def _build_qwen2(profile: Mapping[str, Any]):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config(**_text_common(profile))
    return Qwen2ForCausalLM(config)


def _build_llama(profile: Mapping[str, Any]):
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(**_text_common(profile))
    return LlamaForCausalLM(config)


def _build_glm(profile: Mapping[str, Any]):
    from transformers import GlmConfig, GlmForCausalLM

    common = _text_common(profile)
    common["head_dim"] = int(common["hidden_size"]) // int(common["num_attention_heads"])
    common["pad_token_id"] = 0
    config = GlmConfig(**common)
    return GlmForCausalLM(config)


def _build_mixtral(profile: Mapping[str, Any]):
    from transformers import MixtralConfig, MixtralForCausalLM

    moe = _mapping(profile, "moe")
    config = MixtralConfig(
        **_text_common(profile),
        num_local_experts=int(moe["num_experts"]),
        num_experts_per_tok=int(moe["moe_router_topk"]),
        output_router_logits=False,
    )
    return MixtralForCausalLM(config)


def _build_deepseek_v3(profile: Mapping[str, Any]):
    from transformers import DeepseekV3Config, DeepseekV3ForCausalLM

    model = _mapping(profile, "model")
    moe = _mapping(profile, "moe")
    mla = _mapping(profile, "mla")
    config = DeepseekV3Config(
        vocab_size=int(model["vocab_size"]),
        hidden_size=int(model["hidden_size"]),
        intermediate_size=int(model["ffn_hidden_size"]),
        moe_intermediate_size=int(moe["moe_intermediate_size"]),
        num_hidden_layers=int(model["num_layers"]),
        num_attention_heads=int(model["num_attention_heads"]),
        num_key_value_heads=int(model["num_query_groups"]),
        n_shared_experts=int(moe["n_shared_experts"]),
        n_routed_experts=int(moe["num_experts"]),
        n_group=int(moe["moe_router_num_groups"]),
        topk_group=int(moe["topk_group"]),
        num_experts_per_tok=int(moe["moe_router_topk"]),
        first_k_dense_replace=1,
        kv_lora_rank=int(mla["kv_lora_rank"]),
        q_lora_rank=int(mla["q_lora_rank"]),
        qk_rope_head_dim=int(mla["qk_rope_head_dim"]),
        qk_nope_head_dim=int(mla["qk_nope_head_dim"]),
        v_head_dim=int(mla["v_head_dim"]),
        max_position_embeddings=int(model["max_position_embeddings"]),
        attention_dropout=float(model.get("attention_dropout", 0.0)),
        use_cache=False,
    )
    return DeepseekV3ForCausalLM(config)


def _build_internvl(profile: Mapping[str, Any]):
    from transformers import InternVLConfig, InternVLForConditionalGeneration

    model = _mapping(profile, "model")
    mm = _mapping(profile, "multimodal")
    text_config = {
        "model_type": "qwen2",
        "vocab_size": int(model["vocab_size"]),
        "hidden_size": int(model["hidden_size"]),
        "intermediate_size": int(model["ffn_hidden_size"]),
        "num_hidden_layers": int(model["num_layers"]),
        "num_attention_heads": int(model["num_attention_heads"]),
        "num_key_value_heads": int(model["num_query_groups"]),
        "max_position_embeddings": int(model["max_position_embeddings"]),
        "use_cache": False,
    }
    vision_config = {
        "hidden_size": int(mm["vision_hidden_size"]),
        "intermediate_size": int(mm["vision_ffn_hidden_size"]),
        "num_hidden_layers": int(mm["vision_num_layers"]),
        "num_attention_heads": int(mm["vision_num_attention_heads"]),
        "image_size": int(mm["image_size"]),
        "patch_size": int(mm["patch_size"]),
        "num_channels": int(mm["num_channels"]),
    }
    config = InternVLConfig(
        vision_config=vision_config,
        text_config=text_config,
        image_token_id=min(int(model["vocab_size"]) - 1, 1),
        image_seq_length=int(mm["image_seq_length"]),
        downsample_ratio=float(mm["downsample_ratio"]),
        tie_word_embeddings=False,
    )
    return InternVLForConditionalGeneration(config)


def _build_cogvideox(profile: Mapping[str, Any]):
    try:
        from diffusers import CogVideoXTransformer3DModel
    except ImportError as exc:  # pragma: no cover - environment preflight reports this
        raise RuntimeError("CogVideoX RQ2 workload requires diffusers") from exc

    model = _mapping(profile, "model")
    video = _mapping(profile, "video")
    return CogVideoXTransformer3DModel(
        num_attention_heads=int(model["num_attention_heads"]),
        attention_head_dim=int(video["attention_head_dim"]),
        in_channels=int(video["in_channels"]),
        out_channels=int(video["out_channels"]),
        time_embed_dim=int(model["hidden_size"]),
        text_embed_dim=int(video["text_embed_dim"]),
        num_layers=int(model["num_layers"]),
        sample_width=int(video["sample_width"]),
        sample_height=int(video["sample_height"]),
        sample_frames=int(video["frames"]),
        patch_size=int(video["patch_size"]),
        temporal_compression_ratio=int(video["temporal_compression_ratio"]),
        max_text_seq_length=int(video["max_text_seq_length"]),
    )


def _empty_weight_context():
    try:
        from accelerate import init_empty_weights
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("accelerate is required for empty-weight model preflight") from exc
    return init_empty_weights()


def _mapping(profile: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = profile.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"workload profile is missing object field {key!r}")
    return value
