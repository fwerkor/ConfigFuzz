from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FrameworkProfile:
    key: str
    display_name: str
    source_subdirs: tuple[str, ...]
    parameters: tuple[str, ...]
    backend: str
    accelerator: str
    role: str


_PROFILES: dict[str, FrameworkProfile] = {
    "pytorch-cuda": FrameworkProfile(
        key="pytorch-cuda",
        display_name="PyTorch Native/CUDA",
        source_subdirs=("torch",),
        parameters=(
            "world_size",
            "group_size",
            "backend",
            "device_id",
            "sharding_strategy",
            "mixed_precision",
            "cpu_offload",
            "backward_prefetch",
            "forward_prefetch",
            "limit_all_gathers",
            "use_orig_params",
            "device_mesh",
        ),
        backend="CUDA/NCCL",
        accelerator="NVIDIA GPU",
        role="general-purpose distributed framework",
    ),
    "deepspeed": FrameworkProfile(
        key="deepspeed",
        display_name="DeepSpeed",
        source_subdirs=("deepspeed",),
        parameters=(
            "train_batch_size",
            "train_micro_batch_size_per_gpu",
            "gradient_accumulation_steps",
            "stage",
            "offload_optimizer",
            "offload_param",
            "overlap_comm",
            "reduce_scatter",
            "allgather_bucket_size",
            "reduce_bucket_size",
            "sub_group_size",
            "sequence_parallel_size",
            "tensor_parallel",
            "pipeline_parallel_size",
            "fp16",
            "bf16",
        ),
        backend="CUDA/NCCL",
        accelerator="NVIDIA GPU",
        role="distributed training and memory optimization stack",
    ),
    "megatron-core": FrameworkProfile(
        key="megatron-core",
        display_name="Megatron-Core",
        source_subdirs=("megatron",),
        parameters=(
            "tensor_model_parallel_size",
            "pipeline_model_parallel_size",
            "context_parallel_size",
            "expert_model_parallel_size",
            "sequence_parallel",
            "virtual_pipeline_model_parallel_size",
            "num_layers",
            "hidden_size",
            "ffn_hidden_size",
            "num_attention_heads",
            "num_query_groups",
            "num_moe_experts",
            "micro_batch_size",
            "global_batch_size",
            "fp16",
            "bf16",
            "recompute_granularity",
            "use_distributed_optimizer",
        ),
        backend="CUDA/NCCL",
        accelerator="NVIDIA GPU",
        role="large-model parallel training stack",
    ),
    "transformers-accelerate": FrameworkProfile(
        key="transformers-accelerate",
        display_name="Transformers/Accelerate",
        source_subdirs=("src/transformers", "src/accelerate"),
        parameters=(
            "per_device_train_batch_size",
            "gradient_accumulation_steps",
            "gradient_checkpointing",
            "fp16",
            "bf16",
            "tf32",
            "torch_compile",
            "fsdp",
            "deepspeed",
            "mixed_precision",
            "dispatch_batches",
            "split_batches",
            "even_batches",
            "use_seedable_sampler",
        ),
        backend="PyTorch/CUDA",
        accelerator="NVIDIA GPU",
        role="high-level training and launcher integration stack",
    ),
}

_ALIASES = {
    "pytorch": "pytorch-cuda",
    "cuda": "pytorch-cuda",
    "deepspeed": "deepspeed",
    "megatron": "megatron-core",
    "megatron-core": "megatron-core",
    "transformers": "transformers-accelerate",
    "accelerate": "transformers-accelerate",
    "transformers-accelerate": "transformers-accelerate",
}


def list_framework_profiles() -> tuple[FrameworkProfile, ...]:
    return tuple(_PROFILES[key] for key in sorted(_PROFILES))


def get_framework_profile(name: str) -> FrameworkProfile:
    normalized = name.strip().lower().replace("_", "-")
    key = _ALIASES.get(normalized, normalized)
    try:
        return _PROFILES[key]
    except KeyError as exc:
        available = ", ".join(sorted(_PROFILES))
        raise ValueError(f"unknown framework profile {name!r}; choose one of: {available}") from exc
