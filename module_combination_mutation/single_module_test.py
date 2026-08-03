"""
基于单模块变异生成的配置文件跑测：读取目录下或单个 .json 配置，
逐个用 SingleTextDecoderTemplate / SingleImageEncoderTemplate 实例化、前向 / Loss / 反向。
支持多卡、权重保存、基于权重目录或单一权重启动。
"""
import argparse
import csv
import glob
import json
import os
import time
import torch.nn.functional as F

os.environ.setdefault("ASCEND_PRECISION_MODE", "must_keep_origin_dtype")

from typing import Dict, Any

import torch

import mindspeed.megatron_adaptor
from megatron.core.transformer import TransformerConfig
from megatron.training.initialize import initialize_megatron
from megatron.training import get_args

from mindspeed_mm.configs.config import mm_extra_args_provider

from modules.pool import TEXT_DECODER_DICT_POOL, IMAGE_ENCODER_DICT_POOL
from templates.template import TEMPLATE_REGISTRY
from templates.single_text_decoder_template import SingleTextDecoderTemplate
from templates.single_image_encoder_template import SingleImageEncoderTemplate
from utils import (
    set_results_dir,
    get_results_dir,
    log_section,
    log_step,
    log_bullet,
    log_newline,
    get_device,
)
def extra_args_provider(parser):
    parser.add_argument(
        "--config-dir",
        required=False,
        type=str,
        default="",
        help="单模块配置文件目录（内含 1-1-*.json 等），将逐个跑测",
    )
    parser.add_argument(
        "--config",
        required=False,
        type=str,
        default="",
        help="指定只跑某一个配置（.json 文件路径）。需配合 --config-dir 或单独使用",
    )
    parser.add_argument(
        "--single-module-test-optimizer",
        dest="single_module_test_optimizer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否启用 optimizer（step/zero_grad）；禁用则仅 forward+loss+backward",
    )
    parser.add_argument(
        "--iterations",
        dest="single_module_test_iterations",
        type=int,
        default=2,
        help="每个 config 执行 forward+loss+backward 的轮次（默认 2）",
    )
    parser.add_argument(
        "--all-modules",
        required=False,
        type=str,
        default="./modules.json",
        help="all modules json file（用于注册 DictPool，与 single_module_mutate 一致）",
    )
    parser.add_argument(
        "--results-dir",
        required=False,
        type=str,
        default="./results/_single_module",
        help="结果目录，用于 profile、日志等",
    )
    parser.add_argument(
        "--save-ckpt",
        dest="save_ckpt",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="是否保存每个 config 的权重（rank0 保存）",
    )
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default="./ckpts",
        help="权重保存根目录",
    )
    parser.add_argument(
        "--load-ckpt",
        dest="load_ckpt",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="是否在跑测前加载权重",
    )
    parser.add_argument(
        "--load-ckpt-dir",
        type=str,
        default="",
        help="待加载权重目录（包含 *.pt）",
    )
    parser.add_argument(
        "--ckpt",
        required=False,
        type=str,
        default="",
        help="指定要加载的单个权重文件（*.pt）。需配合 --config 使用；优先于 --load-ckpt-dir",
    )
    parser = mm_extra_args_provider(parser)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dummy_dir = os.path.join(script_dir, "dummy_mm")
    parser.set_defaults(
        mm_model=os.path.join(dummy_dir, "model.json"),
        mm_data=os.path.join(dummy_dir, "data.json"),
        mm_tool=os.path.join(dummy_dir, "tool.json"),
    )
    return parser


def _apply_config_preprocess_to_cfg(cfg: Dict[str, Any]) -> None:
    """对单个 config 应用预处理（与 single_module_mutate/mm_test 一致）"""
    # if cfg.get("position_embedding_type", "rope") == "mrope":
    #     cfg["use_fused_rotary_pos_emb"] = False
    if isinstance(cfg.get("params_dtype"), str):
        mapping = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
            "torch.bfloat16": torch.bfloat16,
            "torch.float16": torch.float16,
            "torch.float32": torch.float32,
        }
        cfg["params_dtype"] = mapping.get(cfg["params_dtype"], torch.bfloat16)
    if "shape_order" not in cfg:
        cfg["shape_order"] = "SBH"
    if cfg.get("moe_grouped_gemm", False):
        cfg["moe_grouped_gemm"] = False

    # target_hidden = 32
    # target_heads = 16
    # if cfg.get("hidden_size", target_hidden) > target_hidden:
    #     cfg["hidden_size"] = target_hidden
    # if cfg.get("num_attention_heads", target_heads) > target_heads:
    #     cfg["num_attention_heads"] = target_heads
    # num_heads = cfg.get("num_attention_heads", target_heads)
    # if cfg.get("num_query_groups", num_heads) > num_heads:
    #     cfg["num_query_groups"] = num_heads
    # if cfg.get("ffn_hidden_size", 64) > 64:
    #     cfg["ffn_hidden_size"] = 64
    # if cfg.get("num_moe_experts", 4) > 4:
    #     cfg["num_moe_experts"] = 4
    # num_experts = cfg.get("num_moe_experts", 4)
    # num_groups = max(cfg.get("n_group", 1), 1)
    # if num_groups > num_experts:
    #     num_groups = num_experts
    #     cfg["n_group"] = num_groups
    # group_topk = max(cfg.get("topk_group", 1), 1)
    # max_topk_per_group = num_experts // num_groups
    # max_topk = max_topk_per_group * group_topk
    # if cfg.get("moe_router_topk", 0) > max_topk:
    #     cfg["moe_router_topk"] = max_topk
    # if cfg.get("moe_ffn_hidden_size", 32) > 32:
    #     cfg["moe_ffn_hidden_size"] = 32

    for key in ("vision_encoder", "vision_projector",):
        if key in cfg and isinstance(cfg[key], dict):
            _apply_config_preprocess_to_cfg(cfg[key])
    # if "vision_encoder" in cfg and "vision_projector" in cfg:
    #     ve = cfg["vision_encoder"]
    #     vp = cfg["vision_projector"]
    #     if isinstance(ve, dict) and isinstance(vp, dict) and "hidden_size" in ve:
    #         vp["input_size"] = ve["hidden_size"]


def preprocess_single_module_config(record: Dict[str, Any]) -> None:
    """对单模块配置记录做预处理，原地修改 record['config']"""
    if "config" in record and isinstance(record["config"], dict):
        _apply_config_preprocess_to_cfg(record["config"])


def load_modules_json(json_path: str) -> Dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def config_preprocess(config: Dict[str, Any]) -> Dict[str, Any]:
    for _, modules in config.items():
        for _, cfg in modules.items():
            if isinstance(cfg, dict):
                _apply_config_preprocess_to_cfg(cfg)
    return config


def register_all_modules(all_modules: Dict[str, Any], args) -> None:
    text_decoders = all_modules.get("text_decoders", {})
    image_encoders = all_modules.get("image_encoders", {})
    for name, config in text_decoders.items():
        if name in ("internvl2_2b", "qwen2vl_2b", "videoalign"):
            continue
        TEXT_DECODER_DICT_POOL.register_one(name, config)
    for name, config in image_encoders.items():
        IMAGE_ENCODER_DICT_POOL.register_one(name, config)


def register_single_templates():
    TEMPLATE_REGISTRY.register(SingleTextDecoderTemplate())
    TEMPLATE_REGISTRY.register(SingleImageEncoderTemplate())


def _module_type_to_template_name(module_type: str) -> str:
    if module_type == "text_decoder":
        return "single_text_decoder"
    if module_type == "image_encoder":
        return "single_image_encoder"
    raise ValueError(f"Unknown module_type for single_module_test: {module_type}")


def _get_mem_backend():
    if hasattr(torch, "npu") and getattr(torch.npu, "is_available", lambda: False)():
        return "npu", torch.npu
    if getattr(torch.cuda, "is_available", lambda: False)():
        return "cuda", torch.cuda
    return None, None


def _safe_mb(value: float):
    return None if value is None else value / (1024.0 ** 2)


def _maybe_call(obj, name: str, *args, **kwargs):
    fn = getattr(obj, name, None)
    if fn is None:
        return None
    try:
        return fn(*args, **kwargs)
    except Exception:
        return None


def _sync_backend(mem_backend):
    if mem_backend is None:
        return
    _maybe_call(mem_backend, "synchronize")


def _reset_peak_stats(mem_backend):
    if mem_backend is None:
        return
    _maybe_call(mem_backend, "reset_peak_memory_stats")
    _maybe_call(mem_backend, "empty_cache")


def _get_mem_stats_mb(mem_backend):
    if mem_backend is None:
        return {}
    allocated = _maybe_call(mem_backend, "memory_allocated")
    reserved = _maybe_call(mem_backend, "memory_reserved")
    max_allocated = _maybe_call(mem_backend, "max_memory_allocated")
    max_reserved = _maybe_call(mem_backend, "max_memory_reserved")
    return {
        "allocated_mb": _safe_mb(allocated),
        "reserved_mb": _safe_mb(reserved),
        "max_allocated_mb": _safe_mb(max_allocated),
        "max_reserved_mb": _safe_mb(max_reserved),
    }


def _dist_max(value: float, enabled: bool, device: torch.device):
    if value is None or not enabled:
        return value
    t = torch.tensor([value], device=device, dtype=torch.float32)
    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX)
    return float(t.item())

def _dist_pick_rank(value: float, enabled: bool, device: torch.device, src_rank: int):
    """Pick scalar value from a specific distributed rank via broadcast."""
    if value is None or not enabled:
        return value
    t = torch.tensor([value], device=device, dtype=torch.float32)
    torch.distributed.broadcast(t, src=src_rank)
    return float(t.item())


def _append_csv_row(csv_path: str, fieldnames, row: Dict[str, Any], rank: int):
    if rank != 0:
        return
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    write_header = (not os.path.exists(csv_path)) or (os.path.getsize(csv_path) == 0)
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


def _compute_pretrain_style_loss(output: torch.Tensor) -> torch.Tensor:
    """Mimic pretrain_gpt loss style: fp32 token loss with a full loss mask."""
    logits = output.float()
    if logits.dim() < 2 or logits.shape[-1] <= 1:
        # Fallback for non-logit outputs.
        return logits.pow(2).mean()

    vocab_size = int(logits.shape[-1])
    token_count = int(logits.numel() // vocab_size)
    labels = torch.arange(token_count, device=logits.device, dtype=torch.long) % vocab_size
    token_losses = F.cross_entropy(logits.reshape(-1, vocab_size), labels, reduction="none")
    loss_mask = torch.ones_like(token_losses, dtype=torch.float32)
    total_tokens = torch.clamp(loss_mask.sum(), min=1.0)
    return torch.sum(token_losses * loss_mask) / total_tokens


def _summarize_output(output: torch.Tensor, max_items: int = 8) -> str:
    """Return a compact output snapshot for CSV logging."""
    try:
        detached = output.detach()
        flat = detached.reshape(-1)
        take_n = min(int(flat.numel()), max_items)
        sample = flat[:take_n].float().cpu().tolist()
        sample_text = ", ".join(f"{v:.6g}" for v in sample)
        return (
            f"shape={tuple(detached.shape)}; "
            f"dtype={detached.dtype}; "
            f"sample=[{sample_text}]"
        )
    except Exception as exc:
        return f"shape={tuple(output.shape)}; summary_error={exc}"


def _state_dict_to_cpu(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in state_dict.items():
        if torch.is_tensor(v):
            out[k] = v.detach().cpu()
        else:
            out[k] = v
    return out


def _save_mm_instance_ckpt(
    *,
    mm_instance,
    ckpt_root: str,
    run_tag: str,
    config_idx: int,
    config_file: str,
    template_name: str,
    rank: int,
):
    if rank != 0:
        return None
    os.makedirs(ckpt_root, exist_ok=True)
    stem = os.path.splitext(os.path.basename(config_file))[0]
    ckpt_path = os.path.join(ckpt_root, f"{stem}.pt")
    payload = {
        "format": "single_module_test_state_dict_v1",
        "config_idx": config_idx,
        "config_file": os.path.basename(config_file),
        "template": template_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "state_dict": _state_dict_to_cpu(mm_instance.state_dict()),
    }
    torch.save(payload, ckpt_path)
    return ckpt_path


def _find_ckpt_for_config(load_ckpt_dir: str, config_idx: int, config_file: str) -> str:
    stem = os.path.splitext(os.path.basename(config_file))[0]
    patterns = [
        os.path.join(load_ckpt_dir, f"*_{config_idx:03d}_{stem}.pt"),
        os.path.join(load_ckpt_dir, f"*_{stem}.pt"),
    ]
    candidates = []
    for pat in patterns:
        candidates.extend(glob.glob(pat))
    if not candidates:
        return ""
    candidates = sorted(set(candidates), key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def _load_mm_instance_ckpt(*, mm_instance, ckpt_path: str):
    payload = torch.load(ckpt_path, map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
    elif isinstance(payload, dict):
        state_dict = payload
    else:
        raise ValueError(f"Unsupported checkpoint payload type: {type(payload)}")
    missing, unexpected = mm_instance.load_state_dict(state_dict, strict=False)
    return missing, unexpected


def main():
    initialize_megatron(extra_args_provider=extra_args_provider, args_defaults={})
    args = get_args()
    dist_enabled = torch.distributed.is_initialized()
    rank = torch.distributed.get_rank() if dist_enabled else 0
    world_size = torch.distributed.get_world_size() if dist_enabled else 1
    log_step("Info", f"Distributed: {dist_enabled}", rank=rank)
    log_step("Info", f"Rank: {rank}", rank=rank)
    log_step("Info", f"World size: {world_size}", rank=rank)
    optimizer_enabled = getattr(args, "single_module_test_optimizer", True)
    log_step("Info", f"single_module_test optimizer enabled: {optimizer_enabled}", rank=rank)
    iterations = int(getattr(args, "single_module_test_iterations", 2))
    if iterations < 1:
        raise ValueError(f"--iterations must be >= 1, got {iterations}")

    set_results_dir(args.results_dir)
    save_ckpt_enabled = getattr(args, "save_ckpt", False)
    ckpt_root = os.path.abspath(os.path.expanduser(getattr(args, "ckpt_dir", "./ckpts")))
    ckpt_file_arg = getattr(args, "ckpt", "") or ""
    ckpt_file_path = ""
    if ckpt_file_arg:
        cand = os.path.expanduser(ckpt_file_arg)
        if (not os.path.isabs(cand)) and (os.path.sep not in cand):
            raise ValueError(f"--ckpt must be a file path (not a filename): {ckpt_file_arg}")
        cand = os.path.abspath(cand)
        if (not os.path.isfile(cand)) or (not cand.endswith(".pt")):
            raise FileNotFoundError(f"--ckpt must be an existing .pt file path: {cand}")
        ckpt_file_path = cand

    load_ckpt_enabled = bool(getattr(args, "load_ckpt", False)) and bool(getattr(args, "load_ckpt_dir", ""))
    load_ckpt_dir = os.path.abspath(os.path.expanduser(getattr(args, "load_ckpt_dir", ""))) if getattr(args, "load_ckpt_dir", "") else ""
    run_tag = os.path.basename(os.path.abspath(get_results_dir()))
    if save_ckpt_enabled:
        log_step("CKPT", f"Save enabled: {save_ckpt_enabled}", rank=rank)
        log_step("CKPT", f"CKPT root: {ckpt_root}", rank=rank)
    if ckpt_file_path:
        log_step("CKPT", f"Load single ckpt: {ckpt_file_path}", rank=rank)
    if load_ckpt_enabled and (not ckpt_file_path):
        log_step("CKPT", f"Load from dir: {load_ckpt_dir}", rank=rank)
    if hasattr(args, "profile_save_path"):
        args.profile_save_path = os.path.join(get_results_dir(), "profile")
    log_section("单模块配置跑测（前向 / Loss / 反向）", rank=rank)
    log_step("Init", f"Results directory: {get_results_dir()}", rank=rank)
    log_step("Config", f"Config directory: {getattr(args, 'config_dir', '')}", rank=rank)
    log_step("Config", f"Config file: {getattr(args, 'config', '')}", rank=rank)

    if not hasattr(args, "rope_scaling_type"):
        args.rope_scaling_type = None
    if not hasattr(args, "rope_scaling_factor"):
        args.rope_scaling_factor = 1.0
    if not hasattr(args, "rope_scaling_mscale_all_dim"):
        args.rope_scaling_mscale_all_dim = 0
    if args.qk_nope_head_dim is None and args.qk_rope_head_dim is None:
        args.qk_nope_head_dim = 0
        args.qk_rope_head_dim = args.hidden_size // args.num_attention_heads

    all_modules = load_modules_json(args.all_modules)
    all_modules = config_preprocess(all_modules)
    register_all_modules(all_modules, args)
    register_single_templates()

    selected = getattr(args, "config", "") or ""
    config_dir_arg = getattr(args, "config_dir", "") or ""

    if config_dir_arg:
        config_dir_abs = os.path.abspath(os.path.expanduser(config_dir_arg))
        if not os.path.isdir(config_dir_abs):
            raise FileNotFoundError(f"--config-dir must be an existing directory: {config_dir_arg}")

    config_path_abs = ""
    if selected:
        cand = os.path.expanduser(selected)
        if (not os.path.isabs(cand)) and (os.path.sep not in cand):
            raise ValueError(f"--config must be a file path (not a filename): {selected}")
        cand = os.path.abspath(cand)
        if (not os.path.isfile(cand)) or (not cand.endswith(".json")):
            raise FileNotFoundError(f"--config must be an existing .json file: {cand}")
        config_path_abs = cand
    elif ckpt_file_path:
        raise ValueError("--ckpt requires --config (single-config run)")

    if (not config_dir_arg) and (not config_path_abs):
        raise ValueError("Must provide either --config-dir or --config")

    if config_path_abs:
        config_dir = os.path.dirname(config_path_abs)
    else:
        config_dir = os.path.abspath(os.path.expanduser(config_dir_arg))
    all_json_files = sorted(
        f for f in os.listdir(config_dir)
        if f.endswith(".json") and os.path.isfile(os.path.join(config_dir, f))
    )
    if not all_json_files:
        raise FileNotFoundError(f"No .json files in config directory: {config_dir}")

    configs_to_run = []
    if config_path_abs:
        cand_file = os.path.basename(config_path_abs)
        orig_idx = all_json_files.index(cand_file) if cand_file in all_json_files else 0
        configs_to_run = [(orig_idx, cand_file, config_path_abs)]
    else:
        configs_to_run = [(i, f, os.path.join(config_dir, f)) for i, f in enumerate(all_json_files)]

    log_step("Config", f"Found {len(all_json_files)} config(s): {all_json_files}", rank=rank)
    if selected:
        log_step("Config", f"Selected --config -> will run: {[x[1] for x in configs_to_run]}", rank=rank)
    log_newline(rank=rank)

    mem_backend_name, mem_backend = _get_mem_backend()
    csv_path = os.path.join(get_results_dir(), "runtime_info.csv")
    csv_fields = [
        "config_idx",
        "config_file",
        "module_type",
        "module_name",
        "Iteration",
        "Execution Time (s)",
        "NPU Memory (MB)",
        "loss",
        "output_preview",
    ]

    for run_idx, (orig_idx, filename, path) in enumerate(configs_to_run):
        config_idx = orig_idx + 1
        log_section(f"Config {run_idx + 1} / {len(configs_to_run)} (idx={config_idx}): {filename}", rank=rank)
        with open(path, "r", encoding="utf-8") as f:
            record = json.load(f)
        if "module_type" not in record or "module_name" not in record or "config" not in record:
            raise ValueError(
                f"Single module config must have module_type, module_name, config: {path}"
            )
        preprocess_single_module_config(record)
        module_type = record["module_type"]
        module_name = record["module_name"]
        config = record["config"]
        template_name = _module_type_to_template_name(module_type)
        template = TEMPLATE_REGISTRY.get(template_name)
        log_step("Template", template_name, rank=rank)
        log_step("Module", f"{module_type}: {module_name}", rank=rank)
        template.set_module_name(module_name)
        template.set_module_config(config)

        log_step("Build", "实例化 MMInstance", rank=rank)
        _sync_backend(mem_backend)
        t_build0 = time.perf_counter()
        mm_instance = template.instantiate()
        _sync_backend(mem_backend)
        build_time_s = time.perf_counter() - t_build0
        build_time_s_max = _dist_max(build_time_s, dist_enabled, get_device())
        mem_after_build = _get_mem_stats_mb(mem_backend)
        allocated_mb_after_build_max = _dist_max(mem_after_build.get("allocated_mb"), dist_enabled, get_device())
        reserved_mb_after_build_max = _dist_max(mem_after_build.get("reserved_mb"), dist_enabled, get_device())

        model_graph = [mm_instance]

        if ckpt_file_path or load_ckpt_enabled:
            if ckpt_file_path:
                ckpt_path = ckpt_file_path
            else:
                ckpt_path = _find_ckpt_for_config(load_ckpt_dir, config_idx, filename)
                if not ckpt_path:
                    raise FileNotFoundError(
                        f"Checkpoint not found for config {config_idx} ({filename}). "
                        f"load_ckpt_dir={load_ckpt_dir}"
                    )
            if dist_enabled:
                torch.distributed.barrier()
            if rank == 0:
                log_step("CKPT", f"Loading: {ckpt_path}", rank=rank)
            missing, unexpected = _load_mm_instance_ckpt(mm_instance=mm_instance, ckpt_path=ckpt_path)
            if rank == 0 and (missing or unexpected):
                log_step("CKPT", f"load_state_dict strict=False | missing={len(missing)} unexpected={len(unexpected)}", rank=rank)
            if dist_enabled:
                torch.distributed.barrier()

        optimizer = None
        if optimizer_enabled:
            from megatron.core.optimizer import get_megatron_optimizer, OptimizerConfig
            from megatron.core.distributed import DistributedDataParallelConfig
            from megatron.core.distributed import DistributedDataParallel as DDP

            ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=False)
            model = [
                DDP(
                    TransformerConfig(**mm_instance.config_dict),
                    ddp_config,
                    model_chunk,
                    disable_bucketing=(model_chunk_idx > 0),
                )
                for (model_chunk_idx, model_chunk) in enumerate(model_graph)
            ]
            optimizer_config = OptimizerConfig(
                optimizer="adam",
                lr=1e-4,
                weight_decay=0.01,
            )
            optimizer = get_megatron_optimizer(
                config=optimizer_config,
                model_chunks=model,
            )

        _reset_peak_stats(mem_backend)
        _sync_backend(mem_backend)
        for it in range(iterations):
            iter_idx = it + 1
            log_step("TEST", f"前向+Loss+反向 轮次 {iter_idx}", rank=rank)
            iter_start = time.perf_counter()
            output = model_graph[0].forward(step_seed=orig_idx * 100 + it)
            log_bullet(f"Output shape: {output.shape}", rank=rank)
            output.requires_grad_(True)
            loss = _compute_pretrain_style_loss(output)
            log_bullet(f"Loss: {loss.item():.6f}", rank=rank)
            log_bullet("反向传播中...", indent=2, rank=rank)
            loss.backward()
            if optimizer is not None:
                log_bullet("优化器 step...", indent=2, rank=rank)
                optimizer.step()
                optimizer.zero_grad()
            _sync_backend(mem_backend)
            iter_end = time.perf_counter()
            iter_exec_s = iter_end - iter_start
            iter_exec_s_max = _dist_max(iter_exec_s, dist_enabled, get_device())
            iter_mem = _get_mem_stats_mb(mem_backend).get("allocated_mb")
            iter_mem_max = _dist_max(iter_mem, dist_enabled, get_device())
            loss_val = float(loss.detach().item())
            last_rank = max(world_size - 1, 0)
            loss_val_from_last_rank = _dist_pick_rank(
                loss_val, dist_enabled, get_device(), src_rank=last_rank
            )
            output_preview = _summarize_output(output)
            _append_csv_row(
                csv_path=csv_path,
                fieldnames=csv_fields,
                row={
                    "config_idx": config_idx,
                    "config_file": filename,
                    "module_type": module_type,
                    "module_name": module_name,
                    "Iteration": iter_idx,
                    "Execution Time (s)": iter_exec_s_max,
                    "NPU Memory (MB)": iter_mem_max,
                    "loss": loss_val_from_last_rank,
                    "output_preview": output_preview,
                },
                rank=rank,
            )
            log_step("OK", f"轮次 {iter_idx} 完成: {filename}", rank=rank)
            log_newline(rank=rank)

        ckpt_path = None
        if save_ckpt_enabled:
            ckpt_path = _save_mm_instance_ckpt(
                mm_instance=mm_instance,
                ckpt_root=ckpt_root,
                run_tag=run_tag,
                config_idx=config_idx,
                config_file=filename,
                template_name=template_name,
                rank=rank,
            )
            if ckpt_path is not None:
                log_step("CKPT", f"Saved: {ckpt_path}", rank=rank)
            if dist_enabled:
                torch.distributed.barrier()


if __name__ == "__main__":
    main()
