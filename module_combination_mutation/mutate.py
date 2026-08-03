import argparse, sys, json
import os

# 保证 Ascend 精度模式为合法值，避免 ACL_PRECISION_MODE 报错
# 参考 MindSpeed / lm-sv 默认配置，使用 must_keep_origin_dtype
os.environ.setdefault("ASCEND_PRECISION_MODE", "must_keep_origin_dtype")

from typing import Dict, Any
import torch

import mindspeed.megatron_adaptor
from megatron.core import mpu
from megatron.training.initialize import initialize_megatron
from megatron.training import get_args

from mindspeed_mm.configs.config import mm_extra_args_provider

from modules.pool import TEXT_DECODER_DICT_POOL, IMAGE_ENCODER_DICT_POOL, AE_DICT_POOL
from templates.template import TEMPLATE_REGISTRY
from templates.image_model_template import ImageModelTemplate
from templates.video_model_template import VideoModelTemplate
from it_combine.interleave import InterleaveCombineStrategy
from it_combine.strategy import IMAGE_TEXT_COMBINE_STRATEGY_REGISTRY
from utils import set_results_dir, get_results_dir, resolve_path, log_section, log_step, log_bullet, make_json_serializable

npu_mem = torch.npu

def extra_args_provider(parser):
    parser.add_argument(
        "--all-modules",
        required=False,
        type=str,
        default="./modules.json",
        help="all modules json file",
    )
    parser.add_argument(
        "--rounds",
        required=False,
        type=int,
        default=10,
        help="mutating rounds",
    )
    parser.add_argument(
        "--results-dir",
        required=False,
        type=str,
        default="./results",
        help="目录路径，用于存放运行产生的文件、日志等，默认为项目根目录下的 results",
    )
    parser.add_argument(
        "--no-create-results-subdir",
        action="store_true",
        help="不在 results-dir 下再自动创建 mutate_* 子目录，直接使用传入路径（供 mm_mutate.sh --dir-name 使用）",
    )

    # MindSpeed-MM 会在 validate_args_patch 阶段假设 args.mm.model 必然存在（即 mm_model 必须可加载）。
    # 本项目不是“固定模型训练”，但仍需要给 MindSpeed-MM 一个合法的占位配置以通过校验。
    # 这里把默认值固定到项目内的 dummy_mm/ 下；如需真实配置，可在命令行显式传入覆盖。
    parser = mm_extra_args_provider(parser)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dummy_dir = os.path.join(script_dir, "dummy_mm")
    parser.set_defaults(
        mm_model=os.path.join(dummy_dir, "model.json"),
        mm_data=os.path.join(dummy_dir, "data.json"),
        mm_tool=os.path.join(dummy_dir, "tool.json"),
    )
    return parser

def load_modules_json(json_path: str) -> Dict[str, Any]:
    """加载 modules.json 文件"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def _apply_config_preprocess_to_cfg(cfg: Dict[str, Any]) -> None:
    """对单个 config 应用预处理，并递归处理嵌套的 vision_encoder、vision_projector"""
    if cfg.get("position_embedding_type", "rope") == "mrope":
        cfg["use_fused_rotary_pos_emb"] = False
    if isinstance(cfg.get("params_dtype"), str):
        mapping = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }
        cfg["params_dtype"] = mapping[cfg["params_dtype"]]
    if "shape_order" not in cfg:
        cfg["shape_order"] = "SBH"
    if cfg.get("moe_grouped_gemm", False):
        cfg["moe_grouped_gemm"] = False

    # minimize the model size（统一 hidden_size，包括嵌套子模块）
    # NPU RoPE 要求 head_dim 为偶数，故 hidden_size 需 >= num_attention_heads * 2
    target_hidden = 32
    target_heads = 16
    if cfg.get("hidden_size", target_hidden) > target_hidden:
        cfg["hidden_size"] = target_hidden
    if cfg.get("num_attention_heads", target_heads) > target_heads:
        cfg["num_attention_heads"] = target_heads
    # num_query_groups 需 <= num_attention_heads，否则 attention view 会与 linear 维度不一致
    num_heads = cfg.get("num_attention_heads", target_heads)
    if cfg.get("num_query_groups", num_heads) > num_heads:
        cfg["num_query_groups"] = num_heads
    if cfg.get("ffn_hidden_size", 64) > 64:
        cfg["ffn_hidden_size"] = 64
    if cfg.get("num_moe_experts", 4) > 4:
        cfg["num_moe_experts"] = 4
    # group_limited_topk 要求 topk//group_topk <= num_experts//num_groups，否则 NPU topk 报错
    num_experts = cfg.get("num_moe_experts", 4)
    num_groups = max(cfg.get("n_group", 1), 1)
    if num_groups > num_experts:
        num_groups = num_experts
        cfg["n_group"] = num_groups
    group_topk = max(cfg.get("topk_group", 1), 1)
    max_topk_per_group = num_experts // num_groups
    max_topk = max_topk_per_group * group_topk
    if cfg.get("moe_router_topk", 0) > max_topk:
        cfg["moe_router_topk"] = max_topk
    if cfg.get("moe_ffn_hidden_size", 32) > 32:
        cfg["moe_ffn_hidden_size"] = 32

    # 递归处理嵌套子模块（如 image_encoder 的 vision_encoder、vision_projector）
    for key in ("vision_encoder", "vision_projector",):
        if key in cfg and isinstance(cfg[key], dict):
            _apply_config_preprocess_to_cfg(cfg[key])

    # vision_projector.input_size 需与 vision_encoder.hidden_size 一致
    if "vision_encoder" in cfg and "vision_projector" in cfg:
        ve = cfg["vision_encoder"]
        vp = cfg["vision_projector"]
        if isinstance(ve, dict) and isinstance(vp, dict) and "hidden_size" in ve:
            vp["input_size"] = ve["hidden_size"]
            # InternVL projector (InternVLMLP) uses `vit_hidden_size` to derive its LayerNorm/input dims.
            # When we cap/scale `vision_encoder.hidden_size` for smaller test models,
            # we must keep `vision_projector.vit_hidden_size` consistent.
            if "vit_hidden_size" in vp:
                vp["vit_hidden_size"] = ve["hidden_size"]

def config_preprocess(config: Dict[str, Any]):
    for _, modules in config.items():
        for _, cfg in modules.items():
            if isinstance(cfg, dict):
                _apply_config_preprocess_to_cfg(cfg)
    return config

def register_all_modules(all_modules: Dict[str, Any], args: argparse.Namespace):
    text_decoders = all_modules["text_decoders"]
    text_encoders = all_modules["text_encoders"]
    image_encoders = all_modules["image_encoders"]
    aes = all_modules["aes"]

    for name, config in text_decoders.items():
        if name == "qwen2vl_2b" or name == "videoalign":
            continue
        TEXT_DECODER_DICT_POOL.register_one(name, config)
    for name, config in image_encoders.items():
        IMAGE_ENCODER_DICT_POOL.register_one(name, config)
    for name, config in aes.items():
        AE_DICT_POOL.register_one(name, config)

def register_all_templates():
    TEMPLATE_REGISTRY.register(ImageModelTemplate())
    # TEMPLATE_REGISTRY.register(VideoModelTemplate())

def register_all_image_text_combine_strategies():
    IMAGE_TEXT_COMBINE_STRATEGY_REGISTRY.register("interleave", InterleaveCombineStrategy())

def main():
    initialize_megatron(extra_args_provider=extra_args_provider, args_defaults={})
    args = get_args()
    # finish_mpu_init()
    # 打印是否启动distributed
    print(f"Distributed: {torch.distributed.is_initialized()}")
    print(f"Rank: {torch.distributed.get_rank()}")
    print(f"World size: {torch.distributed.get_world_size()}")

    # 初始化 results 目录，并让 profile 等输出写入该目录
    set_results_dir(
        args.results_dir,
        create_run_subdir=not getattr(args, "no_create_results_subdir", False),
    )
    if hasattr(args, "profile_save_path"):
        args.profile_save_path = os.path.join(get_results_dir(), "profile")
    log_section("模块组合变异工具")
    log_step("Init", f"Results directory: {get_results_dir()}")

    if not hasattr(args, "rope_scaling_type"):
        args.rope_scaling_type = None
    if not hasattr(args, "rope_scaling_factor"):
        args.rope_scaling_factor = 1.0
    if not hasattr(args, "rope_scaling_mscale_all_dim"):
        args.rope_scaling_mscale_all_dim = 0
    
    if args.qk_nope_head_dim is None and args.qk_rope_head_dim is None:
        args.qk_nope_head_dim = 0
        args.qk_rope_head_dim = args.hidden_size // args.num_attention_heads

    # init the modules
    all_modules = load_modules_json(args.all_modules)
    all_modules = config_preprocess(all_modules)
    register_all_modules(all_modules, args)
    register_all_templates()
    register_all_image_text_combine_strategies()

    # mutate
    rounds = args.rounds
    log_step("Config", f"Rounds: {rounds}")
    print()

    for round in range(rounds):
        log_section(f"Round {round + 1} / {rounds}")
        template = TEMPLATE_REGISTRY.random_choice()
        log_step("Template", template.name)

        log_step("Select", "选择模块")
        template.select_modules()

        log_step("Build", "实例化 MMInstance")
        mm_instance = template.instantiate()
        log_step("TEST", "前向计算 Loss 反向传播")
        output = mm_instance.forward()
        log_bullet(f"Output shape: {output.shape}")
        output.requires_grad_(True)
        loss = output.norm()
        log_bullet(f"Loss: {loss.item():.6f}")
        log_bullet("反向传播中...", indent=2)
        loss.backward()
        log_step("OK", "本 round 完成")
        instance_config = mm_instance.get_config_dict()
        with open(resolve_path("configs", f"round_{round}.json"), "w") as f:
            json.dump(make_json_serializable(instance_config), f, indent=4)
        dot_path = template.dump_graph(output_path=resolve_path("dots", f"graph_round{round}.dot"))
        log_step("Graph", f"Saved to: {dot_path}")

if __name__ == "__main__":
    main()