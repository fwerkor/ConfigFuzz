import argparse, sys, json
import os
import unicodedata

# 保证 Ascend 精度模式为合法值，避免 ACL_PRECISION_MODE 报错
# 参考 MindSpeed / lm-sv 默认配置，使用 must_keep_origin_dtype
os.environ.setdefault("ASCEND_PRECISION_MODE", "must_keep_origin_dtype")

from typing import Any, Dict, List
import torch
import torch.nn.functional as F
import random
import secrets
import mindspeed.megatron_adaptor
from megatron.core import mpu

from megatron.training.initialize import initialize_megatron
from megatron.training import get_args

from mindspeed_mm.configs.config import mm_extra_args_provider

from modules.pool import TEXT_DECODER_DICT_POOL, IMAGE_ENCODER_DICT_POOL
from common import SupportedModules
from templates.template import TEMPLATE_REGISTRY
from templates.single_text_decoder_template import SingleTextDecoderTemplate
from templates.single_image_encoder_template import SingleImageEncoderTemplate
from utils import (
    set_results_dir,
    get_results_dir,
    resolve_path,
    log_section,
    log_step,
    log_bullet,
    make_json_serializable,
)
from single_module_mutator import SingleModuleMutator

npu_mem = torch.npu
import unicodedata


def _east_asian_display_width(s: str) -> int:
    """
    Approximate terminal display width for alignment.

    East Asian Wide/Full-width chars usually occupy 2 columns in terminals,
    while others occupy 1 column. Python's len() counts characters, not
    terminal columns, so we avoid len() for table alignment.
    """

    def _ch_width(ch: str) -> int:
        return 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1

    return sum(_ch_width(ch) for ch in s)


def _pad_display_width(s: str, width: int, align: str) -> str:
    w = _east_asian_display_width(s)
    pad_n = max(width - w, 0)
    if align == "left":
        return s + (" " * pad_n)
    if align == "right":
        return (" " * pad_n) + s
    raise ValueError(f"unknown align: {align!r}")


def extra_args_provider(parser):
    parser.add_argument(
        "--all-modules",
        required=False,
        type=str,
        default="./modules.json",
        help="all modules json file",
    )
    parser.add_argument(
        "--type",
        required=False,
        type=str,
        default=None,
        help="只针对某一类模块做变异，例如: text_decoder / image_encoder",
    )
    parser.add_argument(
        "--rounds",
        required=False,
        type=int,
        default=10,
        help="mutating rounds",
    )
    parser.add_argument(
        "--iterations",
        required=False,
        type=int,
        default=100,
        help="mutating iterations for each round",
    )
    parser.add_argument(
        "--results-dir",
        required=False,
        type=str,
        default="./results/_single_module",
        help="目录路径，用于存放运行产生的文件、日志等，默认为项目根目录下的 results",
    )
    parser.add_argument(
        "--no-create-results-subdir",
        action="store_true",
        help="不在 results-dir 下再自动创建 mutate_* 子目录，直接使用传入路径（供 single_module_mutate.sh --dir-name 使用）",
    )

    # MindSpeed-MM 会在 validate_args_patch 阶段假设 args.mm.model 必然存在（即 mm_model 必须可加载）。
    # single_module_mutate 不绑定任何固定训练模型，这里提供项目内 dummy_mm/ 作为默认占位配置；
    # 如需真实配置，可在命令行显式传入 --mm-model/--mm-data/--mm-tool 覆盖。
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
    target_hidden = 2048
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

def config_preprocess(config: Dict[str, Any]):
    for _, modules in config.items():
        for _, cfg in modules.items():
            if isinstance(cfg, dict):
                _apply_config_preprocess_to_cfg(cfg)
    return config


def _parse_type_to_supported_module(type_str: str) -> SupportedModules:
    """把用户输入的 type 字符串解析成 SupportedModules。

    支持的示例: text_decoder, image_encoder, text decoder, image encoder, text, image ...
    """
    if type_str is None:
        raise ValueError("type_str must not be None")
    normalized = (
        type_str.strip()
        .lower()
        .replace(" ", "")
        .replace("-", "")
        .replace("_", "")
    )
    if "text" in normalized or normalized in {"textdecoder", "decoder"}:
        return SupportedModules.TEXT_DECODER
    if "image" in normalized or normalized in {"imageencoder", "encoder"}:
        return SupportedModules.IMAGE_ENCODER
    raise ValueError(
        f"Unsupported --type={type_str!r}. Expect text_decoder or image_encoder."
    )

def register_all_modules(all_modules: Dict[str, Any], args: argparse.Namespace):
    text_decoders = all_modules["text_decoders"]
    text_encoders = all_modules["text_encoders"]
    image_encoders = all_modules["image_encoders"]

    for name, config in text_decoders.items():
        if name == "qwen2vl_2b" or name == "videoalign":
            continue
        TEXT_DECODER_DICT_POOL.register_one(name, config)
    for name, config in image_encoders.items():
        IMAGE_ENCODER_DICT_POOL.register_one(name, config)

def register_single_templates():
    TEMPLATE_REGISTRY.register(SingleTextDecoderTemplate())
    TEMPLATE_REGISTRY.register(SingleImageEncoderTemplate())


def _get_mutation_schema_path() -> str:
    """返回单模块变异 schema 的路径（相对当前文件）"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, "single_module_mutate_dict.json")


def main():
    initialize_megatron(extra_args_provider=extra_args_provider, args_defaults={})
    args = get_args()

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
    register_single_templates()

    type_filter_module: SupportedModules | None = None
    allowed_templates = TEMPLATE_REGISTRY.templates
    if getattr(args, "type", None):
        type_filter_module = _parse_type_to_supported_module(args.type)
        allowed_templates = [
            t for t in allowed_templates if t.get_module_type() == type_filter_module
        ]
        if not allowed_templates:
            raise ValueError(
                f"No templates found for type={args.type!r} (parsed: {type_filter_module})"
            )
        log_step("Config", f"Type filter: {type_filter_module.value}")

    # 初始化单模块变异器
    mutation_schema_path = _get_mutation_schema_path()
    single_mutator = SingleModuleMutator(schema_path=mutation_schema_path)

    # mutate: 每轮随机选择一个模块类型，每次迭代在上一轮配置基础上做变异
    rounds = args.rounds
    iterations = args.iterations
    log_step("Config", f"Rounds: {rounds}")
    log_step("Config", f"Iterations per round: {iterations}")
    print()

    round_summaries: List[Dict[str, Any]] = []

    for round in range(rounds):
        log_section(f"Round {round + 1} / {rounds}")
        # 每轮在允许的模板集合里随机选择一个（单 text decoder 或单 image encoder）
        template = (
            allowed_templates[0]
            if len(allowed_templates) == 1
            else secrets.choice(allowed_templates)
        )
        log_step("Template", template.name)
        log_step("Select", "选择模块")
        template.select_modules()

        module_type = template.get_module_type()
        module_name = template.get_module_name()
        current_config = template.get_module_config()
        log_step("Module", f"{module_type}: {module_name}")

        round_iter_success = 0

        for iteration in range(iterations):
            log_section(f"Iteration {iteration + 1} / {iterations}: {module_type.value.upper()} from {module_name}")

            # 记录本轮变异前配置，用于崩溃时回退
            before_config = dict(current_config) if current_config is not None else {}

            # iteration 0 使用 template.select_modules 选出的原始配置，
            # 之后每次迭代在上一迭代的配置基础上做变异
            if iteration == 0:
                log_step("Mutation", "使用原始配置（未变异）")
                mutated_config = current_config
            else:
                mutated_config = single_mutator.mutate(
                    module_type=module_type,
                    base_config=current_config,
                    mutation_num=3,
                )
                log_step("Mutation", f"在上一迭代配置基础上变异，模块类型: {module_type.value.upper()}")

            # 更新当前配置，供下一次迭代继续变异
            current_config = mutated_config

            # 将变异后的配置写回模板实例
            template.set_module_config(mutated_config)

            try:
                # 实例化 MMInstance
                log_step("Build", "实例化 MMInstance")
                mm_instance = template.instantiate()

                # 前向计算+Loss+反向传播
                log_step("TEST", "前向计算 -> Loss -> 反向传播")
                output = mm_instance.forward()
                log_bullet(f"Output shape: {output.shape}")
                output.requires_grad_(True)
                loss = output.norm()
                log_bullet(f"Loss: {loss.item():.6f}")
                log_bullet("反向传播中...", indent=2)
                loss.backward()
                log_step("OK", "本 iteration 完成")

                # 在成功完成测试的情况下，将配置保存为 JSON 文件
                round_idx = round + 1
                iter_idx = iteration + 1
                config_record = make_json_serializable(
                    {
                        "module_type": module_type.value,
                        "module_name": module_name,
                        "config": current_config,
                    }
                )
                config_filename = f"{round_idx}-{iter_idx}-{module_type.value}.json"
                config_path = resolve_path("configs", config_filename)
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(config_record, f, ensure_ascii=False, indent=2)
                round_iter_success += 1
            except Exception as e:
                log_step("Error", f"本 iteration 崩溃，原因: {e}")
                log_bullet("无效变异，回退到变异前配置并继续下一次迭代", indent=2)
                # 回退到变异前配置
                current_config = before_config
                template.set_module_config(before_config)
                continue

        round_summaries.append(
            {
                "round_index": round + 1,
                "module_type": module_type.value,
                "module_name": module_name,
                "iter_success": round_iter_success,
                "iter_total": iterations,
            }
        )

    # 运行总结（脚本正常结束前打印）
    log_section("多模态模型模块内变异总结")
    log_step("Summary", f"运行轮数 (rounds): {rounds}，每轮迭代次数: {iterations}")
    log_step("Summary", f"结果保存目录: {get_results_dir()}")
    log_step("Summary", "各轮模块与迭代成功率:")
    if round_summaries and (
        not torch.distributed.is_initialized()
        or torch.distributed.get_rank() == 0
    ):
        w_mt = max(len("类型"), *(len(r["module_type"]) for r in round_summaries))
        w_mn = max(len("模块"), *(len(r["module_name"]) for r in round_summaries))
        col_frac = max(
            len(f"{r['iter_success']}/{r['iter_total']}") for r in round_summaries
        )
        ind = "    "
        hdr = (
            f"{ind}{'Round':>5}  "
            f"{'模块类型':<{w_mt}}  "
            f"{'来源模型':<{w_mn}}  "
            f"{'迭代':>{col_frac}}  "
            f"{'变异成功率':>8}"
        )
        print(hdr)
        print(f"{ind}{'-' * (len(hdr) - len(ind))}")
        for rec in round_summaries:
            r = rec["round_index"]
            mt = rec["module_type"]
            mn = rec["module_name"]
            isucc, itot = rec["iter_success"], rec["iter_total"]
            ipct = (100.0 * isucc / itot) if itot else 0.0
            frac = f"{isucc}/{itot}"
            print(
                f"{ind}{r:>5}  "
                f"{mt:<{w_mt}}  "
                f"{mn:<{w_mn}}  "
                f"{frac:>{col_frac}}  "
                f"{ipct:>7.1f}%"
            )
    print()

if __name__ == "__main__":
    main()