import os
import time
import torch, random
from typing import Tuple, Optional, Dict, Any

# 结果目录：全局存储当前 results 路径，在 main 中初始化
_results_dir: str = ""

# 控制台输出宽度，用于分隔线等
_CONSOLE_WIDTH = 72


def get_device() -> torch.device:
    """返回当前进程应使用的 NPU 设备（分布式下为当前 rank 的 NPU，单卡为 npu:0）。"""
    try:
        if getattr(torch.npu, "is_available", lambda: False)():
            return torch.device(f"npu:{torch.npu.current_device()}")
        return torch.device("cpu")
    except Exception:
        return "Ascend"


def _should_log(rank) -> bool:
    """仅 rank 0 输出。rank 为 None 且已初始化分布式时，用当前进程 rank 判断；否则视为单进程，输出。"""
    if rank is not None:
        return rank == 0
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return True


def print_line(length: int = None, rank=None) -> None:
    """打印一条分隔线。rank 非 None 且不为 0 时不打印。"""
    if not _should_log(rank):
        return
    print("-" * (length or _CONSOLE_WIDTH))


def log_section(title: str, width: int = None, rank=None) -> None:
    """打印分节标题（带上下边框）。rank 非 None 且不为 0 时不打印。"""
    if not _should_log(rank):
        return
    w = width or _CONSOLE_WIDTH
    print()
    print("=" * w)
    print(f"  {title}")
    print("=" * w)


def log_step(prefix: str, message: str, indent: int = 0, rank=None) -> None:
    """带前缀的单行日志，如 [Select] xxx。rank 非 None 且不为 0 时不打印。"""
    if not _should_log(rank):
        return
    pad = "  " * indent
    print(f"{pad}[{prefix}] {message}")


def log_bullet(message: str, indent: int = 1, rank=None) -> None:
    """缩进的条目，如   • xxx。rank 非 None 且不为 0 时不打印。"""
    if not _should_log(rank):
        return
    pad = "  " * indent
    print(f"{pad}• {message}")


def log_newline(rank=None) -> None:
    """打印一个空行。rank 非 None 且不为 0 时不打印。"""
    if not _should_log(rank):
        return
    print()


def log_tensor_summary(name: str, tensor: torch.Tensor, *, indent: int = 2, rank=None, max_items: int = 8) -> None:
    """Log a compact tensor summary for PTA/MSA step-by-step alignment."""
    try:
        detached = tensor.detach()
        flat = detached.reshape(-1)
        take_n = min(int(flat.numel()), max_items)
        sample = flat[:take_n].float().cpu().tolist()
        sample_text = ", ".join(f"{v:.6g}" for v in sample)
        stats = detached.float()
        min_v = float(stats.min().item()) if stats.numel() > 0 else 0.0
        max_v = float(stats.max().item()) if stats.numel() > 0 else 0.0
        mean_v = float(stats.mean().item()) if stats.numel() > 0 else 0.0
        if stats.numel() > 0:
            try:
                std_v = float(stats.std(unbiased=False).item())
            except TypeError:
                try:
                    std_v = float(stats.std(correction=0).item())
                except TypeError:
                    std_v = float(stats.std(ddof=0).item())
        else:
            std_v = 0.0
        log_bullet(
            f"{name}: shape={tuple(detached.shape)} dtype={detached.dtype} "
            f"min={min_v:.6g} max={max_v:.6g} mean={mean_v:.6g} std={std_v:.6g} "
            f"sample=[{sample_text}]",
            indent=indent,
            rank=rank,
        )
    except Exception as exc:
        log_bullet(f"{name}: summary_failed={exc}", indent=indent, rank=rank)


def make_json_serializable(obj):
    """递归将 dict/list 中的 dtype、torch/numpy 等不可 JSON 序列化的对象转为可序列化形式。"""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(v) for v in obj]
    # 原生 JSON 支持的类型直接返回
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    # numpy/torch dtype（type(obj).__name__ == "dtype"）
    if type(obj).__name__ == "dtype":
        return str(obj)
    # torch 的 dtype 如 torch.float32 实际是 type 的实例
    if isinstance(obj, type):
        return f"{getattr(obj, '__module__', '')}.{obj.__name__}"
    try:
        import numpy as np
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.dtype):
            return str(obj)
    except ImportError:
        pass
    # 其他不可序列化对象转为字符串
    return str(obj)


def set_results_dir(path: str, create_run_subdir: bool = True) -> str:
    """设置并创建 results 目录，返回绝对路径。
    若 create_run_subdir 且 path 不是 run_* 子目录，则在其下创建 run_YYYYMMDD_HHMMSS。
    """
    global _results_dir
    path = os.path.abspath(os.path.expanduser(path))
    if create_run_subdir and "mutate_" not in os.path.basename(path):
        run_name = "mutate_" + time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(path, run_name)
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, "profile"), exist_ok=True)
    os.makedirs(os.path.join(path, "configs"), exist_ok=True)
    _results_dir = path
    return _results_dir


def get_results_dir() -> str:
    """获取当前 results 目录路径"""
    return _results_dir if _results_dir else os.path.abspath("./results")


def get_logs_dir() -> str:
    """获取日志目录路径（与 results 目录相同，日志直接存放在 run 目录下）"""
    return get_results_dir()


def get_profile_dir() -> str:
    """获取 profile 子目录路径"""
    return os.path.join(get_results_dir(), "profile")


def resolve_path(*parts: str) -> str:
    """在 results 目录下解析路径，如 resolve_path('output', 'fusion_result.json')"""
    return os.path.join(get_results_dir(), *parts)

def ensure_decoder_input_shape(
    input: torch.Tensor,
    hidden_size: int,
    decoder,
    share_embeddings_and_output_weights: bool = False,
) -> torch.Tensor:
    """
    将输入投影为 decoder 期望的 hidden [batch, seq, hidden_size]。
    - 若 input 最后一维已是 hidden_size，直接返回。
    - 若 input 为 logits [batch, seq, vocab]，用 output_layer 投影。
    - 若 input 为其他 hidden 维度（如 combine 后维度不一致），用线性投影对齐形状。
    """
    input_dim = input.shape[-1]
    if input_dim == hidden_size:
        return input
    # 上一 decoder 的 logits -> hidden
    if share_embeddings_and_output_weights:
        output_weight = decoder.shared_embedding_or_output_weight()
    else:
        # GPTModel 在 post_process=False 时没有 output_layer，部分模型用 lm_head
        out_layer = getattr(decoder, "output_layer", None) or getattr(decoder, "lm_head", None)
        output_weight = out_layer.weight if out_layer is not None and hasattr(out_layer, "weight") else None
    if output_weight is not None and output_weight.shape[0] == input_dim:
        return torch.matmul(input, output_weight)
    # 其他 hidden 维度不一致：使用确定性 shape 对齐，避免每步新建随机初始化线性层。
    if input_dim > hidden_size:
        return input[..., :hidden_size]
    pad_shape = list(input.shape)
    pad_shape[-1] = hidden_size - input_dim
    pad = torch.zeros(*pad_shape, device=input.device, dtype=input.dtype)
    return torch.cat([input, pad], dim=-1)


def create_decoder_attention_mask(
    attention_mask: torch.Tensor,
    seq_len: int,
    causal: bool = True,
) -> torch.Tensor:
    """
    将 [batch, seq] 的 attention_mask 转为 NPU FlashAttention 所需的 4D 格式 [batch, 1, seq, seq]。
    attention_mask: True 表示有效 token。
    返回: True 表示需要 mask（不 attend）的位置。
    """
    batch_size = attention_mask.shape[0]
    device = attention_mask.device
    # 因果 mask：j > i 时 mask
    if causal:
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        )
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len)
    else:
        causal_mask = torch.zeros(batch_size, 1, seq_len, seq_len, device=device, dtype=torch.bool)
    # padding mask：key 为 padding 时 mask
    padding_mask = (~attention_mask).unsqueeze(1).unsqueeze(2)
    padding_mask = padding_mask.expand(batch_size, 1, seq_len, seq_len)
    return causal_mask | padding_mask

def generate_input_ids_and_position_ids(
    batch_size: int,
    vocab_size: int,
    max_sequence_length: int = 16,
    is_random: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate input ids, position ids and attention mask for a batch of data.
    Args:
        batch_size: The batch size of the data.
        vocab_size: The vocabulary size of the data.
        max_sequence_length: The maximum sequence length of the data.
        is_random: Whether to generate random input_ids. If False, generate deterministic input_ids.
    Returns:
        input_ids: [batch_size, max_sequence_length]
        position_ids: [batch_size, max_sequence_length]
        attention_mask: [batch_size, max_sequence_length], 1 for valid tokens
    """
    device = get_device()
    import os
    msa_run = os.environ.get("LMSV_MM_MSARUN") == "1"
    input_device = "Ascend" if msa_run else device
    if is_random:
        input_ids = torch.randint(0, vocab_size, (batch_size, max_sequence_length), dtype=torch.long, device=input_device)
    else:
        total_tokens = batch_size * max_sequence_length
        input_ids = torch.arange(total_tokens, dtype=torch.long, device=input_device).reshape(batch_size, max_sequence_length) % vocab_size
    position_ids = torch.arange(max_sequence_length, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
    # NPU FlashAttentionScore 仅支持 bool 或 uint8，不支持 int64
    attention_mask = torch.ones((batch_size, max_sequence_length), dtype=torch.bool, device=device)
    return input_ids, position_ids, attention_mask

def generate_images(
    batch_size: int,
    image_size: int,
    dtype: torch.dtype = torch.bfloat16,
    is_random: bool = True,
) -> torch.Tensor:
    """
    Generate images for a batch of data.
    Args:
        batch_size: The batch size of the data.
        image_size: The size of the images.
        is_random: Whether to generate random images. If False, generate deterministic images.
    Returns:
        images: [batch_size, 3, image_size, image_size]
    """
    # 与文本输入保持一致，直接在 NPU 上生成图像，避免 CPU / NPU 设备不一致；分布式下使用当前 rank 设备
    device = get_device()
    if is_random:
        return torch.randn(batch_size, 3, image_size, image_size, device=device, dtype=dtype)
    total = batch_size * 3 * image_size * image_size
    base = torch.arange(total, device=device, dtype=torch.float32)
    # 归一化到 [-1, 1]，保证固定且数值稳定，再转换到目标 dtype
    fixed = (base % 1024) / 511.5 - 1.0
    return fixed.reshape(batch_size, 3, image_size, image_size).to(dtype=dtype)


def _round_down_to_multiple(value: int, multiple: int) -> int:
    """Round `value` down to nearest multiple of `multiple` (at least `multiple`)."""
    if multiple <= 0:
        return value
    value = int(value)
    if value < multiple:
        return multiple
    return (value // multiple) * multiple


def _resolve_dtype(dtype: Optional[object], ae_config: Optional[Dict[str, Any]] = None) -> torch.dtype:
    """Resolve dtype from either explicit `dtype` or `ae_config['dtype']`."""
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        return {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}.get(
            dtype, torch.bfloat16
        )
    if ae_config is not None:
        cfg_dtype = ae_config.get("dtype", None)
        if isinstance(cfg_dtype, torch.dtype):
            return cfg_dtype
        if isinstance(cfg_dtype, str):
            return {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}.get(
                cfg_dtype, torch.bfloat16
            )
    return torch.bfloat16


def generate_video_tensor(
    batch_size: int,
    num_frames: int,
    height: int,
    width: int,
    dtype: Optional[object] = None,
    ae_config: Optional[Dict[str, Any]] = None,
    device: Optional[torch.device] = None,
    value_range: Tuple[float, float] = (-1.0, 1.0),
    spatial_multiple: int = 8,
    is_random: bool = True,
) -> torch.Tensor:
    """
    Generate a 5D video tensor in shape expected by most VAE implementations: [B, 3, T, H, W].

    Notes:
    - We round H/W down to `spatial_multiple` (OpenSora VAE commonly needs multiples of 8).
    - Values are sampled from `value_range` (default: [-1, 1]) to mimic normalized pixels.
    """
    if device is None:
        device = get_device()

    dtype_ = _resolve_dtype(dtype, ae_config=ae_config)

    t = max(int(num_frames), 1)
    h = _round_down_to_multiple(int(height), spatial_multiple)
    w = _round_down_to_multiple(int(width), spatial_multiple)

    low, high = value_range
    if is_random:
        # torch.rand doesn't support bfloat16 on all backends; sample in fp32 then cast.
        x = torch.rand(batch_size, 3, t, h, w, device=device, dtype=torch.float32)
        x = x * (high - low) + low
    else:
        total = batch_size * 3 * t * h * w
        # Use deterministic pattern to ensure exactly the same tensor across repeated calls.
        base = torch.arange(total, device=device, dtype=torch.float32) % 1024
        x = (base / 1023.0) * (high - low) + low
        x = x.reshape(batch_size, 3, t, h, w)
    return x.to(dtype_)


def generate_video_tensor_from_ae_config(
    batch_size: int,
    num_frames: int,
    ae_config: Optional[Dict[str, Any]] = None,
    max_spatial_size: int = 64,
    max_num_frames: int = 8,
    dtype: Optional[object] = None,
    device: Optional[torch.device] = None,
    value_range: Tuple[float, float] = (-1.0, 1.0),
    is_random: bool = True,
) -> torch.Tensor:
    """
    Convenience wrapper: infer (H, W, dtype) from `ae_config` and cap them to keep tests light.
    """
    ae_config = ae_config or {}
    sample_size = ae_config.get("sample_size", max_spatial_size)
    # Keep frames small by default since AE encode is expensive.
    t = min(int(num_frames), int(max_num_frames))
    if "sample_tsize" in ae_config and t <= 0:
        t = int(ae_config["sample_tsize"])

    height = int(min(sample_size, max_spatial_size))
    width = height
    return generate_video_tensor(
        batch_size=batch_size,
        num_frames=t,
        height=height,
        width=width,
        dtype=dtype,
        ae_config=ae_config,
        device=device,
        value_range=value_range,
        is_random=is_random,
    )