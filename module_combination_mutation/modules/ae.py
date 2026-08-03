import torch

from mindspeed_mm.models.ae import AEModel
from mindspeed_mm.configs.config import ConfigReader

from utils import log_step, get_device

def register_ae(*names: str):
    """装饰器：将 ae 类注册到构建器表，扩展新 ae 时只需加此装饰器，无需修改 pool.py"""
    def decorator(cls):
        for name in names:
            AE_BUILDERS[name] = cls
        return cls
    return decorator


AE_BUILDERS = {}

class AE:
    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config
        self.ae = None

    def _build(self):
        # Subclasses should create `self.ae` here.
        pass

    def encode(self, video: torch.Tensor) -> torch.Tensor:
        pass

    def _resolve_target_dtype(self):
        """Resolve target dtype from `config['dtype']` like "bf16"/"fp16"/"fp32"."""
        dtype_cfg = getattr(self, "config", {}).get("dtype", None)
        if isinstance(dtype_cfg, torch.dtype):
            return dtype_cfg
        if isinstance(dtype_cfg, str):
            return {
                "bf16": torch.bfloat16,
                "bfloat16": torch.bfloat16,
                "fp16": torch.float16,
                "fp32": torch.float32,
            }.get(dtype_cfg, None)
        return None

    def _move_and_align_dtype(self):
        if self.ae is None:
            return

        try:
            device = get_device()
            target_dtype = self._resolve_target_dtype()
            if target_dtype is not None:
                self.ae.to(device=device, dtype=target_dtype)
            else:
                self.ae.to(device=device)
        except Exception:
            # Keep the tool running even if dtype/device cast fails for some models.
            log_step("Warning", f"Failed to move/cast ae '{self.name}'", indent=1)

@register_ae("opensora2.0", "opensora1.2")
class OpenSora2AE(AE):
    @classmethod
    def build(cls, name: str, config: dict) -> "AE":
        log_step("Build", f"ae: {name}", indent=1)
        return cls(name, config)

    def __init__(self, name: str, config: dict):
        super().__init__(name, config)
        self.model_config = ConfigReader({"ae": config})

        self._build()

    def _build(self):
        self.ae = AEModel(self.model_config.ae).eval()
        self._move_and_align_dtype()

    
    def encode(self, video: torch.Tensor) -> torch.Tensor:
        latents, _ = self.ae.encode(video)
        return latents