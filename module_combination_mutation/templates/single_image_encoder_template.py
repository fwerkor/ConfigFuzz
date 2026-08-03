import random
import secrets
import torch

from templates.template import Template, MMInstance
from modules.pool import IMAGE_ENCODER_DICT_POOL
from utils import generate_images, log_bullet
from common import SupportedModules

class SingleImageEncoderTemplate(Template):
    def __init__(self):
        super().__init__("single_image_encoder")

        # the single module
        self.image_encoder_name = None
        self.image_encoder_config = None

    def select_modules(self):
        # randomly select a image encoder
        image_encoder_config = IMAGE_ENCODER_DICT_POOL.random_choice()
        self.image_encoder_name = list(image_encoder_config.keys())[0]
        self.image_encoder_config = image_encoder_config[self.image_encoder_name]

    def instantiate(self) -> MMInstance:
        config = {
            "image_encoder": self.image_encoder_config,
            "text_decoder": None,
            "text_encoder": None,
            "video_encoder": None
        }

        instance = MMInstance(name="single_image_encoder", config=config, template=self)
        image_encoder = IMAGE_ENCODER_DICT_POOL.build(self.image_encoder_name, self.image_encoder_config)
        instance.set_image_encoder(image_encoder, self.image_encoder_config)
        return instance

    def forward(self, instance: MMInstance, step_seed=None):
        # 随机 batch_size 和 max_sequence_length；分布式下 step_seed 由调用方传入，保证各 rank 同一步得到相同 shape
        # if step_seed is not None:
        #     rng = random.Random(step_seed)
        #     batch_size = rng.randrange(1, 5)
        #     max_sequence_length = rng.randrange(1, 17)
        # else:
        #     batch_size = secrets.randbelow(4) + 1
        #     max_sequence_length = secrets.randbelow(16) + 1

        batch_size = 4

        image_encoder = instance.image_encoder
        # Keep image input dtype consistent with vision encoder params_dtype.
        dtype = torch.bfloat16
        try:
            ve_cfg = getattr(image_encoder, "config", {}).get("vision_encoder", {})
            params_dtype = ve_cfg.get("params_dtype", None)
            if isinstance(params_dtype, torch.dtype):
                dtype = params_dtype
            elif isinstance(params_dtype, str):
                dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}.get(params_dtype, torch.bfloat16)
            else:
                if ve_cfg.get("bf16", False) is True:
                    dtype = torch.bfloat16
                elif ve_cfg.get("fp16", False) is True:
                    dtype = torch.float16
                else:
                    dtype = torch.float32
        except Exception:
            dtype = torch.bfloat16
        images = generate_images(batch_size, image_encoder.image_size, dtype=dtype, is_random=False)
        log_bullet(f"images {images.shape}", indent=2)
        image_features = image_encoder.encode(images)
        log_bullet(f"image_features {image_features.shape}", indent=2)

        return image_features

    def get_module_type(self) -> SupportedModules:
        """获取模块类型（仅用于模块内变异）"""
        return SupportedModules.IMAGE_ENCODER

    def get_module_name(self) -> str:
        """获取模块名称（仅用于模块内变异）"""
        return self.image_encoder_name

    def set_module_name(self, module_name: str):
        """设置模块名称（仅用于模块内变异）"""
        self.image_encoder_name = module_name

    def get_module_config(self) -> dict:
        """获取模块配置（仅用于模块内变异）"""
        return self.image_encoder_config

    def set_module_config(self, module_config: dict):
        """设置模块配置（仅用于模块内变异）"""
        self.image_encoder_config = module_config