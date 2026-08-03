import secrets
import torch.nn.functional as F

from copy import deepcopy
from utils import log_step
from modules.text_decoder import TextDecoder, TEXT_DECODER_BUILDERS
from modules.image_encoder import ImageEncoder
from modules.image_encoder import IMAGE_ENCODER_BUILDERS
from modules.ae import AE, AE_BUILDERS

class TextDecoderDictPool:
    def __init__(self):
        self.text_decoders = {}

    def register_all(self, text_decoders: dict):
        self.text_decoders = text_decoders

    def register_one(self, name: str, config: dict):
        self.text_decoders[name] = config

    def get(self, name: str) -> dict:
        if name not in self.text_decoders:
            raise ValueError(f"Text decoder {name} not found")
        return {name: self.text_decoders[name]}
    
    def random_choice(self) -> dict:
        if not self.text_decoders:
            raise ValueError("No text decoders registered in TEXT_DECODER_REGISTRY")
        name = secrets.choice(list(self.text_decoders.keys()))
        return {name: self.text_decoders[name]}

    def build(self, name: str, config: dict) -> TextDecoder:
        _config = deepcopy(config)
        if _config.get("activation_func") == "silu":
            _config["activation_func"] = F.silu
        elif _config.get("activation_func") == "quick_gelu":
            from mindspeed_mm.utils.utils import quick_gelu
            _config["activation_func"] = quick_gelu
        else:
            _config["activation_func"] = F.gelu
        if name not in TEXT_DECODER_BUILDERS:
            raise ValueError(f"Invalid text decoder: {name}. Available: {list(TEXT_DECODER_BUILDERS.keys())}")
        DecoderClass = TEXT_DECODER_BUILDERS[name]
        return DecoderClass.build(name, _config)

class ImageEncoderDictPool:
    def __init__(self):
        self.image_encoders = {}

    def register_all(self, image_encoders: dict):
        self.image_encoders = image_encoders

    def register_one(self, name: str, config: dict):
        self.image_encoders[name] = config

    def get(self, name: str) -> dict:
        if name not in self.image_encoders:
            raise ValueError(f"Image encoder {name} not found")
        return {name: self.image_encoders[name]}
        
    def random_choice(self) -> dict:
        if not self.image_encoders:
            raise ValueError("No image encoders registered in IMAGE_ENCODER_REGISTRY")
        name = secrets.choice(list(self.image_encoders.keys()))
        return {name: self.image_encoders[name]}

    def build(self, name: str, config: dict) -> ImageEncoder:
        _config = deepcopy(config)
        if name not in IMAGE_ENCODER_BUILDERS:
            raise ValueError(
                f"Invalid image encoder: {name}. Available: {list(IMAGE_ENCODER_BUILDERS.keys())}"
            )
        EncoderClass = IMAGE_ENCODER_BUILDERS[name]
        return EncoderClass.build(name, _config)

class AEDictPool:
    def __init__(self):
        self.aes = {}

    def register_all(self, aes: dict):
        self.aes = aes
        
        
    def register_one(self, name: str, config: dict):
        self.aes[name] = config

    def get(self, name: str) -> dict:
        if name not in self.aes:
            raise ValueError(f"AE {name} not found")
        return {name: self.aes[name]}

    def random_choice(self) -> dict:
        if not self.aes:
            raise ValueError("No AEs registered in AE_REGISTRY")
        name = secrets.choice(list(self.aes.keys()))
        return {name: self.aes[name]}

    def build(self, name: str, config: dict) -> AE:
        _config = deepcopy(config)
        if name not in AE_BUILDERS:
            raise ValueError(f"Invalid AE: {name}. Available: {list(AE_BUILDERS.keys())}")
        AEClass = AE_BUILDERS[name]
        return AEClass.build(name, _config)

TEXT_DECODER_DICT_POOL = TextDecoderDictPool()
IMAGE_ENCODER_DICT_POOL = ImageEncoderDictPool()
AE_DICT_POOL = AEDictPool()