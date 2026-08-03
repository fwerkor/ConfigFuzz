import random
import secrets

from templates.template import Template, MMInstance
from modules.pool import TEXT_DECODER_DICT_POOL
from utils import log_bullet, generate_input_ids_and_position_ids
from common import SupportedModules

class SingleTextDecoderTemplate(Template):
    def __init__(self):
        super().__init__("single_text_decoder")

        # the single module
        self.text_decoder_name = None
        self.text_decoder_config = None

    def select_modules(self):
        # randomly select a text decoder
        text_decoder_config = TEXT_DECODER_DICT_POOL.random_choice()
        self.text_decoder_name = list(text_decoder_config.keys())[0]
        self.text_decoder_config = text_decoder_config[self.text_decoder_name]
        log_bullet(f"text decoder: {self.text_decoder_name}")

    def instantiate(self) -> MMInstance:
        config = {
            "text_decoder": self.text_decoder_config,
            "image_encoder": None,
            "text_encoder": None,
            "video_encoder": None
        }

        instance = MMInstance(name="single_text_decoder", config=config, template=self)
        text_decoder = TEXT_DECODER_DICT_POOL.build(self.text_decoder_name, self.text_decoder_config)
        instance.add_text_decoder(text_decoder, self.text_decoder_config)
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
        max_sequence_length = 16

        decoder = instance.text_decoders[0]

        # embdding
        vocab_size = self.text_decoder_config.get("vocab_size", 5000)
        input_ids, position_ids, attention_mask = generate_input_ids_and_position_ids(
            batch_size=batch_size,
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            is_random=False,
        )
        log_bullet(f"input_ids {input_ids.shape}, position_ids {position_ids.shape}, attention_mask {attention_mask.shape}", indent=2)
        text_embedding = decoder.embedding(input_ids, position_ids, attention_mask)
        log_bullet(f"text_embedding {text_embedding.shape}", indent=2)

        # text decode
        decoder_input = text_embedding
        decoder_input = decoder.decode(
            input=decoder_input, attention_mask=attention_mask, batch_first=True,
            input_ids=input_ids, position_ids=position_ids
        )
        log_bullet(f"decoder '{decoder.name}' -> {decoder_input.shape}", indent=2)
        
        return decoder_input

    def get_module_type(self) -> SupportedModules:
        """获取模块类型（仅用于模块内变异）"""
        return SupportedModules.TEXT_DECODER

    def get_module_name(self) -> str:
        """获取模块名称（仅用于模块内变异）"""
        return self.text_decoder_name

    def set_module_name(self, module_name: str):
        """设置模块名称（仅用于模块内变异）"""
        self.text_decoder_name = module_name

    def get_module_config(self) -> dict:
        """获取模块配置（仅用于模块内变异）"""
        return self.text_decoder_config

    def set_module_config(self, module_config: dict):
        """设置模块配置（仅用于模块内变异）"""
        self.text_decoder_config = module_config