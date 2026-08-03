import random
import secrets

from templates.template import Template, MMInstance
from modules.pool import TEXT_DECODER_DICT_POOL, AE_DICT_POOL
from utils import (
    log_bullet,
    generate_input_ids_and_position_ids,
    generate_video_tensor_from_ae_config,
)

class VideoModelTemplate(Template):
    def __init__(self):
        super().__init__("video_model")

        # modules for text embedding
        self.text_embedding_decoder_name = None
        self.text_embedding_decoder_config = None

        # modules for ae
        self.ae_name = None
        self.ae_config = None

    def select_modules(self):
        # randomly select a text embedding decoder
        text_embedding_decoder_config = TEXT_DECODER_DICT_POOL.random_choice()
        self.text_embedding_decoder_name = list(text_embedding_decoder_config.keys())[0]
        self.text_embedding_decoder_config = text_embedding_decoder_config[self.text_embedding_decoder_name]
        log_bullet(f"text embedding decoder: {self.text_embedding_decoder_name}")

        # randomly select a ae
        ae_config = AE_DICT_POOL.random_choice()
        self.ae_name = list(ae_config.keys())[0]
        self.ae_config = ae_config[self.ae_name]
        log_bullet(f"ae: {self.ae_name}")

    def apply_config(self, instance_config: dict) -> None:
        self.text_embedding_decoder_name = instance_config["text_embedding_decoder"]["name"]
        self.text_embedding_decoder_config = instance_config["text_embedding_decoder"]["config"]
        self.ae_name = instance_config["ae"]["name"]
        self.ae_config = instance_config["ae"]["config"]

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

        # embdding
        if instance.text_embedding_decoder is not None:
            # Randomize batch size and sequence length to exercise shape handling.
            # Keep lengths small by default to avoid heavy compute.
            input_ids, position_ids, attention_mask = generate_input_ids_and_position_ids(
                batch_size=batch_size,
                vocab_size=instance.text_embedding_decoder.vocab_size,
                max_sequence_length=max_sequence_length,
                is_random=False,
            )
            log_bullet(f"input_ids {input_ids.shape}, position_ids {position_ids.shape}, attention_mask {attention_mask.shape}", indent=2)
            text_embedding = instance.text_embedding_decoder.embedding(input_ids, position_ids, attention_mask)
            log_bullet(f"text_embedding {text_embedding.shape}", indent=2)
        else:
            raise ValueError(f"No available text embedding module")

        # ae encode
        if instance.ae is not None:
            # generate video tensor
            video = generate_video_tensor_from_ae_config(
                batch_size=batch_size,
                num_frames=max_sequence_length,
                ae_config=self.ae_config,
                max_spatial_size=64,
                max_num_frames=8,
                is_random=False,
            )
            log_bullet(f"video {video.shape}", indent=2)
            video_embedding = instance.ae.encode(video)
            log_bullet(f"video_embedding {video_embedding.shape}", indent=2)
        else:
            raise ValueError(f"No available ae module")

        return video_embedding

    def instantiate(self) -> MMInstance:
        config = {
            "ae": self.ae_config,
            "text_encoder": None,
            "diffusion": None,
            "predictor": None
        }

        instance = MMInstance(name="video_model", config=config, template=self)
        text_embedding_decoder = TEXT_DECODER_DICT_POOL.build(self.text_embedding_decoder_name, self.text_embedding_decoder_config)
        instance.set_text_embedding_decoder(text_embedding_decoder, self.text_embedding_decoder_config)
        ae = AE_DICT_POOL.build(self.ae_name, self.ae_config)
        instance.set_ae(ae, self.ae_config)
        return instance

    def dump_graph(self, output_path: str = None) -> str:
        pass