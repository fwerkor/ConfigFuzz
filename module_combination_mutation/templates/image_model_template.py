import os
import random
import secrets
import torch

from templates.template import Template, MMInstance
from modules.pool import TEXT_DECODER_DICT_POOL, IMAGE_ENCODER_DICT_POOL
from utils import generate_input_ids_and_position_ids, generate_images, resolve_path, log_step, log_bullet, log_tensor_summary
from it_combine.strategy import IMAGE_TEXT_COMBINE_STRATEGY_REGISTRY

class ImageModelTemplate(Template):
    def __init__(self):
        super().__init__("image_model")

        # modules for text embedding
        self.text_embedding_decoder_name = None
        self.text_embedding_decoder_config = None

        # modules for image encode
        self.image_encoder_name = None
        self.image_encoder_config = None

        # modules for image text combine
        self.image_text_combine_strategy = None

        # modules for text decode
        self.text_decoders_configs = {}

    def select_modules(self):
        # randomly select a text decoder for embedding
        text_embedding_decoder_config = TEXT_DECODER_DICT_POOL.random_choice()
        self.text_embedding_decoder_name = list(text_embedding_decoder_config.keys())[0]
        self.text_embedding_decoder_config = text_embedding_decoder_config[self.text_embedding_decoder_name]
        log_bullet(f"text embedding decoder: {self.text_embedding_decoder_name}")

        # randomly select a image encoder
        image_encoder_config = IMAGE_ENCODER_DICT_POOL.random_choice()
        self.image_encoder_name = list(image_encoder_config.keys())[0]
        self.image_encoder_config = image_encoder_config[self.image_encoder_name]
        log_bullet(f"image encoder: {self.image_encoder_name}")

        # randomly select a image text combine strategy
        self.image_text_combine_strategy = IMAGE_TEXT_COMBINE_STRATEGY_REGISTRY.random_choice()
        log_bullet(f"image-text combine strategy: {self.image_text_combine_strategy.name}")

        # randomly select several text decoders for decoding
        for _ in range(secrets.randbelow(3) + 1):
            text_decoder_config = TEXT_DECODER_DICT_POOL.random_choice()
            text_decoder_name = list(text_decoder_config.keys())[0]
            self.text_decoders_configs[text_decoder_name] = text_decoder_config[text_decoder_name]
        log_bullet(f"text decoders: {list(self.text_decoders_configs.keys())}")

    def apply_config(self, instance_config: dict) -> None:
        """从实例配置（如 round_*.json）填充模板，用于基于配置目录的跑测。"""
        self.text_embedding_decoder_name = instance_config["text_embedding_decoder"]["name"]
        self.text_embedding_decoder_config = instance_config["text_embedding_decoder"]["config"]
        self.image_encoder_name = instance_config["image_encoder"]["name"]
        self.image_encoder_config = instance_config["image_encoder"]["config"]
        self.image_text_combine_strategy = IMAGE_TEXT_COMBINE_STRATEGY_REGISTRY.get(
            instance_config["image_text_combine_strategy"]
        )
        self.text_decoders_configs = {
            item["name"]: item["config"] for item in instance_config["text_decoders"]
        }

    def _resolve_text_hidden_size(self, text_embedding_decoder) -> int:
        cfg = getattr(text_embedding_decoder, "config", None)
        if cfg is not None and hasattr(cfg, "hidden_size"):
            return int(getattr(cfg, "hidden_size"))
        model_cfg = getattr(text_embedding_decoder, "model_config", None)
        td_cfg = getattr(model_cfg, "text_decoder", None) if model_cfg is not None else None
        if td_cfg is not None and hasattr(td_cfg, "hidden_size"):
            return int(getattr(td_cfg, "hidden_size"))
        return 32

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
            log_tensor_summary("input_ids", input_ids, indent=3)
            log_tensor_summary("position_ids", position_ids, indent=3)
            log_tensor_summary("attention_mask", attention_mask, indent=3)
            # 跳过真实 embedding，直接构造固定 text_embedding 以排除 embedding 侧差异。
            hidden_size = self._resolve_text_hidden_size(instance.text_embedding_decoder)
            total = max_sequence_length * batch_size * hidden_size
            fixed = torch.arange(total, device=position_ids.device, dtype=torch.float32)
            fixed = (fixed % 1024) / 511.5 - 1.0
            text_embedding = fixed.reshape(max_sequence_length, batch_size, hidden_size).to(dtype=torch.bfloat16)
            log_bullet("text_embedding source: FIXED_TENSOR (skip decoder.embedding)", indent=2)
            log_bullet(f"text_embedding {text_embedding.shape}", indent=2)
            log_tensor_summary("text_embedding", text_embedding, indent=3)
        else:
            raise ValueError(f"No available text embedding module")

        # image encode
        image_size = instance.image_encoder.image_size
        # Keep image input dtype consistent with vision encoder params_dtype.
        # MindSpore Ascend PTA is strict about MatMul/Conv dtype matching (x1/x2 must be same).
        dtype = torch.bfloat16
        try:
            ve_cfg = getattr(instance.image_encoder, "config", {}).get("vision_encoder", {})
            params_dtype = ve_cfg.get("params_dtype", None)
            if isinstance(params_dtype, torch.dtype):
                dtype = params_dtype
            elif isinstance(params_dtype, str):
                dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}.get(params_dtype, torch.bfloat16)
            else:
                # Fallback: use boolean bf16 flag if present
                if ve_cfg.get("bf16", False) is True:
                    dtype = torch.bfloat16
                elif ve_cfg.get("fp16", False) is True:
                    dtype = torch.float16
                else:
                    dtype = torch.float32
        except Exception:
            # If config structure is unexpected, keep default bf16 to match module defaults.
            dtype = torch.bfloat16
        images = generate_images(batch_size, image_size, dtype=dtype, is_random=False)
        log_bullet(f"images {images.shape}", indent=2)
        log_tensor_summary("images", images, indent=3)
        image_features = instance.image_encoder.encode(images)
        log_bullet(f"image_features {image_features.shape}", indent=2)
        log_tensor_summary("image_features", image_features, indent=3)

        # image text combine
        combined_embeddings = instance.image_text_combine_strategy.combine(image_features, text_embedding, input_ids, position_ids, attention_mask)
        log_bullet(f"combined_embeddings {combined_embeddings.shape}", indent=2)
        log_tensor_summary("combined_embeddings", combined_embeddings, indent=3)

        # text decode
        decoder_input = combined_embeddings
        for text_decoder in instance.text_decoders:
            log_bullet(f"decoder '{text_decoder.name}' 前向中...", indent=2)
            decoder_input = text_decoder.decode(
                input=decoder_input, attention_mask=attention_mask, batch_first=True,
                input_ids=input_ids, position_ids=position_ids
            )
            log_bullet(f"decoder '{text_decoder.name}' -> {decoder_input.shape}", indent=2)
            log_tensor_summary(f"decoder '{text_decoder.name}' output", decoder_input, indent=3)

        return decoder_input

    def instantiate(self) -> MMInstance:
        config = {
            "text_decoder": self.text_decoders_configs,
            "image_encoder": self.image_encoder_config,
            "text_encoder": None,
            "video_encoder": None
        }

        instance = MMInstance(name="image_model", config=config, template=self)
        text_embedding_decoder = TEXT_DECODER_DICT_POOL.build(self.text_embedding_decoder_name, self.text_embedding_decoder_config)
        instance.set_text_embedding_decoder(text_embedding_decoder, self.text_embedding_decoder_config)
        image_encoder = IMAGE_ENCODER_DICT_POOL.build(self.image_encoder_name, self.image_encoder_config)
        instance.set_image_encoder(image_encoder, self.image_encoder_config)
        instance.set_image_text_combine_strategy(self.image_text_combine_strategy)
        for text_decoder_name, text_decoder_config in self.text_decoders_configs.items():
            text_decoder = TEXT_DECODER_DICT_POOL.build(text_decoder_name, text_decoder_config)
            instance.add_text_decoder(text_decoder, text_decoder_config)

        return instance

    def dump_graph(self, output_path: str = None) -> str:
        """生成并保存 ImageModel 计算图的 dot 文件。"""
        if output_path is None:
            output_path = resolve_path("graph.dot")

        lines = [
            "digraph ImageModel {",
            "    rankdir=LR;",
            "    node [shape=box];",
            "",
            "    /* 输入 */",
            '    input_ids [label="input_ids"];',
            '    position_ids [label="position_ids"];',
            '    attention_mask [label="attention_mask"];',
            '    images [label="images"];',
            "",
            "    /* 文本嵌入 */",
            f'    text_embedding [label="embedding\\n({self.text_embedding_decoder_name})"];',
            '    input_ids -> text_embedding;',
            '    position_ids -> text_embedding;',
            '    attention_mask -> text_embedding;',
            "",
            "    /* 图像编码 */",
            f'    image_features [label="encode\\n({self.image_encoder_name})"];',
            '    images -> image_features;',
            "",
            "    /* 图文融合 */",
            f'    combined [label="combine\\n({self.image_text_combine_strategy.name})"];',
            '    text_embedding -> combined;',
            '    image_features -> combined;',
            "",
            "    /* 文本解码链（接在 combine 之后） */",
        ]
        # 用安全 ID 避免名字中的点号等破坏 DOT 解析，解码链顺序：combined -> decoder_0 -> decoder_1 -> ... -> output
        prev = "combined"
        for i, (name, config) in enumerate(self.text_decoders_configs.items()):
            node_id = f"decoder_{i}"
            lines.append(f'    {node_id} [label="decode\\n({name})"];')
            lines.append(f"    {prev} -> {node_id};")
            prev = node_id
        lines.append(f'    output [label="output"];')
        lines.append(f"    {prev} -> output;")
        lines.append("}")

        # 确保输出目录存在并写入 dot 文件
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(output_path, "w") as f:
            f.write("\n".join(lines))

        return output_path
