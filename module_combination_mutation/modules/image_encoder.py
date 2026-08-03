import torch
import secrets

from megatron.training import get_args
from megatron.core import mpu

from utils import log_step, log_bullet, get_device, log_tensor_summary
from mindspeed_mm.models.vision.vision_model import VisionModel
from mindspeed_mm.configs.config import ConfigReader
from mindspeed_mm.utils.transformer_model_config import get_model_config
from mindspeed_mm.utils.utils import EncoderBalanceComm

# llava
from mindspeed_mm.models.common.module_spec.llava_layer_spec import get_layer_spec, get_mlp_module_spec

# internvl
from mindspeed_mm.models.common.module_spec.internvl_layer_spec import get_language_layer_spec, get_vit_layer_spec

def register_image_encoder(*names: str):
    """装饰器：将 image encoder 类注册到构建器表，扩展新 encoder 时只需加此装饰器。"""
    def decorator(cls):
        for name in names:
            IMAGE_ENCODER_BUILDERS[name] = cls
        return cls
    return decorator


# name -> EncoderClass，各 encoder 通过 @register_image_encoder 自行注册
IMAGE_ENCODER_BUILDERS = {}


class ImageEncoder:
    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config
        self.image_size = config["vision_encoder"]["image_size"]
        self.encoder = None

    def _build(self):
        pass

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        pass

    def _resolve_target_dtype(self) -> torch.dtype:
        """
        Resolve the expected compute dtype for the vision encoder.
        We intentionally align the *entire* encoder module params/bias with the dtype used
        for `images` in `ImageModelTemplate.forward()` to avoid conv dtype mismatches.
        """
        # Most configs in this repo put dtype under `vision_encoder`.
        ve_cfg = (self.config or {}).get("vision_encoder", {}) or {}
        params_dtype = ve_cfg.get("params_dtype", None)

        if isinstance(params_dtype, torch.dtype):
            return params_dtype
        if isinstance(params_dtype, str):
            return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}.get(
                params_dtype, torch.bfloat16
            )

        # Fallbacks used by some configs.
        if ve_cfg.get("bf16", False) is True:
            return torch.bfloat16
        if ve_cfg.get("fp16", False) is True:
            return torch.float16
        if ve_cfg.get("fp32", False) is True:
            return torch.float32

        # Default to bf16 because the template currently generates `images` in bf16 by default
        # (unless `params_dtype` is explicitly provided).
        return torch.bfloat16

@register_image_encoder("llava")
class LLavaImageEncoder(ImageEncoder):
    @classmethod
    def build(cls, name: str, config: dict) -> "ImageEncoder":
        log_step("Build", f"image encoder: {name}", indent=1)
        return cls(name, config)

    def __init__(self, name: str, config: dict):
        super().__init__(name, config)
        self.IGNORE_INDEX = -100
        self.IMAGE_TOKEN_INDEX = -200
        self.model_config = ConfigReader({"image_encoder": config})
        self.model_config.image_encoder.vision_encoder = get_model_config(self.model_config.image_encoder.vision_encoder)
        self.model_config.image_encoder.vision_encoder.vision_transformer_layer_spec = get_layer_spec(is_vit=True)
        self.model_config.image_encoder.vision_projector = get_model_config(self.model_config.image_encoder.vision_projector)
        self.model_config.image_encoder.vision_projector.vision_projection_layer_spec = get_mlp_module_spec(use_te=False).submodules

        self._build()


    def _build(self):
        self.encoder = VisionModel(
            self.model_config.image_encoder,
            self.model_config.image_encoder.vision_encoder.vision_transformer_layer_spec,
            self.model_config.image_encoder.vision_projector.vision_projection_layer_spec
        )
        try:
            device = get_device()
            dtype = self._resolve_target_dtype()
            # MindSpore/PTA on Ascend is strict about dtype matching for Conv/MatMul inputs.
            self.encoder.to(device=device, dtype=dtype)
        except Exception:
            log_step("Warning", f"Failed to move the encoder to {device}", indent=1)

    def encode(self, images: torch.Tensor) -> torch.Tensor:

        # new_input_embeds, attention_mask = self.prepare_inputs_labels_for_multimodal(
        #     input_ids,
        #     position_ids,
        #     attention_mask,
        #     images,
        #     text_embeddings
        # )
        log_tensor_summary(f"{self.name}.encode.images_in", images, indent=2)
        image_features = self.encoder(images)
        log_tensor_summary(f"{self.name}.encode.vision_model_out", image_features, indent=2)
        return image_features

    def prepare_inputs_labels_for_multimodal(
            self,
            input_ids,
            position_ids,
            attention_mask,
            images,
            text_embeddings
    ):
        labels = torch.full_like(input_ids, self.IGNORE_INDEX)
        image_features = self.encoder(images)
        log_bullet(f"image_features shape: {image_features.shape}", indent=2)
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)

        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask]
                     for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)
                     ]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == self.IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                # text_embeddings 为 (seq_len, batch_size, hidden)，取第 batch_idx 个 batch 再按 mask 过滤
                cur_input_embeds_1 = text_embeddings[:, batch_idx, :][attention_mask[batch_idx]]
                cur_input_embeds = cur_input_embeds_1
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == self.IMAGE_TOKEN_INDEX)[0].tolist() + [
                cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1:image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1:image_token_indices[i + 1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = text_embeddings[:, batch_idx, :][attention_mask[batch_idx]]
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(
                        torch.full((cur_image_features.shape[0],), self.IGNORE_INDEX, device=cur_labels.device,
                                   dtype=cur_labels.dtype))

            cur_new_input_embeds = [x for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        # tokenizer_model_max_length = getattr(self.config, 'language_max_sequence_length', None)
        # if tokenizer_model_max_length is not None:
        #     new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
        #     new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), self.IGNORE_INDEX, dtype=new_labels[0].dtype,
                                       device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype,
                                                              device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype,
                                                             device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        causal_attention_mask = torch.triu(
            torch.ones(new_input_embeds.shape[0], 1, new_input_embeds.shape[1], new_input_embeds.shape[1],
                       device=new_input_embeds.device),
            diagonal=1
        ).bool()
        attention_mask = ~attention_mask
        expanded_attention_mask = attention_mask[:, None, None, :].expand(
            new_input_embeds.shape[0], 1, new_input_embeds.shape[1], new_input_embeds.shape[1]
        )
        # 用逻辑或替代 masked_fill(..., True)，避免 NPU 上 masked_fill 布尔填充报错
        attention_mask = causal_attention_mask | expanded_attention_mask

        return new_input_embeds, attention_mask

@register_image_encoder("internvl2.5")
class InternVLImageEncoder(ImageEncoder):
    @classmethod
    def build(cls, name: str, config: dict) -> "ImageEncoder":
        log_step("Build", f"image encoder: {name}", indent=1)
        return cls(name, config)

    def __init__(self, name: str, config: dict):
        super().__init__(name, config)
        self.IGNORE_INDEX = -100
        self.IMAGE_TOKEN_INDEX = -200
        self.model_config = ConfigReader({"image_encoder": config})
        self.model_config.image_encoder.vision_encoder = get_model_config(self.model_config.image_encoder.vision_encoder)

        self._build()

    def _build(self):
        transformer_layer_spec = get_vit_layer_spec(self.model_config.image_encoder.vision_encoder)
        self.encoder = VisionModel(
            config = self.model_config.image_encoder,
            encoder_transformer_layer_spec = transformer_layer_spec
        )
        try:
            device = get_device()
            dtype = self._resolve_target_dtype()
            self.encoder.to(device=device, dtype=dtype)
        except Exception:
            log_step("Warning", f"Failed to move the encoder to {device}", indent=1)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        log_tensor_summary(f"{self.name}.encode.images_in", images, indent=2)
        vit_embeds = self.encoder(images)
        log_tensor_summary(f"{self.name}.encode.vit_after_vision_model", vit_embeds, indent=2)

        args = get_args()
        encoder_dp_balance = bool(getattr(args, "encoder_dp_balance", False))
        dist_train = bool(getattr(args, "dist_train", False))
        log_bullet(
            f"{self.name}.encode.runtime: encoder_dp_balance={encoder_dp_balance} "
            f"dist_train={dist_train} world_size="
            f"{(torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1)}",
            indent=2,
        )

        # Some configs/runs may not define `encoder_dp_balance` on args; keep it optional.
        if encoder_dp_balance:
            vit_embeds = EncoderBalanceComm.apply(
                vit_embeds,
                mpu.get_data_parallel_group(),
                None,
            )
            log_tensor_summary(f"{self.name}.encode.vit_after_encoder_balance_comm", vit_embeds, indent=2)

        # `internvl_model.py` filters vit_embeds using `image_flags`, but in this
        # random-shape harness we don't have dataset-provided flags, so default
        # to "all images are valid".
        image_flags = torch.ones((vit_embeds.shape[0], 1), dtype=torch.long, device=vit_embeds.device)

        if dist_train:
            from mindspeed.core.multi_modal.dist_train.inner_data_parallel import gather_from_inner_dp_region
            from mindspeed.core.multi_modal.dist_train.utils import need_inner_data_parallel

            need_inner = need_inner_data_parallel()
            log_bullet(f"{self.name}.encode.dist_train: need_inner_data_parallel={need_inner}", indent=2)
            if need_inner:
                vit_embeds = gather_from_inner_dp_region(
                    vit_embeds,
                    inner_dp_parallel_output_grad=False,
                )
                log_tensor_summary(f"{self.name}.encode.vit_after_inner_dp_gather", vit_embeds, indent=2)
            vit_embeds = vit_embeds[: image_flags.shape[0]]
            log_tensor_summary(f"{self.name}.encode.vit_after_flag_len_slice", vit_embeds, indent=2)

        image_flags = image_flags.squeeze(-1)
        vit_embeds = vit_embeds[image_flags == 1].reshape(1, -1, vit_embeds.shape[-1]).clone()
        log_tensor_summary(f"{self.name}.encode.vit_after_flags_reshape", vit_embeds, indent=2)

        output = vit_embeds.contiguous()
        log_tensor_summary(f"{self.name}.encode.output", output, indent=2)
        return output