import torch
import secrets
from typing import Optional

from megatron.core import mpu
from utils import create_decoder_attention_mask, ensure_decoder_input_shape, log_step, get_device, log_tensor_summary
from megatron.core.transformer import TransformerConfig
from megatron.core.models.gpt import GPTModel

from mindspeed_mm.models.text_decoder.moe_model import MOEModel
from mindspeed_mm.models.common.module_spec.get_layer_spec import get_llm_layer_spec
from mindspeed_mm.configs.config import ConfigReader
from mindspeed_mm.utils.transformer_model_config import get_model_config

# llava
from mindspeed_mm.models.common.module_spec.llava_layer_spec import get_layer_spec

# internvl
from mindspeed_mm.models.common.module_spec.internvl_layer_spec import get_language_layer_spec


def register_text_decoder(*names: str):
    """装饰器：将 text decoder 类注册到构建器表，扩展新 decoder 时只需加此装饰器，无需修改 pool.py"""
    def decorator(cls):
        for name in names:
            TEXT_DECODER_BUILDERS[name] = cls
        return cls
    return decorator


# name -> DecoderClass，各 decoder 通过 @register_text_decoder 自行注册
TEXT_DECODER_BUILDERS = {}

def _deterministic_token_ids(vocab_size: int, batch_size: int, seq_len: int, *, device) -> torch.Tensor:
    """Build deterministic token ids shared by PTA/MSA."""
    total = batch_size * seq_len
    return (
        torch.arange(total, dtype=torch.long, device=device)
        .reshape(batch_size, seq_len)
        % int(vocab_size)
    )

def _build_token_ids(vocab_size: int, batch_size: int, seq_len: int, *, device, context: str) -> torch.Tensor:
    """Always use deterministic fallback to keep PTA/MSA aligned."""
    log_step(
        "DETERMINISTIC_FALLBACK",
        f"token_ids(vocab={vocab_size}, shape=({batch_size}, {seq_len}), device={device}, context={context})",
        indent=1,
    )
    return _deterministic_token_ids(vocab_size, batch_size, seq_len, device=device)


class TextDecoder:
    def __init__(self, name: str, config, pre_process: bool = True, post_process: bool = False):
        self.name = name
        self.config = config
        self.decoder = None
        self.pre_process = pre_process
        self.post_process = post_process

        self.vocab_size = None

    def _build(self):
        pass

    def embedding(self, input_ids: torch.Tensor, position_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        pass

    def decode(self, input: torch.Tensor, attention_mask: torch.Tensor, batch_first: bool = False,
               input_ids: torch.Tensor = None, position_ids: torch.Tensor = None, labels: torch.Tensor = None) -> torch.Tensor:
        pass

@register_text_decoder("deepseekv3_vl", "deepseekvl2")
class DeepSeekVLTextDecoder(TextDecoder):
    @classmethod
    def build(cls, name: str, config: dict) -> "TextDecoder":
        text_decoder_config = TransformerConfig(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            **config
        )
        log_step("Build", f"text decoder: {name}", indent=1)
        return cls(name, text_decoder_config, pre_process=True, post_process=True)

    def __init__(self, name: str, config: TransformerConfig, pre_process: bool = True, post_process: bool = False):
        super().__init__(name, config, pre_process, post_process)

        # set the vocab size from config's language_vocab_size or vocab_size (TransformerConfig uses getattr)
        self.vocab_size = getattr(config, "vocab_size", None) or getattr(config, "language_vocab_size", 32000)

        self.share_embeddings_and_output_weights = not getattr(config, 'untie_embeddings_and_output_weights', True)
        self.pp_size = mpu.get_pipeline_model_parallel_world_size()
        if mpu.get_virtual_pipeline_model_parallel_world_size() is not None:
            raise NotImplementedError("Not support virtual_pipeline_model_parallel now")
        else:
            self.pp_rank = mpu.get_pipeline_model_parallel_rank()

        self._build()

    def _build(self):
        if self.pp_size <= 1:
            self.decoder = MOEModel(
                config=self.config,
                transformer_layer_spec=get_llm_layer_spec(self.config),
                vocab_size=self.config.vocab_size,
                max_sequence_length=self.config.max_position_embeddings,
                parallel_output=self.config.parallel_output,
                position_embedding_type=self.config.position_embedding_type,
                share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
                rotary_base=self.config.rope_theta if getattr(self.config, 'rope_theta', None) else self.config.rotary_base,
                pre_process=self.pre_process,
                post_process=self.post_process
            )
            try:
                device = get_device()
                self.decoder.to(device)
            except Exception:
                log_step("Warning", "Failed to move the decoder to current device", indent=1)
            return
        
        if self.pp_size != len(self.config.pipeline_num_layers):
            raise ValueError(f"length of pipeline_num_layers must equal to pipeline-model-parallel-size, "
                             f"but got pipeline_num_layers length:{len(self.config.pipeline_num_layers)} "
                             f"and pipeline-model-parallel-size:{self.pp_size}.")

        local_num_layers = self.config.pipeline_num_layers[self.pp_rank]
        if local_num_layers == 0:
            self.add_text_decoder = False
            return None

        pipeline_start_index = sum(self.config.pipeline_num_layers[:self.pp_rank])
        pipeline_end_index = sum(self.config.pipeline_num_layers[:self.pp_rank + 1])
        
        pre_process = pipeline_start_index == 0
        post_process = pipeline_end_index == self.config.num_layers
        first_k_dense_replace = self.config.first_k_dense_replace - pipeline_start_index

        # print(
        #     f"text decoder pipeline config:\
        #     pp_rank:{self.pp_rank},\
        #     pre_process:{pre_process},\
        #     post_process:{post_process},\
        #     local_num_layers:{local_num_layers}"
        # )
        # num_layers will be divided by pp_size in TransformerBlock from megatron.core
        self.config.num_layers = self.pp_size * local_num_layers
        self.config.first_k_dense_replace = first_k_dense_replace

        self.decoder = MOEModel(
                config=self.config,
                transformer_layer_spec=get_llm_layer_spec(self.config),
                vocab_size=self.config.vocab_size,
                max_sequence_length=self.config.max_position_embeddings,
                parallel_output=self.config.parallel_output,
                position_embedding_type=self.config.position_embedding_type,
                share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
                rotary_base=self.config.rope_theta if getattr(self.config, 'rope_theta', None) else self.config.rotary_base,
                pre_process=pre_process,
                post_process=post_process
            )
        try:
            device = get_device()
            self.decoder.to(device)
        except Exception:
            log_step("Warning", "Failed to move the decoder to current device", indent=1)
        
    def embedding(self, input_ids: torch.Tensor, position_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.decoder.embedding(input_ids, position_ids)

    def decode(self, input: torch.Tensor, attention_mask: torch.Tensor, batch_first: bool = False,
               input_ids: torch.Tensor = None, position_ids: torch.Tensor = None, labels: torch.Tensor = None) -> torch.Tensor:
        input = ensure_decoder_input_shape(
            input, self.config.hidden_size, self.decoder, self.share_embeddings_and_output_weights
        )
        log_tensor_summary(f"{self.name}.decode.input_aligned", input, indent=2)
        # MOEModel 期望 decoder_input 为 (seq_len, batch, hidden)。
        # 这里的 input 可能来自不同模块/策略，维度顺序不总是可靠；优先用 attention_mask 的 batch 维来纠正。
        if attention_mask is not None and attention_mask.dim() == 2:
            mask_bsz = attention_mask.shape[0]
            if input.dim() >= 2:
                if batch_first and input.shape[0] != mask_bsz and input.shape[1] == mask_bsz:
                    batch_first = False
                elif (not batch_first) and input.shape[1] != mask_bsz and input.shape[0] == mask_bsz:
                    batch_first = True

        if batch_first:
            # input: (batch, seq, hidden) -> (seq, batch, hidden)
            decoder_input = input.transpose(0, 1)
            seq_len = input.shape[1]
            batch_size = input.shape[0]
        else:
            # input: (seq, batch, hidden)
            decoder_input = input
            seq_len = input.shape[0]
            batch_size = input.shape[1]
        device = input.device
        log_tensor_summary(f"{self.name}.decode.decoder_input", decoder_input, indent=2)
        # NPU FlashAttention 需要 4D attention_mask [B, 1, Sq, Skv]，将 2D [B, seq] 转为 4D
        valid_mask_2d = None
        if attention_mask is not None and attention_mask.dim() == 2:
            if attention_mask.shape[1] != seq_len:
                pad_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=attention_mask.device)
                min_len = min(attention_mask.shape[1], seq_len)
                pad_mask[:, :min_len] = attention_mask[:, :min_len]
                attention_mask = pad_mask
            valid_mask_2d = attention_mask  # True = valid
            attention_mask = create_decoder_attention_mask(attention_mask, seq_len, causal=True)
        if attention_mask is not None:
            log_tensor_summary(f"{self.name}.decode.attention_mask_4d", attention_mask, indent=2)
        # 由调用方传入 input_ids、position_ids、labels，为空或 shape 不匹配时在 decode 内构造
        if input_ids is None or input_ids.shape != (batch_size, seq_len):
            import os
            if os.environ.get("LMSV_MM_MSARUN") == "1":
                input_ids = _build_token_ids(
                    self.config.vocab_size,
                    batch_size,
                    seq_len,
                    device="Ascend",
                    context=f"{self.name}.decode fallback input_ids (MSA)",
                )
            elif os.environ.get("LMSV_MM_PTARUN") == "1":
                input_ids = _build_token_ids(
                    self.config.vocab_size,
                    batch_size,
                    seq_len,
                    device=str(device),
                    context=f"{self.name}.decode fallback input_ids (PTA)",
                )
            else:
                raise ValueError("LMSV_MM_MSARUN or LMSV_MM_PTARUN must be set")
        if position_ids is None or position_ids.shape != (batch_size, seq_len):
            position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
        log_tensor_summary(f"{self.name}.decode.input_ids", input_ids, indent=2)
        log_tensor_summary(f"{self.name}.decode.position_ids", position_ids, indent=2)
        if labels is None:
            # print(f"device: {str(device)}")
            # print(f"vocab_size: {self.config.vocab_size}")
            # print(f"batch_size: {batch_size}")
            # print(f"seq_len: {seq_len}")
            # print(f"dtype: {torch.long}")
            import os
            if os.environ.get("LMSV_MM_MSARUN") == "1":
                labels = _build_token_ids(
                    self.config.vocab_size,
                    batch_size,
                    seq_len,
                    device="Ascend",
                    context=f"{self.name}.decode fallback labels (MSA)",
                )
            elif os.environ.get("LMSV_MM_PTARUN") == "1":
                labels = _build_token_ids(
                    self.config.vocab_size,
                    batch_size,
                    seq_len,
                    device=str(device),
                    context=f"{self.name}.decode fallback labels (PTA)",
                )
            else:
                raise ValueError("LMSV_MM_MSARUN or LMSV_MM_PTARUN must be set")
            if valid_mask_2d is not None:
                ignore_fill = torch.full_like(labels, -100)
                labels = torch.where(valid_mask_2d, labels, ignore_fill)
        if labels is not None:
            log_tensor_summary(f"{self.name}.decode.labels", labels, indent=2)
        output = self.decoder(
            input_ids = input_ids,
            position_ids = position_ids,
            attention_mask = attention_mask,
            decoder_input = decoder_input,
            labels = labels
        )
        final_output = output[0] if isinstance(output, tuple) else output
        log_tensor_summary(f"{self.name}.decode.output", final_output, indent=2)
        return final_output

@register_text_decoder("llava1.5")
class LLavaTextDecoder(TextDecoder):
    @classmethod
    def build(cls, name: str, config: dict) -> "TextDecoder":
        log_step("Build", f"text decoder: {name}", indent=1)
        return cls(name, config, pre_process=True, post_process=True)

    def __init__(self, name: str, config: dict, pre_process: bool = True, post_process: bool = False):
        super().__init__(name, config, pre_process, post_process)
        self.model_config = ConfigReader({"text_decoder": config})
        self.model_config.text_decoder = get_model_config(self.model_config.text_decoder)
        self.model_config.text_decoder.language_transformer_layer_spec = get_layer_spec(is_vit=False)

        # set the vocab size from config's language_vocab_size or vocab_size
        self.vocab_size = config.get("language_vocab_size", config.get("vocab_size", 32000))

        self._build()

    def _build(self):
        self.decoder = GPTModel(
            config = self.model_config.text_decoder,
            transformer_layer_spec = self.model_config.text_decoder.language_transformer_layer_spec,
            vocab_size = self.model_config.text_decoder.language_vocab_size,
            max_sequence_length = self.model_config.text_decoder.language_max_sequence_length,
            position_embedding_type = self.model_config.text_decoder.lm_position_embedding_type
            )
        
        try:
            device = get_device()
            self.decoder.to(device)
        except Exception:
            log_step("Warning", "Failed to move the decoder to current device", indent=1)

    def embedding(self, input_ids: torch.Tensor, position_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.decoder.embedding(input_ids, position_ids)

    def decode(self, input: torch.Tensor, attention_mask: torch.Tensor, batch_first: bool = False,
               input_ids: torch.Tensor = None, position_ids: torch.Tensor = None, labels: torch.Tensor = None) -> torch.Tensor:
        input = ensure_decoder_input_shape(
            input, self.model_config.text_decoder.hidden_size, self.decoder,
            getattr(self.decoder, 'share_embeddings_and_output_weights', False)
        )
        log_tensor_summary(f"{self.name}.decode.input_aligned", input, indent=2)
        log_tensor_summary(f"{self.name}.decode.input_aligned", input, indent=2)
        # decoder expects (seq_len, batch, hidden)。
        # input 的维度顺序不一定可靠，优先用 attention_mask 的 batch 维做纠正。
        if attention_mask is not None and attention_mask.dim() == 2:
            mask_bsz = attention_mask.shape[0]
            if input.dim() >= 2:
                if batch_first and input.shape[0] != mask_bsz and input.shape[1] == mask_bsz:
                    batch_first = False
                elif (not batch_first) and input.shape[1] != mask_bsz and input.shape[0] == mask_bsz:
                    batch_first = True

        if batch_first:
            decoder_input = input.transpose(0, 1)
            seq_len = input.shape[1]
            batch_size = input.shape[0]
        else:
            decoder_input = input
            seq_len = input.shape[0]
            batch_size = input.shape[1]
        log_tensor_summary(f"{self.name}.decode.decoder_input", decoder_input, indent=2)
        # NPU FlashAttention 需要 4D attention_mask [B, 1, Sq, Skv]，将 2D [B, seq] 转为 4D
        if attention_mask is not None and attention_mask.dim() == 2:
            if attention_mask.shape[1] != seq_len:
                pad_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=attention_mask.device)
                min_len = min(attention_mask.shape[1], seq_len)
                pad_mask[:, :min_len] = attention_mask[:, :min_len]
                attention_mask = pad_mask
            attention_mask = create_decoder_attention_mask(attention_mask, seq_len, causal=True)
        if attention_mask is not None:
            log_tensor_summary(f"{self.name}.decode.attention_mask_4d", attention_mask, indent=2)
        # 由调用方传入，为空或 shape 不匹配时在 decode 内构造
        # Megatron GPTModel 传入 labels 时返回 loss（标量），链式 decoder 需要 logits；故传 labels=None
        device = input.device
        if input_ids is None or input_ids.shape != (batch_size, seq_len):
            import os
            if os.environ.get("LMSV_MM_MSARUN") == "1":
                input_ids = _build_token_ids(
                    self.model_config.text_decoder.language_vocab_size,
                    batch_size,
                    seq_len,
                    device="Ascend",
                    context=f"{self.name}.decode fallback input_ids (MSA)",
                )
            elif os.environ.get("LMSV_MM_PTARUN") == "1":
                input_ids = _build_token_ids(
                    self.model_config.text_decoder.language_vocab_size,
                    batch_size,
                    seq_len,
                    device=str(device),
                    context=f"{self.name}.decode fallback input_ids (PTA)",
                )
            else:
                raise ValueError("LMSV_MM_MSARUN or LMSV_MM_PTARUN must be set")
        if position_ids is None or position_ids.shape != (batch_size, seq_len):
            position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
        log_tensor_summary(f"{self.name}.decode.input_ids", input_ids, indent=2)
        log_tensor_summary(f"{self.name}.decode.position_ids", position_ids, indent=2)
        output = self.decoder(
            input_ids = input_ids,
            position_ids = position_ids,
            attention_mask = attention_mask,
            decoder_input = decoder_input,
            labels = None
        )
        final_output = output[0] if isinstance(output, tuple) else output
        log_tensor_summary(f"{self.name}.decode.output", final_output, indent=2)
        return final_output

@register_text_decoder("internvl2_2b", "internvl", "internvl2.5", "internvl3")
class InternVLTextDecoder(TextDecoder):
    @classmethod
    def build(cls, name: str, config: dict) -> "TextDecoder":
        log_step("Build", f"text decoder: {name}", indent=1)
        return cls(name, config, pre_process=True, post_process=True)

    def __init__(self, name: str, config: dict, pre_process: bool = True, post_process: bool = True):
        super().__init__(name, config, pre_process, post_process)
        self.model_config = ConfigReader({"text_decoder": config})
        self.model_config.text_decoder = get_model_config(self.model_config.text_decoder)
        self.model_config.text_decoder.language_transformer_layer_spec = get_language_layer_spec()

        # set the vocab size from config's language_vocab_size or vocab_size
        self.vocab_size = config.get("language_vocab_size", config.get("vocab_size", 32000))

        self._build()

    def _build(self):
        self.decoder = GPTModel(
            config = self.model_config.text_decoder,
            transformer_layer_spec = self.model_config.text_decoder.language_transformer_layer_spec,
            vocab_size = self.vocab_size,
            max_sequence_length = self.model_config.text_decoder.max_position_embeddings,
            parallel_output=self.model_config.text_decoder.parallel_output,
            position_embedding_type=self.model_config.text_decoder.position_embedding_type,
            rotary_percent=self.model_config.text_decoder.rotary_percent,
            rotary_base=self.model_config.text_decoder.rotary_base,
            pre_process=self.pre_process,
            post_process=self.post_process,
            fp16_lm_cross_entropy=self.model_config.text_decoder.fp16_lm_cross_entropy
        )
        
        try:
            device = get_device()
            self.decoder.to(device)
        except Exception:
            log_step("Warning", "Failed to move the decoder to current device", indent=1)

    def embedding(self, input_ids: torch.Tensor, position_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.decoder.embedding(input_ids, position_ids)

    def _prepare_decoder_attention_mask(self, attention_mask, dtype=torch.float32, device=torch.device("npu"), past_key_values_length=0):
        # create causal mask

        # Copied from transformers.models.bart.modeling_bart._make_causal_mask
        def _make_causal_mask(
            input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
        ):
            """
            Make causal mask used for bi-directional self-attention.
            """
            bsz, tgt_len = input_ids_shape
            mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
            mask_cond = torch.arange(mask.size(-1), device=device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(dtype)

            if past_key_values_length > 0:
                mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
            return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

        # Copied from transformers.models.bart.modeling_bart._expand_mask
        def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
            """
            Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
            """
            bsz, src_len = mask.size()
            tgt_len = tgt_len if tgt_len is not None else src_len

            expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

            inverted_mask = 1.0 - expanded_mask

            return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

        input_shape = attention_mask.shape
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                dtype,
                device=device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, dtype, tgt_len=input_shape[-1]).to(device)
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask.bool()
    
    def decode(self, input: torch.Tensor, attention_mask: torch.Tensor, batch_first: bool = False,
               input_ids: torch.Tensor = None, position_ids: torch.Tensor = None, labels: torch.Tensor = None) -> torch.Tensor:
        input = ensure_decoder_input_shape(
            input, self.model_config.text_decoder.hidden_size, self.decoder,
            getattr(self.decoder, 'share_embeddings_and_output_weights', False)
        )
        # decoder 期望 (seq_len, batch, hidden)，与 DeepSeekVL/LLava 一致
        if attention_mask is not None and attention_mask.dim() == 2:
            mask_bsz = attention_mask.shape[0]
            if input.dim() >= 2:
                if batch_first and input.shape[0] != mask_bsz and input.shape[1] == mask_bsz:
                    batch_first = False
                elif (not batch_first) and input.shape[1] != mask_bsz and input.shape[0] == mask_bsz:
                    batch_first = True
        if batch_first:
            decoder_input = input.transpose(0, 1)
            seq_len = input.shape[1]
            batch_size = input.shape[0]
        else:
            decoder_input = input
            seq_len = input.shape[0]
            batch_size = input.shape[1]
        device = input.device
        log_tensor_summary(f"{self.name}.decode.decoder_input", decoder_input, indent=2)
        # NPU FlashAttention 需要 4D attention_mask [B, 1, Sq, Skv]，使用 create_decoder_attention_mask 与其它 decoder 保持一致
        if attention_mask is not None and attention_mask.dim() == 2:
            if attention_mask.shape[1] != seq_len:
                pad_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=attention_mask.device)
                min_len = min(attention_mask.shape[1], seq_len)
                pad_mask[:, :min_len] = attention_mask[:, :min_len]
                attention_mask = pad_mask
            attention_mask = create_decoder_attention_mask(attention_mask, seq_len, causal=True)
        if attention_mask is not None:
            log_tensor_summary(f"{self.name}.decode.attention_mask_4d", attention_mask, indent=2)
        if input_ids is None or input_ids.shape != (batch_size, seq_len):
            import os
            msa_run = os.environ.get("LMSV_MM_MSARUN") == "1"
            input_ids = _build_token_ids(
                self.vocab_size,
                batch_size,
                seq_len,
                device="Ascend" if msa_run else device,
                context=f"{self.name}.decode fallback input_ids ({'MSA' if msa_run else 'PTA'})",
            )
        if position_ids is None or position_ids.shape != (batch_size, seq_len):
            position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
        log_tensor_summary(f"{self.name}.decode.input_ids", input_ids, indent=2)
        log_tensor_summary(f"{self.name}.decode.position_ids", position_ids, indent=2)
        # Megatron GPTModel 传入 labels 时返回 loss（标量），链式 decoder 需要 logits；故传 labels=None 以获取 logits
        output = self.decoder(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=decoder_input,
            labels=None
        )
        final_output = output[0] if isinstance(output, tuple) else output
        log_tensor_summary(f"{self.name}.decode.output", final_output, indent=2)
        return final_output