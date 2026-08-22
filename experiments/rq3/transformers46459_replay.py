import os, tempfile
import torch
from transformers import AutoModelForCausalLM, Qwen3_5Config, Qwen3_5ForConditionalGeneration

text_config = {
    'bos_token_id': 0, 'eos_token_id': 1, 'pad_token_id': 2,
    'hidden_act': 'silu', 'head_dim': 8, 'hidden_size': 32, 'vocab_size': 99,
    'intermediate_size': 37, 'max_position_embeddings': 128, 'model_type': 'qwen3_5_text',
    'num_attention_heads': 4, 'num_hidden_layers': 2,
    'layer_types': ['full_attention', 'linear_attention'], 'num_key_value_heads': 2,
    'rope_theta': 10000,
    'rope_parameters': {'rope_type':'default','mrope_section':[16,8,8],'mrope_interleaved':True},
    'linear_conv_kernel_dim': 2, 'linear_key_head_dim':16, 'linear_value_head_dim':16,
    'linear_num_key_heads':4, 'linear_num_value_heads':8,
}
vision_config = {
    'depth':2,'in_chans':3,'hidden_act':'gelu_pytorch_tanh','intermediate_size':32,
    'out_hidden_size':32,'hidden_size':32,'num_heads':4,'patch_size':16,
    'spatial_merge_size':1,'temporal_patch_size':2,'num_position_embeddings':16,
}
config = Qwen3_5Config(text_config=text_config, vision_config=vision_config,
    image_token_id=3, video_token_id=4, vision_start_token_id=5, vision_end_token_id=6,
    tie_word_embeddings=True)
with tempfile.TemporaryDirectory() as tmp:
    full = Qwen3_5ForConditionalGeneration(config).to(torch.bfloat16)
    # Touch the selected GPU while keeping the save/load oracle architecture-independent.
    device = torch.device('cuda:0')
    full = full.to(device)
    _ = sum(p.numel() for p in full.parameters())
    full = full.cpu()
    full.save_pretrained(tmp)
    del full
    loaded = AutoModelForCausalLM.from_pretrained(tmp, dtype=torch.float32)
    actual = next(loaded.parameters()).dtype
    print(f'RQ3_DTYPE={actual}', flush=True)
    if actual != torch.float32:
        raise RuntimeError(f'requested float32 but loaded {actual}')
    loaded = loaded.to(device)
    print('RQ3_REPLAY_COMPLETED', flush=True)
