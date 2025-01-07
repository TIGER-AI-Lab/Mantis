import torch
from transformers import AutoModel
from mantis.models.intern_vl_25_8b import InternVLChatModel, InternVLChatConfig, InternVLChatProcessor, InternLM2Tokenizer
path = 'OpenGVLab/InternVL2_5-8B'
tokenizer = InternLM2Tokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

from accelerate import init_empty_weights
config = InternVLChatConfig.from_pretrained(path)
config.low_cpu_mem_usage = True
config.enable_cross_attention = True
config.llm_config.enable_cross_attention = True
model = InternVLChatModel._from_config(config, torch_dtype=torch.bfloat16, use_flash_attn=True).eval()
pretrained_model = AutoModel.from_pretrained(path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, use_flash_attn=True, trust_remote_code=True).eval()
model.load_state_dict(pretrained_model.state_dict(), strict=False)
for layer in model.language_model.model.layers:
    layer.cross_attention.load_state_dict(layer.attention.state_dict(), strict=True)
    layer.cross_attention_norm.load_state_dict(layer.attention_norm.state_dict(), strict=True)
    gate_state_dict = {'cross_attn_attn_gate': torch.zeros(1, device=model.device, dtype=torch.bfloat16)}
    layer.load_state_dict(gate_state_dict, strict=False, assign=True)
processor = InternVLChatProcessor(tokenizer, enable_cross_attention=model.config.enable_cross_attention)
model.save_pretrained('../../checkpoints/InternVL2_5-8B/intern_vl_25_llava_next_700k_pretrain_cross_attn_16384/initial_model')
tokenizer.save_pretrained('../../checkpoints/InternVL2_5-8B/intern_vl_25_llava_next_700k_pretrain_cross_attn_16384/initial_model')