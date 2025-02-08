import torch
import fire
# If you want to load a model using multiple GPUs, please refer to the `Multiple GPUs` section.
from mantis.models.intern_vl_25_8b import InternVLChatModel, InternVLChatConfig, InternVLChatProcessor, InternLM2Tokenizer
from tqdm import tqdm
from collections import defaultdict

def main(
    model_path: str='OpenGVLab/InternVL2_5-8B',
    use_flash_attn: bool=True,
    enable_shared_cross_attention: bool=True,
    local_attention_group_size: int=258*4,
    run_times=1,
):
    
    tokenizer = InternLM2Tokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    config = InternVLChatConfig.from_pretrained(model_path, enable_shared_cross_attention=enable_shared_cross_attention, local_attention_group_size=local_attention_group_size)
    config.llm_config.enable_cross_attention = config.enable_cross_attention
    config.llm_config.local_attention_group_size = config.local_attention_group_size
    config.llm_config.enable_shared_cross_attention = config.enable_shared_cross_attention
    model = InternVLChatModel.from_pretrained(model_path, config=config, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, use_flash_attn=use_flash_attn, trust_remote_code=True, device_map='auto').eval()

    processor = InternVLChatProcessor(tokenizer, enable_cross_attention=model.config.enable_cross_attention, max_num_patches=1, video_num_segments=1024)

    model.img_context_token_id = processor.img_context_token_id
    model.img_start_token_id = processor.img_start_token_id
    model.img_end_token_id = processor.img_end_token_id
    model.bos_token_id = processor.bos_token_id
    

    from mantis.models.conversation import conv_templates

    conv = conv_templates['internvl2_5'].copy()
    conv.append_message(conv.roles[0], 'Please describe the video in detail.')
    conv.append_message(conv.roles[1], None)
    query = conv.get_prompt()
    query = "<video>\n" + query
    model_inputs = processor(query, videos='./mochi.mp4')

    model_inputs['pixel_values'] = model_inputs['pixel_values'].to(torch.bfloat16)
    for key in model_inputs:
        if isinstance(model_inputs[key], torch.Tensor):
            model_inputs[key] = model_inputs[key].to(model.device)

    # print(model_inputs)
    eos_token_id = tokenizer.convert_tokens_to_ids(conv.sep.strip())
    generation_config = dict(max_new_tokens=1, do_sample=False, eos_token_id=eos_token_id)
    
    
    
    print('Start benchmarking...')
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()  
    print("Number of tokens", model_inputs['input_ids'].shape[1])
    print("Number of frames", model_inputs['pixel_values'].shape[0])
    metrics = defaultdict(float)
    for _ in tqdm(range(run_times), desc='Running...'):
        responses, _metrics = model.generate(**model_inputs, **generation_config, benchmark_efficiency=True)
        for key, value in _metrics.items():
            metrics[key] += value
    for key in metrics:
        metrics[key] /= run_times
    end.record()
    torch.cuda.synchronize()
    print(f'Average time: {(start.elapsed_time(end) / run_times):.2f}ms')
    print(f"Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.2f}")
    response = processor.decode(responses[0])
    print(response)
    
if __name__ == '__main__':
    fire.Fire(main)