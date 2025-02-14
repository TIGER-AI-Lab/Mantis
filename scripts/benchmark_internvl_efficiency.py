import torch
import fire
# If you want to load a model using multiple GPUs, please refer to the `Multiple GPUs` section.
from mantis.models.intern_vl_25_8b import InternVLChatModel, InternVLChatConfig, InternVLChatProcessor, InternLM2Tokenizer
from mantis.models.intern_vl_25_8b.modeling_internlm2 import all_events_times, clear_all_events_times, peak_memory_usage
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from pathlib import Path
from tqdm import tqdm
from pathlib import Path

def run_benchmark(model, model_inputs, generation_config, processor, run_times):
    clear_all_events_times()
    print('Start benchmarking...')
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()  
    print("Number of tokens", model_inputs['input_ids'].shape[1])
    print("Number of frames", model_inputs['pixel_values'].shape[0])
    metrics = defaultdict(float)
    for _ in tqdm(range(run_times), desc='Running...'):
        responses, _metrics = model.generate(**model_inputs, **generation_config, benchmark_efficiency=True, use_cache=False)
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
    
    prefill_total_time = metrics['total_prefill_time'] * run_times
    sorted_all_events_times = dict(sorted(all_events_times.items(), key=lambda x: x[0]))
    results = {}
    for key, value in sorted_all_events_times.items():
        level = value['level']
        times = value['times']
        message = value['message']
        num_called = len(times)
        total_time = sum(times)
        average_time = total_time / num_called
        percentage = total_time / prefill_total_time * 100
        # print(level * "    " + f"- {key}: {average_time:.4f} ms ({num_called} calls) ({percentage:.2f}%)")
        print(level * "    " + f"- {key} {message}: {average_time:.4f} ms ({num_called} calls) ({percentage:.2f}%)")
        results[key] = {'average_time': average_time, 'num_called': num_called, 'percentage': percentage, 'message': message, 'level': level, 'total_time_per_run': total_time/run_times}
    results['total_prefill_time_per_run'] = metrics['total_prefill_time']
    results['total_prefill_time'] = prefill_total_time
    results['peak_memory_usage'] = peak_memory_usage()
    response = processor.decode(responses[0])
    print(response)
    print('Done benchmarking!')
    return results

def benchmark_vary_group_size_fix_frames(model, model_inputs, generation_config, processor, run_times, group_sizes, enable_shared_cross_attention, use_flash_attn, total_frames):
    all_results_dict = {}
    for group_size in group_sizes:
        local_attention_group_size = 258 * group_size
        model.config.local_attention_group_size = local_attention_group_size
        for decoder_layer in model.language_model.model.layers:
            decoder_layer.local_attention_group_size = local_attention_group_size
        
        print(f"Input_ids shape: {model_inputs['input_ids'].shape}")
        print(f"Group size: {group_size}")
        print(f"Local attention group size: {local_attention_group_size}")
        print(f"Enable shared cross attention: {enable_shared_cross_attention}")
        print(f"Use Flash Attention: {use_flash_attn}")
        print(f"Running {run_times} times")
        results = run_benchmark(model, model_inputs, generation_config, processor, run_times)
        all_results_dict[group_size] = results
    
    # plot results
    time_key = "total_time_per_run"
    pure_local_kv_flash_attention_times = [x['1.1.3.3.1'][time_key] for x in all_results_dict.values()] # flash_attn_varlen_func
    mlp_keys = ["1.1.1.1", "1.1.1.4", "1.1.3.1", "1.1.3.4", "1.1.5", "1.1.6"]
    mlp_times = [sum([x[key][time_key] for key in mlp_keys]) for x in all_results_dict.values()]
    total_prefill_times = [x['total_prefill_time_per_run'] for x in all_results_dict.values()]
    
    # Create figure and axis with a specific size
    plt.figure(figsize=(12, 8))

    # Plot lines with different styles and markers
    plt.plot(group_sizes, pure_local_kv_flash_attention_times, 'o-', label='Flash Attention', linewidth=2, markersize=8)
    plt.plot(group_sizes, mlp_times, 's-', label='MLP Times', linewidth=2, markersize=8)
    plt.plot(group_sizes, total_prefill_times, '^-', label='Total Prefill', linewidth=2, markersize=8)

    # Set x-axis to logarithmic scale since group sizes grow exponentially
    plt.xscale('log', base=2)

    # Customize the plot
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.xlabel('Local Group Size', fontsize=12)
    plt.ylabel('Time (ms)', fontsize=12)
    plt.title(f'Performance Metrics vs Group Size (Avg run times: {run_times}, Total frames: {total_frames})', fontsize=14, pad=20)

    # Add legend
    plt.legend(fontsize=10, bbox_to_anchor=(1.02, 1), loc='upper left')

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Show the plot
    plt.show()
    plt.savefig('benchmark.png') 
    
class cli:
    def __init__(
        self,
        model_path: str='OpenGVLab/InternVL2_5-8B',
        use_flash_attn: bool=True,
        enable_shared_cross_attention: bool=True,
        run_times=1,
    ):
        local_attention_group_size = 258 * 1
        tokenizer = InternLM2Tokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        config = InternVLChatConfig.from_pretrained(model_path, enable_shared_cross_attention=enable_shared_cross_attention, local_attention_group_size=local_attention_group_size)
        config.llm_config.enable_cross_attention = config.enable_cross_attention
        config.llm_config.local_attention_group_size = config.local_attention_group_size
        config.llm_config.enable_shared_cross_attention = config.enable_shared_cross_attention
        model = InternVLChatModel.from_pretrained(model_path, config=config, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, use_flash_attn=use_flash_attn, trust_remote_code=True, device_map='auto').eval()

        processor = InternVLChatProcessor(tokenizer, enable_cross_attention=model.config.enable_cross_attention, max_num_patches=1)

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
        self.query = query

        # print(model_inputs)
        eos_token_id = tokenizer.convert_tokens_to_ids(conv.sep.strip())
        generation_config = dict(max_new_tokens=1, do_sample=False, eos_token_id=eos_token_id)
        self.model = model
        self.generation_config = generation_config
        self.processor = processor
        self.run_times = run_times
        self.enable_shared_cross_attention = enable_shared_cross_attention
        self.use_flash_attn = use_flash_attn

    def benchmark_vary_group_size_fix_frames(
        self,
        group_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256],
        total_frames=256
    ):
        if isinstance(group_sizes, int):
            group_sizes = [group_sizes]
        model = self.model
        generation_config = self.generation_config
        processor = self.processor
        run_times = self.run_times
        enable_shared_cross_attention = self.enable_shared_cross_attention
        use_flash_attn = self.use_flash_attn
        
        model_inputs = processor(self.query, videos='./mochi.mp4', video_num_segments=total_frames)

        model_inputs['pixel_values'] = model_inputs['pixel_values'].to(torch.bfloat16)
        for key in model_inputs:
            if isinstance(model_inputs[key], torch.Tensor):
                model_inputs[key] = model_inputs[key].to(model.device)
        all_results_dict = {}
        
        for group_size in group_sizes:
            local_attention_group_size = 258 * group_size
            model.config.local_attention_group_size = local_attention_group_size
            for decoder_layer in model.language_model.model.layers:
                decoder_layer.local_attention_group_size = local_attention_group_size
            
            print(f"Input_ids shape: {model_inputs['input_ids'].shape}")
            print(f"Group size: {group_size}")
            print(f"Local attention group size: {local_attention_group_size}")
            print(f"Enable shared cross attention: {enable_shared_cross_attention}")
            print(f"Use Flash Attention: {use_flash_attn}")
            print(f"Running {run_times} times")
            results = run_benchmark(model, model_inputs, generation_config, processor, run_times)
            all_results_dict[group_size] = results
        
        # plot results
        time_key = "total_time_per_run"
        pure_local_kv_flash_attention_times = [x['1.1.3.3.1'][time_key] for x in all_results_dict.values()] # flash_attn_varlen_func
        mlp_keys = ["1.1.1.1", "1.1.1.4", "1.1.3.1", "1.1.3.4", "1.1.5", "1.1.6"]
        mlp_times = [sum([x[key][time_key] for key in mlp_keys]) for x in all_results_dict.values()]
        total_prefill_times = [x['total_prefill_time_per_run'] for x in all_results_dict.values()]
        
        # Create figure and axis with a specific size
        plt.figure(figsize=(12, 8))

        # Plot lines with different styles and markers
        plt.plot(group_sizes, pure_local_kv_flash_attention_times, 'o-', label='Flash Attention', linewidth=2, markersize=8)
        plt.plot(group_sizes, mlp_times, 's-', label='MLP Times', linewidth=2, markersize=8)
        plt.plot(group_sizes, total_prefill_times, '^-', label='Total Prefill', linewidth=2, markersize=8)

        # Set x-axis to logarithmic scale since group sizes grow exponentially
        plt.xscale('log', base=2)

        # Customize the plot
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.xlabel('Local Group Size', fontsize=12)
        plt.ylabel('Time (ms)', fontsize=12)
        plt.title(f'Performance Metrics vs Group Size (Avg run times: {run_times}, Total frames: {total_frames})', fontsize=14, pad=20)

        # Add legend
        plt.legend(fontsize=10, bbox_to_anchor=(1.02, 1), loc='upper left')

        # Adjust layout to prevent label cutoff
        plt.tight_layout()

        # Show the plot
        plt.show()
        plt.savefig(f'benchmark_vary_group_size_fix_frames_{total_frames}.png')
        
    def benchmark_fix_group_size_vary_frames(
        self,
        group_sizes = [8],
        total_frames_list=[16, 32, 64, 128, 256, 512, 1024]
    ):
        if isinstance(group_sizes, int):
            group_sizes = [group_sizes]
        if isinstance(total_frames_list, int):
            total_frames_list = [total_frames_list]
        model = self.model
        generation_config = self.generation_config
        processor = self.processor
        run_times = self.run_times
        enable_shared_cross_attention = self.enable_shared_cross_attention
        use_flash_attn = self.use_flash_attn
        
        
        
        all_results_dict = {}
        for total_frames in total_frames_list:
            model_inputs = processor(self.query, videos='./mochi.mp4', video_num_segments=total_frames)

            model_inputs['pixel_values'] = model_inputs['pixel_values'].to(torch.bfloat16)
            for key in model_inputs:
                if isinstance(model_inputs[key], torch.Tensor):
                    model_inputs[key] = model_inputs[key].to(model.device)
            
            if total_frames not in group_sizes:
                _group_sizes = group_sizes + [total_frames]
            else:
                _group_sizes = group_sizes
            
            for group_size in _group_sizes:
                local_attention_group_size = 258 * group_size
                model.config.local_attention_group_size = local_attention_group_size
                for decoder_layer in model.language_model.model.layers:
                    decoder_layer.local_attention_group_size = local_attention_group_size
                
                print(f"Input_ids shape: {model_inputs['input_ids'].shape}")
                print(f"The number of frames: {total_frames}")
                print(f"Group size: {group_size}")
                print(f"Local attention group size: {local_attention_group_size}")
                print(f"Enable shared cross attention: {enable_shared_cross_attention}")
                print(f"Use Flash Attention: {use_flash_attn}")
                print(f"Running {run_times} times")
                results = run_benchmark(model, model_inputs, generation_config, processor, run_times)
                if group_size not in all_results_dict:
                    all_results_dict[group_size] = {}
                all_results_dict[group_size][total_frames] = results
            if 'full' not in all_results_dict:
                all_results_dict['full'] = {}
            all_results_dict['full'][total_frames] = all_results_dict[total_frames][total_frames]
            
        # plot results
        time_key = "total_time_per_run"
        mlp_keys = ["1.1.1.1", "1.1.1.4", "1.1.3.1", "1.1.3.4", "1.1.5", "1.1.6"]
        x = total_frames_list
        ys = {}
        for group_size in group_sizes + ['full']:
            group_size_all_results_dict = all_results_dict[group_size]
            pure_local_kv_flash_attention_times = [x['1.1.3.3.1'][time_key] for x in group_size_all_results_dict.values()] # flash_attn_varlen_func
            mlp_times = [sum([x[key][time_key] for key in mlp_keys]) for x in group_size_all_results_dict.values()]
            total_prefill_times = [x['total_prefill_time_per_run'] for x in group_size_all_results_dict.values()]
            ys[f'Flash Attention (g={group_size})'] = pure_local_kv_flash_attention_times
            ys[f'MLP Times (g={group_size})'] = mlp_times
            ys[f'Total Prefill (g={group_size})'] = total_prefill_times

            
        # # Create figure and axis with a specific size
        # plt.figure(figsize=(12, 8))
        
        
        
        # Create figure and axis with a specific size
        # plt.style.use('seaborn')  # Use seaborn style for better looking plots
        plt.figure(figsize=(12, 8))

        _group_sizes = group_sizes + ['full']
        
        # # Create custom colormaps for each line
        colors1 = ['#FF9999', '#FF0000']  # Light red to dark red
        colors2 = ['#99FF99', '#00FF00']  # Light green to dark green
        colors3 = ['#9999FF', '#0000FF']  # Light blue to dark blue
        
        # # Use professional color schemes from matplotlib
        # colors1 = plt.cm.viridis(np.linspace(0.3, 0.9, len(_group_sizes)))  # Viridis colormap
        # colors2 = plt.cm.magma(np.linspace(0.3, 0.9, len(_group_sizes)))    # Magma colormap
        # colors3 = plt.cm.plasma(np.linspace(0.3, 0.9, len(_group_sizes)))   # Plasma colormap


        # Create points for gradient
        points = np.linspace(0, 1, len(_group_sizes))


        for i, group_size in enumerate(_group_sizes):
            key = f'Flash Attention (g={group_size})'
            color = plt.matplotlib.colors.to_rgba(colors1[1], (i+1)/len(_group_sizes))
            plt.plot(x, ys[key], 'o-', label=key, linewidth=2, markersize=8, color=color)
            key = f'MLP Times (g={group_size})'
            color = plt.matplotlib.colors.to_rgba(colors2[1], (i+1)/len(_group_sizes))
            plt.plot(x, ys[key], 's-', label=key, linewidth=2, markersize=8, color=color)
            key = f'Total Prefill (g={group_size})'
            color = plt.matplotlib.colors.to_rgba(colors3[1], (i+1)/len(_group_sizes))
            plt.plot(x, ys[key], '^-', label=key, linewidth=2, markersize=8, color=color)

        # Set x-axis to logarithmic scale since group sizes grow exponentially
        plt.xscale('log', base=2)

        # Customize the plot
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.xlabel('Total Frames', fontsize=12)
        plt.ylabel('Time (ms)', fontsize=12)
        plt.title(f'Performance Metrics vs Total Frames (Avg run times: {run_times}, g=group size)', fontsize=14, pad=20)

        # Add legend
        plt.legend(fontsize=10, bbox_to_anchor=(1.02, 1), loc='upper left')

        # Adjust layout to prevent label cutoff
        plt.tight_layout()

        # Show the plot
        plt.show()
        plt.savefig(f'benchmark_fix_group_size_vary_frames_total_frames_{",".join(map(str, total_frames_list))}.png')
        
    def generate(
        self,
        group_size = 8,
        total_frames=128
    ):
        model = self.model
        generation_config = self.generation_config
        generation_config['max_new_tokens'] = 512
        processor = self.processor
        run_times = self.run_times
        enable_shared_cross_attention = self.enable_shared_cross_attention
        use_flash_attn = self.use_flash_attn
        
        model_inputs = processor(self.query, videos='./mochi.mp4', video_num_segments=total_frames)

        model_inputs['pixel_values'] = model_inputs['pixel_values'].to(torch.bfloat16)
        for key in model_inputs:
            if isinstance(model_inputs[key], torch.Tensor):
                model_inputs[key] = model_inputs[key].to(model.device)
        
        local_attention_group_size = 258 * group_size
        model.config.local_attention_group_size = local_attention_group_size
        for decoder_layer in model.language_model.model.layers:
            decoder_layer.local_attention_group_size = local_attention_group_size
        
        print(f"Input_ids shape: {model_inputs['input_ids'].shape}")
        print(f"Group size: {group_size}")
        print(f"Local attention group size: {local_attention_group_size}")
        print(f"Enable shared cross attention: {enable_shared_cross_attention}")
        print(f"Use Flash Attention: {use_flash_attn}")
        print(f"Running {run_times} times")
        print("Number of tokens", model_inputs['input_ids'].shape[1])
        print("Number of frames", model_inputs['pixel_values'].shape[0])
        metrics = defaultdict(float)
        for _ in tqdm(range(run_times), desc='Running...'):
            responses = model.generate(**model_inputs, **generation_config)
        response = processor.decode(responses[0])
        print(response)
        return response
    
    def get_attention(
        self,
        group_size = 8,
        total_frames=128,
        save_file=None,
        video_file='./mochi.mp4'
    ):
        video_name = Path(video_file).stem
        if save_file is None:
            save_file = f'attention_{video_name}_g{group_size}_f{total_frames}.pt'
        model = self.model
        generation_config = self.generation_config
        processor = self.processor
        run_times = self.run_times
        enable_shared_cross_attention = self.enable_shared_cross_attention
        use_flash_attn = self.use_flash_attn
        
        model_inputs = processor(self.query, videos='./mochi.mp4', video_num_segments=total_frames)

        model_inputs['pixel_values'] = model_inputs['pixel_values'].to(torch.bfloat16)
        for key in model_inputs:
            if isinstance(model_inputs[key], torch.Tensor):
                model_inputs[key] = model_inputs[key].to(model.device)
        
        local_attention_group_size = 258 * group_size
        model.config.local_attention_group_size = local_attention_group_size
        for decoder_layer in model.language_model.model.layers:
            decoder_layer.local_attention_group_size = local_attention_group_size
        
        print(f"Input_ids shape: {model_inputs['input_ids'].shape}")
        print(f"Group size: {group_size}")
        print(f"Local attention group size: {local_attention_group_size}")
        print(f"Enable shared cross attention: {enable_shared_cross_attention}")
        print(f"Use Flash Attention: {use_flash_attn}")
        print(f"Running {run_times} times")
        print("Number of tokens", model_inputs['input_ids'].shape[1])
        print("Number of frames", model_inputs['pixel_values'].shape[0])
        
        with torch.no_grad():
            outputs = model(**model_inputs, output_attentions=True, use_cache=False)
        attentions = outputs['attentions']
        print(f"Number of attentions: {len(attentions)}")
        print(f"len(attentions[0]): {len(attentions[0])}")
        print(f"attentions[0][0].shape: {attentions[0][0].shape}")
        print(f"attentions[0][1].shape: {attentions[0][1].shape}")
        
        print(f"Saving attention to {save_file}")
        torch.save(attentions, save_file)
        print(f"Saved attention to {save_file}")
        return        
    
    
if __name__ == '__main__':
    fire.Fire(cli)
    
"""
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True # for large memory in case of segmentation memory error
python benchmark_internvl_efficiency.py benchmark_vary_group_size_fix_frames --total_frames 1024 --group_sizes "1,2,4,8,16,32,64,128,256,512,1024" --run_times 1
python benchmark_internvl_efficiency.py benchmark_fix_group_size_vary_frames --total_frames_list "16,32,64,128,256,512,1024" --group_sizes "8" --run_times 1
python benchmark_internvl_efficiency.py generate --group_size 32 --total_frames 128
python benchmark_internvl_efficiency.py get_attention --group_size 8 --total_frames 8 --use_flash_attn False
"""