import os
import torch
import fire
# If you want to load a model using multiple GPUs, please refer to the `Multiple GPUs` section.
from mantis.models.intern_vl_25_8b import InternVLChatModel, InternVLChatConfig, InternVLChatProcessor, InternLM2Tokenizer
from mantis.models.intern_vl_25_8b.modeling_internlm2 import all_events_times, clear_all_events_times, peak_memory_usage
from mantis.models.conversation import conv_templates
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm
from pathlib import Path

PER_IMAGE_NUM_TOKENS = 263 # 258 + 5

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
        responses, _metrics = model.generate(**model_inputs, **generation_config, benchmark_efficiency=True, use_cache=True)
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
    results['peak_memory_usage'] = torch.cuda.max_memory_allocated() / 1024**2
    response = processor.decode(responses[0])
    print(response)
    print('Done benchmarking!')
    return results
    
class cli:
    def __init__(
        self,
        model_path: str='OpenGVLab/InternVL2_5-8B',
        use_flash_attn: bool=True,
        enable_shared_cross_attention: bool=True,
        top_k=-1,
        predict_type='key_norms_small',
        top_k_starting_layer=0,
        adaptive_local_attention=False,
        prune_during_prefill_layer_idx=-1,
        prune_for_query=False,
        run_times=1,
        max_new_tokens=1
    ):
        local_attention_group_size = -1
        tokenizer = InternLM2Tokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        config = InternVLChatConfig.from_pretrained(model_path, enable_shared_cross_attention=enable_shared_cross_attention, local_attention_group_size=local_attention_group_size, adaptive_local_attention=adaptive_local_attention, prune_during_prefill_layer_idx=prune_during_prefill_layer_idx, prune_for_query=prune_for_query)
        config.llm_config.enable_cross_attention = config.enable_cross_attention
        config.llm_config.local_attention_group_size = config.local_attention_group_size
        config.llm_config.enable_shared_cross_attention = config.enable_shared_cross_attention
        config.llm_config.adaptive_local_attention = config.adaptive_local_attention
        config.llm_config.prune_for_query = config.prune_for_query
        model = InternVLChatModel.from_pretrained(model_path, config=config, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, use_flash_attn=use_flash_attn, trust_remote_code=True, device_map='auto').eval()

        processor = InternVLChatProcessor(tokenizer, enable_cross_attention=model.config.enable_cross_attention)

        model.img_context_token_id = processor.img_context_token_id
        model.img_start_token_id = processor.img_start_token_id
        model.img_end_token_id = processor.img_end_token_id
        model.bos_token_id = processor.bos_token_id
        
        for i, decoder_layer in enumerate(model.language_model.model.layers):
            if i >= top_k_starting_layer:
                decoder_layer.attention.top_k = top_k
            else:
                decoder_layer.attention.top_k = -1
            decoder_layer.attention.predict_type = predict_type
            if i == prune_during_prefill_layer_idx:
                decoder_layer.prune_during_prefill = True
            
        conv = conv_templates['internvl2_5'].copy()
        conv.append_message(conv.roles[0], 'Please describe the video in detail.')
        conv.append_message(conv.roles[1], None)
        query = conv.get_prompt()
        query = "<video>\n" + query
        self.query = query

        # print(model_inputs)
        eos_token_id = tokenizer.convert_tokens_to_ids(conv.sep.strip())
        generation_config = dict(max_new_tokens=max_new_tokens, do_sample=False, eos_token_id=eos_token_id)
        self.model = model
        self.generation_config = generation_config
        self.processor = processor
        self.run_times = run_times
        self.enable_shared_cross_attention = enable_shared_cross_attention
        self.use_flash_attn = use_flash_attn
        self.top_k = top_k
        self.predict_type = predict_type
        self.top_k_starting_layer = top_k_starting_layer
        self.max_new_tokens = max_new_tokens
        # self.get_attention(group_size=8, total_frames=8)
        # self.generate(group_size=-1, total_frames=16, images='test_image_1.jpg')

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
            local_attention_group_size = PER_IMAGE_NUM_TOKENS * group_size
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
        peak_memory_usages = [x['peak_memory_usage'] for x in all_results_dict.values()]
        
        # Create figure and axis with a specific size
        # plt.figure(figsize=(12, 8))
        fig, ax1 = plt.subplots(figsize=(12, 8))

        # Plot lines with different styles and markers
        ax1.plot(group_sizes, pure_local_kv_flash_attention_times, 'o-', label='Flash Attention', linewidth=2, markersize=8)
        ax1.plot(group_sizes, mlp_times, 's-', label='MLP Times', linewidth=2, markersize=8)
        ax1.plot(group_sizes, total_prefill_times, '^-', label='Total Prefill', linewidth=2, markersize=8)
        ax1.plot(group_sizes, [sum(x) for x in zip(pure_local_kv_flash_attention_times, mlp_times)], 'v-', label='Flash Attention + MLP Times', linewidth=2, markersize=8)

        # Set x-axis to logarithmic scale since group sizes grow exponentially
        ax1.set_xscale('log', base=2)

        # Customize the plot
        ax1.grid(True, which="both", ls="-", alpha=0.2)
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.set_xlabel('Local Group Size', fontsize=12)
        ax1.set_ylabel('Time (ms)', fontsize=12)
        
        ax2 = ax1.twinx()
        ax2.set_ylabel('Memory Usage (MB)', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='black')
        ax2.plot(group_sizes, peak_memory_usages, 'x-', label='Peak Memory Usage', linewidth=2, markersize=8)
        
        plt.title(f'Performance Metrics vs Group Size (Avg run times: {run_times}, Total frames: {total_frames}, Top K: {self.top_k}, Predict Type: {self.predict_type}, Max New Tokens: {self.max_new_tokens}, Use Flash Attn: {self.use_flash_attn})', fontsize=14, pad=20)

        # Add legend
        # plt.legend(fontsize=10, bbox_to_anchor=(1.02, 1), loc='upper left')
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, bbox_to_anchor=(1.15, 1), loc='upper left')

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
                local_attention_group_size = PER_IMAGE_NUM_TOKENS * group_size
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
            peak_memory_usages = [x['peak_memory_usage'] for x in group_size_all_results_dict.values()]
            ys[f'Flash Attention (g={group_size})'] = pure_local_kv_flash_attention_times
            ys[f'MLP Times (g={group_size})'] = mlp_times
            ys[f'Total Prefill (g={group_size})'] = total_prefill_times
            ys[f'Peak Memory Usage (g={group_size})'] = peak_memory_usages

            
        # # Create figure and axis with a specific size
        # plt.figure(figsize=(12, 8))
        
        
        
        # Create figure and axis with a specific size
        # plt.style.use('seaborn')  # Use seaborn style for better looking plots
        fig, ax1 = plt.subplots(figsize=(12, 8))

        _group_sizes = group_sizes + ['full']
        
        # # Create custom colormaps for each line
        colors1 = ['#FF9999', '#FF0000']  # Light red to dark red
        colors2 = ['#99FF99', '#00FF00']  # Light green to dark green
        colors3 = ['#9999FF', '#0000FF']  # Light blue to dark blue
        colors4 = ['#FFCC99', '#FF9900']  # Light orange to dark orange for memory
        
        # # Use professional color schemes from matplotlib
        # colors1 = plt.cm.viridis(np.linspace(0.3, 0.9, len(_group_sizes)))  # Viridis colormap
        # colors2 = plt.cm.magma(np.linspace(0.3, 0.9, len(_group_sizes)))    # Magma colormap
        # colors3 = plt.cm.plasma(np.linspace(0.3, 0.9, len(_group_sizes)))   # Plasma colormap


        # Create points for gradient
        points = np.linspace(0, 1, len(_group_sizes))


        for i, group_size in enumerate(_group_sizes):
            key = f'Flash Attention (g={group_size})'
            color = plt.matplotlib.colors.to_rgba(colors1[1], (i+1)/len(_group_sizes))
            ax1.plot(x, ys[key], 'o-', label=key, linewidth=2, markersize=8, color=color)
            key = f'MLP Times (g={group_size})'
            color = plt.matplotlib.colors.to_rgba(colors2[1], (i+1)/len(_group_sizes))
            ax1.plot(x, ys[key], 's-', label=key, linewidth=2, markersize=8, color=color)
            key = f'Total Prefill (g={group_size})'
            color = plt.matplotlib.colors.to_rgba(colors3[1], (i+1)/len(_group_sizes))
            ax1.plot(x, ys[key], '^-', label=key, linewidth=2, markersize=8, color=color)

        # Set x-axis to logarithmic scale since group sizes grow exponentially
        ax1.set_xscale('log', base=2)

        # Customize the plot
        ax1.grid(True, which="both", ls="-", alpha=0.2)
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.set_xlabel('Total Frames', fontsize=12)
        ax1.set_ylabel('Time (ms)', fontsize=12)
        
        # Create secondary y-axis (right side) for memory usage
        ax2 = ax1.twinx()

        # Plot memory usage data on the secondary y-axis
        for i, group_size in enumerate(_group_sizes):
            key = f'Peak Memory Usage (g={group_size})'
            if key in ys:
                color = plt.matplotlib.colors.to_rgba(colors4[1], (i+1)/len(_group_sizes))
                line = ax2.plot(x, ys[key], 'x-', label=key, linewidth=2, markersize=8, color=color)
                # Prefix the label to indicate it's on the right axis
                line[0].set_label(f"{key} (right axis)")

        # Configure secondary y-axis
        ax2.set_ylabel('Memory Usage (MB)', fontsize=12, color=colors4[1])
        ax2.tick_params(axis='y', labelcolor=colors4[1])

        plt.title(f'Performance Metrics vs Total Frames (Avg run times: {run_times}, g={",".join(map(str, group_sizes))}, Top K: {self.top_k}, Predict Type: {self.predict_type}, Max New Tokens: {self.max_new_tokens}, Use Flash Attn: {self.use_flash_attn})', fontsize=14, pad=20)

        # Add legend
        # Combine legends from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, bbox_to_anchor=(1.15, 1), loc='upper left')
        # plt.legend(fontsize=10, bbox_to_anchor=(1.02, 1), loc='upper left')

        # Adjust layout to prevent label cutoff
        plt.tight_layout()

        # Show the plot
        plt.show()
        plt.savefig(f'benchmark_fix_group_size_vary_frames_total_frames_{",".join(map(str, total_frames_list))}.png')
        
    def generate(
        self,
        group_size = 8,
        total_frames=128,
        query=None,
        videos=None,
        images=None,
    ):
        """
        Args:
            group_size: int
            total_frames: int
            top_k: int, whether to use select top_k for past_key_values
            predict_type: one of ["salient_tokens", "attention_weights", "attention_weights_sum", "attention_weights_sum_head_tail",
                     "attention_weights_sum_per_image", "attention_weights_sum_with_random", "attention_weights_deduplication",
                     "vector_norms", "key_norms", "output_norms", "weighted_norms"]
        """
        model = self.model
        generation_config = self.generation_config
        processor = self.processor
        run_times = self.run_times
        enable_shared_cross_attention = self.enable_shared_cross_attention
        use_flash_attn = self.use_flash_attn
        
        if videos is None and images is None:
            videos = './mochi.mp4'
        assert [videos, images].count(None) == 1, "Either videos or images should be provided."
        if videos is not None:
            if query is not None:
                conv = conv_templates['internvl2_5'].copy()
                conv.append_message(conv.roles[0], query)
                conv.append_message(conv.roles[1], None)
                query = conv.get_prompt()
                query = "<video>\n" + query
            else:
                query = self.query
            
            model_inputs = processor(query, videos=videos, video_num_segments=total_frames)
        else:
            conv = conv_templates['internvl2_5'].copy()
            conv.append_message(conv.roles[0], query or 'Please describe the image in detail.')
            conv.append_message(conv.roles[1], None)
            query = conv.get_prompt()
            query = "<image>\n" + query
            model_inputs = processor(query, images=images)

        model_inputs['pixel_values'] = model_inputs['pixel_values'].to(torch.bfloat16)
        for key in model_inputs:
            if isinstance(model_inputs[key], torch.Tensor):
                model_inputs[key] = model_inputs[key].to(model.device)
        
        local_attention_group_size = PER_IMAGE_NUM_TOKENS * group_size
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
        # return response
    
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
        
        local_attention_group_size = PER_IMAGE_NUM_TOKENS * group_size
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
        
        conv = conv_templates['internvl2_5'].copy()
        conv.append_message(conv.roles[0], 'Please describe the video in detail.')
        conv.append_message(conv.roles[1], None)
        query = conv.get_prompt()
        query = "<video>\n" + query
        for key in model_inputs:
            print(f"{key}: {model_inputs[key].shape}")
        with torch.no_grad():
            print("Generating...")
            outputs = model.generate(**model_inputs, **generation_config, use_cache=True)
            print("Generated")
            model_inputs['input_ids'] = torch.cat([model_inputs['input_ids'], outputs], dim=1)
            model_inputs['attention_mask'] = torch.cat([model_inputs['attention_mask'], torch.ones((1, outputs.shape[1]), dtype=torch.long, device=model_inputs['attention_mask'].device)], dim=1)
            print("Number of tokens", model_inputs['input_ids'].shape[1])
            print("Forward to create attention map")
            outputs = model(**model_inputs, use_cache=True, output_attentions=True, output_hidden_states=False)
        print(outputs.keys())
        print(f"len(attentions): {len(outputs['attentions'])}")
        print(f"len(attentions[0]): {len(outputs['attentions'][0])}")
        print(f"len(attentions[0][0]): {len(outputs['attentions'][0][0])}")
        attentions = outputs['attentions']
        past_key_values = outputs['past_key_values']
        print(f"inputs_ids.shape: {model_inputs['input_ids'].shape}")
        print(f"len(past_key_values): {len(outputs['past_key_values'])}")
        print(f"Number of attentions: {len(attentions)}")
        print(f"len(attentions[0]): {len(attentions[0])}")
        print(f"attentions[0][0].shape: {attentions[0][0].shape}")
        print(f"attentions[0][1].shape: {attentions[0][1].shape}")
        
        print(f"Saving attention to {save_file}")
        torch.save(attentions, save_file)
        print(f"Saved attention to {save_file}")
        
        input_ids_save_file = save_file.replace('attention', 'input_ids')
        print(f"Saving input_ids to {input_ids_save_file}")
        torch.save(model_inputs['input_ids'], input_ids_save_file)
        print(f"Saved input_ids to {input_ids_save_file}")  
        
        past_key_values_save_file = save_file.replace('attention', 'past_key_values')
        print(f"Saving past_key_values to {past_key_values_save_file}")
        torch.save(past_key_values, past_key_values_save_file)
        print(f"Saved past_key_values to {past_key_values_save_file}")
        return        
    
    
if __name__ == '__main__':
    fire.Fire(cli)
    
"""
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True # for large memory in case of segmentation memory error
python benchmark_internvl_efficiency.py benchmark_vary_group_size_fix_frames --total_frames 1024 --group_sizes "1,2,4,8,16,32,64,128,256,512,1024" --run_times 1
python benchmark_internvl_efficiency.py benchmark_fix_group_size_vary_frames --total_frames_list "16,32,64,128,256,512,1024" --group_sizes "8" --run_times 1

python benchmark_internvl_efficiency.py benchmark_fix_group_size_vary_frames --total_frames_list "16,32,64,128,256" --group_sizes "8" --run_times 1 --top_k 100 --predict_type 'key_norms_small' --max_new_tokens 1 --use_flash_attn True
python benchmark_internvl_efficiency.py benchmark_vary_group_size_fix_frames --total_frames 256 --group_sizes "1,2,4,8,16,32,64,128" --run_times 1 --top_k -1 --predict_type 'key_norms_small' --max_new_tokens 1 --use_flash_attn True

python benchmark_internvl_efficiency.py benchmark_vary_group_size_fix_frames --total_frames 128 --group_sizes "1,2,4,8,16,32,64,128" --run_times 1 --top_k -1 --predict_type 'key_norms_small' --max_new_tokens 1 --use_flash_attn True

python benchmark_internvl_efficiency.py get_attention --group_size 8 --total_frames 8 --use_flash_attn False

python benchmark_internvl_efficiency.py benchmark_fix_group_size_vary_frames --total_frames_list "16,32,64,128,256" --group_sizes "4" --run_times 1 --top_k 100 --predict_type 'attention_weights_sum' --max_new_tokens 128 --use_flash_attn True

### Generation
predict_type: one of ["salient_tokens", "attention_weights", "attention_weights_sum", "attention_weights_sum_head_tail",
                     "attention_weights_sum_per_image", "attention_weights_sum_with_random", "attention_weights_deduplication",
                     "vector_norms", "key_norms", "output_norms", "weighted_norms"]
                     
python benchmark_internvl_efficiency.py generate --group_size 32 --total_frames 128
python benchmark_internvl_efficiency.py generate --group_size 8 --total_frames 64 --top_k -1 --predict_type 'attention_weights_sum' --max_new_tokens 128
python benchmark_internvl_efficiency.py generate --group_size 4 --total_frames 256 --top_k -1 --predict_type 'attention_weights_sum' --max_new_tokens 512 --use_flash_attn True


python benchmark_internvl_efficiency.py generate --group_size -1 --top_k 50 --predict_type 'key_norms_small' --max_new_tokens 512 --use_flash_attn True --top_k_starting_layer 0 --images 'test_image_1.jpg'


python benchmark_internvl_efficiency.py generate --group_size 32 --total_frames 512 --top_k 512 --predict_type 'key_norms_small' --max_new_tokens 512 --use_flash_attn True --top_k_starting_layer -1 --prune_during_prefill_layer_idx -1 --adaptive_local_attention True --prune_for_query True


python benchmark_internvl_efficiency.py generate --group_size -1 --total_frames 32 --top_k 1 --predict_type 'vector_norms_small' --max_new_tokens 512 --use_flash_attn True --top_k_starting_layer 0 --prune_during_prefill_layer_idx -1 --adaptive_local_attention False

python benchmark_internvl_efficiency.py generate --group_size -1 --total_frames 32 --top_k 50 --predict_type 'key_norms_small_deduplication' --max_new_tokens 512 --use_flash_attn True --top_k_starting_layer 0 --prune_during_prefill_layer_idx 2 --adaptive_local_attention False
python benchmark_internvl_efficiency.py generate --group_size -1 --total_frames 32 --top_k 500 --predict_type 'key_norms_small' --max_new_tokens 512 --use_flash_attn True --top_k_starting_layer 0 --prune_during_prefill_layer_idx -1 --adaptive_local_attention True --prune_for_query True

python benchmark_internvl_efficiency.py generate --group_size 32 --total_frames 32 --top_k 128 --predict_type 'key_norms_small' --max_new_tokens 512 --use_flash_attn True --top_k_starting_layer 3
python benchmark_internvl_efficiency.py generate --group_size 32 --total_frames 32 --top_k 8400 --predict_type 'key_norms' --max_new_tokens 512 --use_flash_attn True --top_k_starting_layer 0


# comparison between the original implementation and my implementation
python benchmark_internvl_efficiency.py generate --group_size 1 --total_frames 16 --top_k -1 --predict_type 'attention_weights_sum' --max_new_tokens 512 --use_flash_attn True
python benchmark_internvl_efficiency.py generate --group_size 16 --total_frames 16 --top_k -1 --predict_type 'attention_weights_sum' --max_new_tokens 512 --use_flash_attn True
python benchmark_internvl_efficiency.py generate --group_size 16 --total_frames 16 --top_k -1 --predict_type 'attention_weights_sum' --max_new_tokens 512 --use_flash_attn True
python benchmark_internvl_efficiency.py generate --group_size 16 --total_frames 16 --top_k -1 --predict_type 'attention_weights_sum' --max_new_tokens 512 --use_flash_attn True --enable_shared_cross_attention False

python benchmark_internvl_efficiency.py generate --group_size 256 --total_frames 256 --top_k -1 --predict_type 'vector_norms' --max_new_tokens 512 --use_flash_attn True # this one should be same as the original one, but it seems my implementation is more memory efficient compared to the original one.
python benchmark_internvl_efficiency.py generate --group_size 4 --total_frames 256 --top_k -1 --predict_type 'attention_weights_sum' --max_new_tokens 512 --use_flash_attn True --enable_shared_cross_attention False
python benchmark_internvl_efficiency.py generate --group_size 32 --total_frames 32 --top_k 20 --predict_type 'attention_weights_sum' --max_new_tokens 128 --use_flash_attn True --query "What is the name of the animal?"
python benchmark_internvl_efficiency.py generate --group_size 32 --total_frames 32 --top_k 20 --predict_type 'attention_weights' --max_new_tokens 128 --use_flash_attn True --query "What is the name of the animal?"
python benchmark_internvl_efficiency.py generate --group_size 32 --total_frames 32 --top_k 20 --predict_type 'attention_weights_sum' --max_new_tokens 128 --use_flash_attn True --query "What is the name of the animal?"
python benchmark_internvl_efficiency.py generate --group_size 32 --total_frames 32 --top_k 50 --predict_type 'attention_weights_sum_head_tail' --max_new_tokens 128 --use_flash_attn True --query "What is the name of the animal?"
python benchmark_internvl_efficiency.py generate --group_size 16 --total_frames 512 --top_k 20 --predict_type 'attention_weights_sum' --max_new_tokens 128 --use_flash_attn True --query "What is the name of the animal?"
python benchmark_internvl_efficiency.py generate --group_size 16 --total_frames 512 --top_k 500 --predict_type 'attention_weights_sum' --max_new_tokens 128 --use_flash_attn True --query "What is the name of the animal?"
python benchmark_internvl_efficiency.py generate --group_size 16 --total_frames 128 --top_k 501 --predict_type 'attention_weights_sum' --max_new_tokens 128 --use_flash_attn True --query "What is the name of the animal?" --top_k_starting_layer 3
python benchmark_internvl_efficiency.py generate --group_size 16 --total_frames 64 --top_k 300 --predict_type 'attention_weights_sum' --max_new_tokens 128 --use_flash_attn True --query "What is the name of the animal?" --top_k_starting_layer 3
python benchmark_internvl_efficiency.py generate --group_size 16 --total_frames 64 --top_k 300 --predict_type 'attention_weights' --max_new_tokens 128 --use_flash_attn True --query "What is the name of the animal?" --top_k_starting_layer 3

# which token did eos attend to analyze why local group attention can cause the repetition of the generation
uv pip install torchcodec --index-url=https://download.pytorch.org/whl/cu124
"""