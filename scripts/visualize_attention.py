import torch
import fire
# If you want to load a model using multiple GPUs, please refer to the `Multiple GPUs` section.
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from pathlib import Path
from tqdm import tqdm
from pathlib import Path

def plot_heatmap(attn: torch.Tensor, save_path="./fig.png", title="Attention Heatmap"):
    # Convert mean attention tensor to numpy array
    attention_data = attn.float().cpu().numpy()
    print(f"attention_data.shape: {attention_data.shape}")

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 10))

    # Create heatmap with log scaling
    heatmap = ax.imshow(attention_data, cmap='viridis', aspect='auto', norm=LogNorm())

    # Add colorbar
    cbar = fig.colorbar(heatmap, ax=ax)
    cbar.set_label('Attention Weight (Log Scale)')

    # Set labels and title
    ax.set_xlabel('Source Tokens')
    ax.set_ylabel('Target Tokens')
    ax.set_title('Attention Heatmap')

    # Save the plot
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    

def main(
    attention_file: str,
    save_dir: str = "./attention_plots",
    mode='average'
):
    # Load attention tensor
    attentions = torch.load(attention_file)
    print(f"Number of attentions: {len(attentions)}")
    print(f"len(attentions[0]): {len(attentions[0])}")
    print(f"attentions[0][0].shape: {attentions[0][0].shape}")
    print(f"attentions[0][1].shape: {attentions[0][1].shape}")


    if mode == 'average':
        local_self_attn = [x[0].mean(dim=(0, 1)) for x in attentions]
        text_to_kv_attn = [x[1].mean(dim=(0, 1)) for x in attentions]
        print(f"len(local_self_attn): {len(local_self_attn)}")
        print(f"len(text_to_kv_attn): {len(text_to_kv_attn)}")
        print(f"local_self_attn[0].shape: {local_self_attn[0].shape}")
        print(f"text_to_kv_attn[0].shape: {text_to_kv_attn[0].shape}")
    else:
        raise ValueError(f"Invalid mode: {mode}")
    save_dir = Path(save_dir) / mode
    save_dir.mkdir(parents=True, exist_ok=True)
    for i, attention in tqdm(enumerate(local_self_attn), total=len(local_self_attn), desc="Plotting Local Self Attention"):
        save_path = save_dir / f"local_self_attn_l{i}.png"
        plot_heatmap(attention, save_path=save_path, title=f"Local Self Attention Layer {i}")
    
    for i, attention in tqdm(enumerate(text_to_kv_attn), total=len(text_to_kv_attn), desc="Plotting Text to Key-Value Attention"):
        save_path = save_dir / f"text_to_kv_attn_l{i}.png"
        plot_heatmap(attention, save_path=save_path, title=f"Text to Key-Value Attention Layer {i}")
        
if __name__ == "__main__":
    fire.Fire(main)

"""
python visualize_attention.py --attention_file ./attention_mochi_g8_f8.pt --save_dir ./attention_plots
"""