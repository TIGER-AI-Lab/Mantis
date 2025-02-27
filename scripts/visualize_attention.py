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

def plot_attention_heatmap(attn: torch.Tensor, save_path="./fig.png", title="Attention Heatmap"):
    # Convert mean attention tensor to numpy array
    attention_data = attn.float().cpu().numpy()
    print(f"attention_data.shape: {attention_data.shape}")
    q_len, k_len = attention_data.shape
    
    q_size = max(int(q_len // 100), 1)
    k_size = max(int(k_len // 100), 1)
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(k_size, q_size))
    print(f"figsize: {fig.get_size_inches()}")

    # Create heatmap with log scaling
    heatmap = ax.imshow(attention_data, cmap='viridis', aspect='auto', norm=LogNorm())

    # Add colorbar
    cbar = fig.colorbar(heatmap, ax=ax)
    cbar.set_label('Attention Weight (Log Scale)')

    # Set labels and title
    ax.set_xlabel('KV Tokens')
    ax.set_ylabel('Query Tokens')
    ax.set_title(title)

    # Save the plot
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
def plot_vector_norm_heatmap(vector_norms: torch.Tensor, save_path="./fig.png", title="Vector Norm Heatmap"):
    """
    Args:
        vectors_norms: [num_layers, seq_len]
    """
    # Convert to numpy for matplotlib
    vector_norms_np = vector_norms.float().detach().cpu().numpy()
    print(vector_norms_np)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create heatmap
    im = ax.imshow(vector_norms_np, cmap='viridis', aspect='auto', norm=LogNorm())
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='L2 Norm')
    
    # Set labels and title
    ax.set_xlabel('Sequence Position')
    ax.set_ylabel('Layer')
    ax.set_title(title)
    
    # Add layer indices on y-axis
    num_layers = vector_norms_np.shape[0]
    ax.set_yticks(np.arange(num_layers))
    ax.set_yticklabels([f'Layer {i}' for i in range(num_layers)])
    
    # # Optional: Add grid lines to separate layers
    ax.set_xticks(np.arange(-.5, vector_norms_np.shape[1], 500), minor=False)
    ax.set_yticks(np.arange(-.5, vector_norms_np.shape[0], 1), minor=True)
    
    ax.grid(which='minor', color='w', linestyle='-', linewidth=0.5)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Heatmap saved to {save_path}")

def main(
    attention_file: str,
    past_key_values_file=None,
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
    save_dir = Path(save_dir) / Path(attention_file).stem / mode
    save_dir.mkdir(parents=True, exist_ok=True)
    # for i, attention in tqdm(enumerate(local_self_attn), total=len(local_self_attn), desc="Plotting Local Self Attention"):
    #     save_path = save_dir / f"local_self_attn_l{i}.png"
    #     plot_attention_heatmap(attention, save_path=save_path, title=f"Local Self Attention Layer {i}")
    
    # for i, attention in tqdm(enumerate(text_to_kv_attn), total=len(text_to_kv_attn), desc="Plotting Text to Key-Value Attention"):
    #     save_path = save_dir / f"text_to_kv_attn_l{i}.png"
    #     plot_attention_heatmap(attention, save_path=save_path, title=f"Text to Key-Value Attention Layer {i}")
        
    if past_key_values_file is not None:
        print("Plotting vector norms heatmap")
        past_key_values = torch.load(past_key_values_file)
        vectors = torch.stack([x[1][0].transpose(0, 1).flatten(1, 2) for x in past_key_values])
        vectors = torch.norm(vectors, 2, dim=-1)
        # plot_attention_heatmap(vectors, save_path=save_dir / "vector_norms.png", title="Vector Norm Heatmap")
        plot_vector_norm_heatmap(vectors, save_path=save_dir / "vector_norms.png", title="Vector Norm Heatmap")
        
        
if __name__ == "__main__":
    fire.Fire(main)

"""
python visualize_attention.py --attention_file ./attention_mochi_g8_f8.pt --save_dir ./attention_plots --past_key_values_file ./past_key_values_mochi_g8_f8.pt
python visualize_attention.py --attention_file ./attention_mochi_g4_f8.pt --save_dir ./attention_plots
"""