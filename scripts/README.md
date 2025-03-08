### Installations
```bash
pip install -e .
pip install seaborn
```

### to run some visualizations

```bash
python benchmark_internvl_efficiency.py get_attention --group_size 8 --total_frames 8 --use_flash_attn False
```
this will save the attention tensors and kv cache

```bash
python visualize_attention.py --attention_file ./attention_mochi_g8_f8.pt --save_dir ./attention_plots --past_key_values_file ./past_key_values_mochi_g8_f8.pt
```
this will save the attention plots

Then run the `attention_analysis.ipynb` notebook to visualize the attention plots