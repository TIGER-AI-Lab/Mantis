# Mantis: Multi-Image Instruction Tuning

**Mantis** is a multimodal conversational AI model that can chat with users about images and text. It's optimized for multi-image reasoning, where inverleaved text and images can be used to generate responses.

- Website: https://tiger-ai-lab.github.io/Mantis/

- Paper: https://arxiv.org/abs/2405.01483

- 🤗 Hugging face space Demo: [Mantis](https://huggingface.co/spaces/TIGER-Lab/Mantis)

- 🤗 Models: [TIGER-Lab/Mantis](https://huggingface.co/collections/TIGER-Lab/mantis-6619b0834594c878cdb1d6e4)

## Installation
```bash
conda create -n mantis python=3.9
conda activate mantis
pip install -e .
# install flash-attention
pip install flash-attn --no-build-isolation
```
## Inference

You can run inference with the following command:
```bash
cd examples
python run_mantis.py
```

Alternatively, you can run the following command with pure hugging face codes, without using the Mantis library:
```bash
python run_mantis_hf.py # with pure hugging face codes
```

## Training
We support training of Mantis based on the Fuyu architecture and the LLaVA architecture. You can train the model with the following command:

**Training Mantis based on LLaMA3 with CLIP/SigLIP encoder:**
- Pretrain Mantis-LLaMA3 Multimodal projector on pretrain data (Stage 1):
```bash
cd mantis/train/pretrain_mllava.sh
```

- Fine-tune the pretrained Mantis-LLaMA3 on Mantis-Instruct (Stage 2):
```bash
cd mantis/train/train_mllava.sh
```

**Training Mantis based on Fuyu-8B:**
- Fine-tune Fuyu-8B on Mantis-Instruct to get Mantis-Fuyu:
```bash
cd mantis/train/train_fuyu.sh
```

## Data
- [🤗 Mantis-Instruct](TIGER-Lab/Mantis-Instruct) 721K text-image interleaved datasets for multi-image instruction tuning
- [🤗 Mantis-Eval](TIGER-Lab/MIQA-Eval) 217 high-quality examples for evaluating LMM's multi-image skills


## Citation
```bibtex
@misc{jiang2024mantis,
    title={Mantis: Interleaved Multi-Image Instruction Tuning},
    url={https://tiger-ai-lab.github.io/Blog/mantis},
    author={Jiang, Dongfu and He, Xuan and Zeng, Huaye and Wei, Cong and Max Ku and Liu, Qian and Chen, Wenhu},
    month={April},
    year={2024}
}
```
