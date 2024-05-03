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
```
## Inference

You can run inference with the following command:
```bash
cd examples
python run_mantis.py
```

## Training
Install the requirements with the following command:
```bash
pip install -e[train,dev]
```

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
- [🤗 Mantis-Instruct](https://huggingface.co/datasets/TIGER-Lab/Mantis-Instruct) 721K text-image interleaved datasets for multi-image instruction tuning
- [🤗 Mantis-Eval](https://huggingface.co/datasets/TIGER-Lab/Mantis-Eval) 217 high-quality examples for evaluating LMM's multi-image skills


## Model Zoo
We provide the following models in the 🤗 Hugging Face model hub:
- [TIGER-Lab/Mantis-8B-clip-llama3](https://huggingface.co/TIGER-Lab/Mantis-8B-clip-llama3)
- [TIGER-Lab/Mantis-8B-siglip-llama3](https://huggingface.co/TIGER-Lab/Mantis-8B-siglip-llama3)
- [https://huggingface.co/TIGER-Lab/Mantis-8B-Fuyu](https://huggingface.co/TIGER-Lab/Mantis-8B-Fuyu)

The following intermediate checkpoints after pre-training the multi-modal projectors are also available for experiments reproducibility (**Please note the follwing checkpoints still needs further fine-tuning on Mantis-Eval to be intelligent. They are not working models.**):
- [TIGER-Lab/Mantis-8B-clip-llama3-pretraind](https://huggingface.co/TIGER-Lab/Mantis-8B-clip-llama3-pretraind)
- [TIGER-Lab/Mantis-8B-siglip-llama3-pretraind](https://huggingface.co/TIGER-Lab/Mantis-8B-siglip-llama3-pretraind)


## Citation
```bibtex
@article{jiang2024mantis,
  title={MANTIS: Interleaved Multi-Image Instruction Tuning},
  author={Jiang, Dongfu and He, Xuan and Zeng, Huaye and Wei, Con and Ku, Max and Liu, Qian and Chen, Wenhu},
  journal={arXiv preprint arXiv:2405.01483},
  year={2024}
}
```
