# Mantis: Multi-Image Instruction Tuning
---

Check out our blog for more details: [Mantis: Multi-Image Instruction Tuning](https://tiger-ai-lab.github.io/Blog/mantis)

Hugging face space Demo: [Mantis](https://huggingface.co/spaces/TIGER-Lab/Mantis)

## Installation
```bash
conda create -n mantis python=3.9
conda activate mantis
pip install -e .
```
## Inference

You can run inference with the following command:
```bash
cd examples
python run_mantis_llava.py
```

Alternatively, you can run the following command to use the pure hugging face codes, without using the Mantis library:
```bash
python run_mantis_llava_hf.py # with pure hugging face codes
```

## Training
**Training codes coming soon**