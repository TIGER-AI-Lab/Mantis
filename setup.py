from setuptools import setup, find_packages

setup(
    name='mantis',
    version='0.0.1',
    description='',
    author='Dongfu Jiang',
    author_email='dongfu.jiang@uwaterloo.ca',
    packages=find_packages(),
    url='https://github.com/jdf-prog/many-image-qa',
    install_requires=[
        "transformers",
        "sentencepiece",
        "torch",
        "Pillow",
        "torch",
        "accelerate",
        "torchvision",
        "datasets==2.18.0"
    ],
    extras_require={
        "train": [
            "fire",
            "easy-openai @ git+https://github.com/jdf-prog/easy-openai.git",
            "tqdm",
            "numpy",
            "requests",
            "matplotlib",
            "transformers_stream_generator",
            "tiktoken",
            "chardet",
            "deepspeed",
            "peft",
            "bitsandbytes",
            "wandb",
            "ninja",
            "scipy",
            "webdataset",
            "braceexpand",
            "pandas",
            "orjson",
            "prettytable",
            "pytest",
            "opencv-python",
            "pyarrow",
            "dask",
            "einops-exts",
            "datasets==2.18.0",
            "open_clip_torch"
        ],
        "eval": [
            "tqdm",
            "numpy",
            "easy-openai @ git+https://github.com/jdf-prog/easy-openai.git",
            "prettytable",
            "fire",
            "datasets==2.18.0"
        ]
    }
)



# change it to pyproject.toml
# [build-system]
