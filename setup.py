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
        "transformers>=4.39",
        "sentencepiece",
        "torch",
        "Pillow",
        "torch",
        "torchvision",
    ],
    extra_requires={
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
            "flash-attn",
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
            "datasets>=2.17.1",
        ],
        "eval": [
            "tqdm",
            "numpy",
            "easy-openai @ git+https://github.com/jdf-prog/easy-openai.git",
            "prettytable",
        ]
    }
)



# change it to pyproject.toml
# [build-system]
