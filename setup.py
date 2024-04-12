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
        "PyYAML",
        "fire",
        "openai>1.0.0",
        "transformers>=4.39",
        "torch",
        "Pillow",
        "torch",
        "torchvision",
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
        "datasets>=2.17.1",
        "sentencepiece"
    ],
)



# change it to pyproject.toml
# [build-system]
