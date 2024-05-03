## Usage

- Each `{model_name}_eval.py` can be run directly by `python {model_name}_eval.py` to roughly check the outputs on 3 examples. This is used for debugging purposes.

- Each `{model_name}_eval.py` defines a mllm model class which has a `__init__` method that takes in `model_id` for which checkpoint to load. This class also should have a `__call__` functions which takes in a list of messages in the following format:

- Please check `__init__.py` for a full list of supportted models.


## Example of adding a new model

- in `{model_name}_eval.py`:
```python

class NewModel():
    # support_multi_image: 'merge' images for False, and use custom image 'sequence' format for True
    support_multi_image = True 
    def __init__(self, model_id:str="HuggingFaceM4/idefics2-8b") -> None:
        """

        Args:
            model_path (str): model name
        """
        # load models and processors
        self.model = ...
        self.processor = ...
        
    def __call__(self, inputs: List[dict]) -> str:
        """
        Generate text from images and text prompt. (batch_size=1) one text at a time.
        Args:
            inputs (List[dict]): [
                {
                    "type": "image",
                    "content": "https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/input/sample_34_1.jpg"
                },
                {
                    "type": "image",
                    "content": "https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/input/sample_337180_3.jpg"
                },
                {
                    "type": "text",
                    "content": "What is difference between two images?"
                }
            ]
            Supports any form of interleaved format of image and text.
        Returns:
            str: generated text
        """
        if self.support_multi_image:
            # process images and texts ...
            generation_kwargs = {
                "max_new_tokens": 4096,
                "num_beams": 1,
                "do_sample": False,
            }
            generated_text = self.model.generate(..., generation_kwargs)
            return generated_text
        else:
            raise NotImplementedError

```

- in `__init__.py`:
```python
MLLM_LIST = [..., "NewModel"]
...
def MLLM_Models(model_name:str):
    if ...:
        ...
    elif model_name == "NewModel":
        from .new_model_eval import NewModel
        return NewModel
    else:
        raise NotImplementedError
```