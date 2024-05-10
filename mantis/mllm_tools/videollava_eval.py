"""
conda create -n vila python=3.10
conda activate vila

pip install --upgrade pip  # enable PEP 660 support
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.4.2/flash_attn-2.4.2+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.4.2+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install -e .
pip install -e ".[train]"

site_pkg_path=$(python -c 'import site; print(site.getsitepackages()[0])')
cp -rv ./llava/train/transformers_replace/* $site_pkg_path/transformers/

# then install mantis for eval, in the root directory of the repo
pip install -e ".[eval]"
"""
import os
import torch
from typing import List
try:
    from mllm_utils import load_images
except ImportError:
    from .mllm_utils import load_images
import re
    
try:
    from videollava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from videollava.conversation import conv_templates, SeparatorStyle
    from videollava.model.builder import load_pretrained_model
    from videollava.utils import disable_torch_init
    from videollava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
except:
    raise ImportError("Please install see mllm_tools/videollava_eval.py for running requirements. Due to the videollava project's bad compatibilities, you need to install a additional environment for running this code.")

class VideoLlava():
    support_multi_image = True
    merged_image_files = []
    def __init__(self, model_path:str='LanguageBind/Video-LLaVA-7B', model_base:str=None) -> None:
        """Llava model wrapper

        Args:
            model_path (str): Video Llava model name
        """
        device = 'cuda'
        self.model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.processor, _ = load_pretrained_model(model_path, model_base, self.model_name, load_8bit=False, load_4bit=True, device=device)
        self.conv_template = conv_templates["llava_v1"]
        self.image_processor = self.processor["image"]

        
    def __call__(self, inputs: List[dict]) -> str:
        """
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
        """
        image_links = [x["content"] for x in inputs if x["type"] == "image"]
        if self.support_multi_image:
            text_prompt = ""
            for i, message in enumerate(inputs):
                if message["type"] == "text":
                    text_prompt += message["content"]
                elif message["type"] == "image":
                    text_prompt += f"{DEFAULT_IMAGE_TOKEN} \n"
            images = load_images(image_links)
            image_tensor = self.image_processor.preprocess(images, return_tensors='pt')['pixel_values']
            if type(image_tensor) is list:
                tensor = [image.to(self.model.device, dtype=torch.float16) for image in image_tensor]
            else:
                tensor = image_tensor.to(self.model.device, dtype=torch.float16)

            conv = self.conv_template.copy()
            roles = conv.roles
            conv.append_message(conv.roles[0], text_prompt)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=tensor,
                    do_sample=True,
                    temperature=0.2,
                    max_new_tokens=1024,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria])

            outputs = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()

            return outputs            
                    
        else:
            raise NotImplementedError
        
    def __del__(self):
        for image_file in self.merged_image_files:
            if os.path.exists(image_file):
                os.remove(image_file)
    
if __name__ == "__main__":
    model = VideoLlava()
    # 0 shot
    zero_shot_exs = [
        {
            "type": "image",
            "content": "https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/input/sample_34_1.jpg"
        },
        {
            "type": "text",
            "content": "What is in the image?"
        }
    ]
    # 1 shot
    one_shot_exs = [
        {
            "type": "image",
            "content": "https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/input/sample_34_1.jpg"
        },
        {
            "type": "text",
            "content": "What is in the image? A zebra."
        },
        {
            "type": "image",
            "content": "https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/input/sample_337180_3.jpg"
        },
        {
            "type": "text",
            "content": "What is in the image?"
        }
    ]
    # 2 shot
    two_shot_exs = [
        {
            "type": "image",
            "content": "https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/input/sample_34_1.jpg"
        },
        {
            "type": "text",
            "content": "What is in the image? A zebra."
        },
        {
            "type": "image",
            "content": "https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/input/sample_337180_3.jpg"
        },
        {
            "type": "text",
            "content": "What is in the image? A black cat."
        },
        {
            "type": "image",
            "content": "https://hips.hearstapps.com/hmg-prod/images/rabbit-breeds-american-white-1553635287.jpg?crop=0.976xw:0.651xh;0.0242xw,0.291xh&resize=980:*"
        },
        {
            "type": "text",
            "content": "What is in the image?"
        }
    ]
    print("### 0 shot")
    print(model(zero_shot_exs))
    print("### 1 shot")
    print(model(one_shot_exs))
    print("### 2 shot")
    print(model(two_shot_exs))
    """
    Output: a tiger and a zebra
    ### 0 shot
    The image features a zebra grazing on grass in a field.
    ### 1 shot
    A black cat.
    ### 2 shot
    A rabbit.
    """
    