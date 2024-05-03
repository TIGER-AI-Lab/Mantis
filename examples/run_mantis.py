from mantis.models.mllava import chat_mllava
from PIL import Image
import torch


image1 = "image1.jpg"
image2 = "image2.jpg"
images = [Image.open(image1), Image.open(image2)]

# load processor and model
from mantis.models.mllava import MLlavaProcessor, LlavaForConditionalGeneration
processor = MLlavaProcessor.from_pretrained("TIGER-Lab/Mantis-8B-siglip-llama3")
attn_implementation = None # or "flash_attention_2"
model = LlavaForConditionalGeneration.from_pretrained("TIGER-Lab/Mantis-8B-siglip-llama3", device_map="cuda", torch_dtype=torch.bfloat16, attn_implementation=attn_implementation)

generation_kwargs = {
    "max_new_tokens": 1024,
    "num_beams": 1,
    "do_sample": False
}

# chat
text = "Describe the difference of <image> and <image> as much as you can."
response, history = chat_mllava(text, images, model, processor, **generation_kwargs)

print("USER: ", text)
print("ASSISTANT: ", response)

text = "How many wallets are there in image 1 and image 2 respectively?"
response, history = chat_mllava(text, images, model, processor, history=history, **generation_kwargs)

print("USER: ", text)
print("ASSISTANT: ", response)

"""
USER:  Describe the difference of <image> and <image> as much as you can.
ASSISTANT:  The second image has more variety in terms of colors and designs. While the first image only shows two brown leather pouches, the second image features four different pouches in various colors and designs, including a purple one with a gold coin, a red one with a gold coin, a black one with a gold coin, and a brown one with a gold coin. This variety makes the second image more visually interesting and dynamic.
USER:  How many wallets are there in image 1 and image 2 respectively?
ASSISTANT:  There are two wallets in image 1, and four wallets in image 2.
"""