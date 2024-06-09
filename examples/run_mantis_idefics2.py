
import requests
import torch
from PIL import Image
from io import BytesIO

from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image


processor = AutoProcessor.from_pretrained("TIGER-Lab/Mantis-8B-Idefics2") # do_image_splitting is False by default
model = AutoModelForVision2Seq.from_pretrained(
    "TIGER-Lab/Mantis-8B-Idefics2",
    device_map="auto"
)
generation_kwargs = {
    "max_new_tokens": 1024,
    "num_beams": 1,
    "do_sample": False
}

# Note that passing the image urls (instead of the actual pil images) to the processor is also possible
image1 = load_image("https://cdn.britannica.com/59/94459-050-DBA42467/Skyline-Chicago.jpg")
image2 = load_image("https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg")
image3 = load_image("https://cdn.britannica.com/68/170868-050-8DDE8263/Golden-Gate-Bridge-San-Francisco.jpg")
images = [image1, image2, image3]


query1 = "What cities image 1, image 2, and image 3 belong to respectively? Answer me in order."
query2 = "Which one do you recommend for a visit? and why?"
query3 = "Which picture has most cars in it?"

### Chat
### Round 1
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "image"},
            {"type": "image"},
            {"type": "text", "text": query1},
        ]
    }    
]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=images, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

# Generate
generated_ids = model.generate(**inputs, **generation_kwargs)
response = processor.batch_decode(generated_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print("User: ", query1)
print("ASSISTANT: ", response[0])

### Round 2
messages.append(
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": response[0]},
        ]
    }
)
messages.append(
    {
        "role": "user",
        "content": [
            {"type": "text", "text": query2},
        ]
    }
)
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=images, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}
generated_ids = model.generate(**inputs, **generation_kwargs)
response = processor.batch_decode(generated_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print("User: ", query2)
print("ASSISTANT: ", response[0])

### Round 3
messages.append(
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": response[0]},
        ]
    }
)
messages.append(
    {
        "role": "user",
        "content": [
            {"type": "text", "text": query3},
        ]
    }
)

prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=images, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}
generated_ids = model.generate(**inputs, **generation_kwargs)
response = processor.batch_decode(generated_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print("User: ", query3)
print("ASSISTANT: ", response[0])


"""
User:  What cities image 1, image 2, and image 3 belong to respectively? Answer me in order.
ASSISTANT:  Chicago, New York, San Francisco
User:  Which one do you recommend for a visit? and why?
ASSISTANT:  New York - because it's a bustling metropolis with iconic landmarks like the Statue of Liberty and the Empire State Building.
User:  Which picture has most cars in it?
ASSISTANT:  Image 3
"""