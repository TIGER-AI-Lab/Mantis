import datasets
import fire
import random
import requests
import json
from typing import List, Dict
from io import BytesIO
from PIL import Image
from tqdm import tqdm
from pathlib import Path

def load_image(image_file):
    if image_file is None:
        return None
    try:
        if image_file.startswith("http"):
            response = requests.get(image_file, timeout=1)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            import os
            image = Image.open(image_file).convert("RGB")
        return image
    except Exception as e:
        print(f"Error loading image {image_file}: {e}")
        return None

def load_images(image_files):
    out = []
    for image_file in image_files:
        if isinstance(image_file, Image.Image):
            image = image_file
        else:
            image = load_image(image_file)
        out.append(image)
    return out


def get_complex_image_denotation(image_idx:int, is_last:bool=False):
    """
    Generate multiple different forms of denotations of the images for each image number.
    For example, if the image number is 1, the denotation can be "the first image", "the leftmost image", etc
    """
    image_number = image_idx + 1
    # List of various ways to denote an image based on its position
    denotations = {
        1: ["the first image", "the initial image", "the opening image", "the foremost image", "the lead image", "the starting image", "the premiere image"],
        2: ["the second image", "the secondary image", "the image directly after the first"],
        3: ["the third image", "the third image in sequence", "the third image to come", "following the second image", "the tertiary image"],
        4: ["the fourth image", "the fourth image in sequence", "the image following the third image", "the fourth to come"],
        5: ["the fifth image", "the fifth image in sequence", "the image following the fourth image", "the fifth image to come"],
        6: ["the sixth image", "the sixth image in sequence", "the image following the fifth image", "the sixth image to come"],
        7: ["the seventh image", "the seventh image in sequence", "the image following the sixth image", "the seventh image to come", "the image of lucky number seven"],
        8: ["the eighth image", "the eighth image in sequence", "the image following the seventh image", "the eighth image to come"],
        9: ["the ninth image", "the ninth image in sequence", "the image following the eighth image", "the ninth image to come"],
        10: ["the tenth image", "the tenth image in sequence", "the image following the ninth image", "the tenth image to come"],
    }
    # add image i to each denotation
    for number, denotation_pool in denotations.items():
        denotation_pool.append(f"image {number}")

    # Default denotation if the image number is not in the specified range
    default_denotation = ["an image", "one of the images", "a picture from the series", "a photograph in the sequence", "a selected image"]
    
    denotation_pool = denotations.get(image_number, default_denotation)
    if is_last:
        if image_number != 1:
            denotation_pool.extend(["the last image", "the final image", "the end image",  "the last image in the series", "the final image in the sequence", "the end image of the series"])
        else:
            denotation_pool = ["the image"]
    return random.choice(denotation_pool)

def get_simple_image_denotation(image_idx:int, is_last:bool=False):
    """
    Generate simple denotation of the images for each image number.
    For example, if the image number is 1, the denotation can be "image 1"
    """
    """
    Generate multiple different forms of denotations of the images for each image number.
    For example, if the image number is 1, the denotation can be "the first image", "the leftmost image", etc
    """
    image_number = image_idx + 1
    # List of various ways to denote an image based on its position
    denotations = {
        1: ["the first image", "the initial image"],
        2: ["the second image", "the secondary image"],
        3: ["the third image",  "the tertiary image"],
        4: ["the fourth image", "the fourth image in sequence"],
        5: ["the fifth image", "the fifth image in sequence"],
        6: ["the sixth image", "the sixth image in sequence"],
        7: ["the seventh image", "the seventh image in sequence"],
        8: ["the eighth image", "the eighth image in sequence"],
        9: ["the ninth image", "the ninth image in sequence"],
        10: ["the tenth image", "the tenth image in sequence"],
    }
    # add image i to each denotation
    for number, denotation_pool in denotations.items():
        denotation_pool.append(f"image {number}")

    # Default denotation if the image number is not in the specified range
    default_denotation = ["an image", "one of the images", "a picture from the series", "a photograph in the sequence", "a selected image"]
    
    denotation_pool = denotations.get(image_number, default_denotation)
    if is_last:
        if image_number != 1:
            denotation_pool.extend(["the last image", "the final image", "the end image",  "the last image in the series", "the final image in the sequence", "the end image of the series"])
        else:
            denotation_pool = ["the image"]
    return random.choice(denotation_pool)


def get_caption_question():
    question_pool = [
        "What do you see in the ",
        "What is in the ",
        "What can you see in the ",
        "What is visible in the ",
        "What details can you identify in the ",
        "What stands out in the ",
        "Can you describe what's in the ",
        "What elements are present in the ",
        "What is depicted in the ",
        "What features are noticeable in the ",
        "How would you describe the contents of the ",
        "What captures your attention in the ",
        "What are the key components of the ",
        "What objects can you spot in the ",
    ]
    
    return random.choice(question_pool)

def get_select_question():
    question_pool = [
        "Which image do you think the caption belongs to?",
        "Which image is the caption describing?",
        "Which image is the caption referring to?",
        "Which image is the caption about?",
        "Which image is the caption related to?",
        "Which image is the caption associated with?",
        "Which image is the caption connected to?",
        "Which image is the caption connected to?",
        "Which image is the caption linked to?",
        "Which image is the caption tied to?",
        "Which image is the caption connected to?",
        "Which image is the caption related to?",
        "Which image is the caption associated with?",
        "Which image is the caption referring to?",
        "Which image is the caption describing?",
        "Which image do you think the caption belongs to?",
    ]
    
    return random.choice(question_pool)

def contrastive_caption_shuffle(captions, images):
    shuffled_idx = list(range(len(images)))
    random.shuffle(shuffled_idx)
    
    conversations = []
    for i in range(len(images)):
        is_last = i == len(images) - 1
        conversations.append(
            {
                "role": "human",
                "content": f"{get_caption_question()}{get_complex_image_denotation(shuffled_idx[i], is_last)}"
            }
        )
        conversations.append(
            {
                "role": "gpt",
                "content": f"{captions[shuffled_idx[i]]}"
            }
        )
    
    if random.random() < 0.1:
        conversations[0]['content'] = f"Here are {len(images)} images: " + ("<image>" * len(images)) + ". " + conversations[0]['content']
    else:
        if random.random() < 0.5:
            conversations[0]['content'] = ("<image> " * len(images)) + conversations[0]['content']
        else:
            conversations[0]['content'] = conversations[0]['content'] + (" <image>" * len(images))
            
            

    return conversations, images

def contrastive_caption_select(captions, images):
    """
    Given image and captions, to select which caption belongs to which image
    """
    shuffled_idx = list(range(len(images)))
    random.shuffle(shuffled_idx)
    
    conversations = []
    for i in range(len(images)):
        is_last = i == len(images) - 1
        conversations.append(
            {
                "role": "human",
                "content": f"{get_select_question()}\n{captions[shuffled_idx[i]]}"
            }
        )
        conversations.append(
            {
                "role": "gpt",
                "content": f"{get_simple_image_denotation(shuffled_idx[i], is_last)}".capitalize()
            }
        )
    if random.random() < 0.1:
        conversations[0]['content'] = f"Here are {len(images)} images: " + ("<image>" * len(images)) + ". " + conversations[0]['content']
    else:
        if random.random() < 0.5:
            conversations[0]['content'] = ("<image> " * len(images)) + conversations[0]['content']
        else:
            conversations[0]['content'] = conversations[0]['content'] + (" <image>" * len(images))
    return conversations, images

contrastive_funcs = [
    contrastive_caption_shuffle,
    contrastive_caption_select
]

def main(
    seed=42,
    dataset_path="Lin-Chen/ShareGPT4V",
    output_file="data/part3_train.json",
    image_save_dir="data",
    shuffle=False,
    max_size=None,
    start_idx=0,
    end_idx=None,
):
    random.seed(seed)
    assert dataset_path == "Lin-Chen/ShareGPT4V", "Only support Lin-Chen/ShareGPT4V dataset"
    dataset = datasets.load_dataset(dataset_path, "ShareGPT4V", split='train')
    
    # def keep_sam_only(item):
    #     return item['image'].startswith("sam/")
    # dataset = dataset.filter(keep_sam_only)

    print(f"Loaded {len(dataset)} items from {dataset_path}")
    if shuffle:
        dataset = dataset.shuffle(seed=seed)
    if max_size:
        dataset = dataset.select(range(max_size))
    
    
    image_save_dir = Path(image_save_dir)    
    image_save_dir.mkdir(parents=True, exist_ok=True)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    all_data = []
    
    if not start_idx:
        start_idx = 0
    if not end_idx:
        end_idx = len(dataset)
    if end_idx and end_idx > 0:
        assert end_idx > start_idx, "end_idx should be greater than start_idx"
        dataset = dataset.select(range(start_idx, end_idx))
        print(f"Selecting from {start_idx} to {end_idx}")

    dataset_iter = iter(dataset)
    idx = start_idx
    
    def map_func(item):
        
    
    
    
    with tqdm(total=len(dataset)) as pbar:
        while True:
            
            if random.random() < 0.3:
                num_selected = 1
            else:
                num_selected = random.randint(2, 8)
            
            items = []
            image_paths = []
            while len(items) < num_selected:
                try:
                    item = next(dataset_iter)
                    
                    image_path = image_save_dir / item['image']
                    if not image_path.exists():
                        idx += 1
                        pbar.update(1)
                        print(f"Skipping {image_path} as it does not exist")
                        continue
                    assert image_path.exists(), f"Image {image_path} does not exist"
                    image = load_image(str(image_path))
                    if not image or image.size[0] < 100 or image.size[1] < 100:
                        idx += 1
                        pbar.update(1)
                        continue
                    image_paths.append(image_path)
                    items.append(item)
                    idx += 1
                    pbar.update(1)
                except StopIteration:
                    break
            
            if len(items) == 0:
                break
                
            image_paths = [str(i.relative_to(output_file.parent)) for i in image_paths]
            captions = [item['conversations'][1]['value'] for item in items]
            
            if num_selected == 1:
                contrastive_func = contrastive_caption_shuffle
            else:
                contrastive_func = random.choice(contrastive_funcs)
            conversations, images = contrastive_func(captions, image_paths)
            
            if conversations:
                all_data.append({
                    "id": f"{dataset_path}-{len(all_data)}-{contrastive_func.__name__}-{idx}-{idx+num_selected}",
                    "images": image_paths,
                    "conversations": conversations,
                })
            else:
                print(f"Skipping {idx}-{idx+num_selected} for it's empty")
                
            
    
    with open(output_file, "w") as f:
        # save to json
        json.dump(all_data, f, indent=4)
        print(f"Saved to {output_file}")
        
        
    
    
if __name__ == "__main__":
    fire.Fire(main)