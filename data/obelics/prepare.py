import fire
import datasets
import json
import hashlib
import requests
import concurrent
import os
from PIL import Image
from io import BytesIO
from pathlib import Path
from tqdm import tqdm

def get_http_image(url):
    try:
        response = requests.get(url, timeout=2)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except Exception as e:
        # print(f"Failed to download image from {url}: {e}")
        return None

def process_example(example, to_save_image_dir, output_file, index):
    image_ids = [hashlib.sha256(image.encode()).hexdigest() for image in example["images"] if image]
    image_paths = [to_save_image_dir / f"{image_id}.png" for image_id in image_ids]
    if not all([image_path.exists() for image_path in image_paths]):
        downloaded_images = []
        for image in example["images"]:
            if image:
                image_id = hashlib.sha256(image.encode()).hexdigest()
                image_path = to_save_image_dir / f"{image_id}.png"
                if not image_path.exists():
                    downloaded_image = get_http_image(image)
                    if downloaded_image:
                        downloaded_images.append(downloaded_image)
        if not len(downloaded_images) == len(image_paths):
            # print(f"Failed to download all images for example {index}")
            return
        for image, image_path in zip(downloaded_images, image_paths):
            image.save(image_path)
    relative_image_paths = [image_path.relative_to(output_file.parent) for image_path in image_paths]
    text = ""
    image_metadata = json.loads(example["metadata"])
    for i in range(len(example["texts"])):
        if example["texts"][i]:
            text += example["texts"][i] + " "
        elif example["images"][i]:
            text += "<image> "
            if image_metadata[i]['alt_text']:
                text += f"({image_metadata[i]['alt_text']}) "
    text = text.strip()
            
    new_example = {
        "id": "obelics_" + str(index),
        "images": [str(image_path) for image_path in relative_image_paths],
        "conversations": [
            {
                "role": "user",
                "content": None,
            },
            {
                "role": "assistant",
                "content": text,
            }
        ]
    }
    with open(output_file, "a+") as f:
        f.write(json.dumps(new_example) + "\n")        

def main(
    seed=42,
    down_sampling_size=300_000,
    output_file="./data/train.jsonl",
    to_save_image_dir="./data/images/",
    max_workers=8,
):
    
    dataset = datasets.load_dataset("HuggingFaceM4/OBELICS", split="train", streaming=True)
    shuffled_dataset = dataset.shuffle(seed=seed, buffer_size=10_000)
    
    output_file = Path(output_file)
    to_save_image_dir = Path(to_save_image_dir)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    to_save_image_dir.mkdir(parents=True, exist_ok=True)
    
    # clear the output file
    with open(output_file, "w") as f:
        pass
    

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i, example in tqdm(enumerate(shuffled_dataset), total=down_sampling_size, desc="Submitting tasks to executor"):
            
            if i >= down_sampling_size:
                break
            
            futures.append(executor.submit(process_example, example, to_save_image_dir, output_file, i))
        
        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            future.result()
    
    
if __name__ == "__main__":
    fire.Fire(main)