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
        response = requests.get(url, timeout=1)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except Exception as e:
        # print(f"Failed to download image from {url}: {e}")
        return None

def process_example(example, to_save_image_dir, output_file, index):
    image_ids = [hashlib.sha256(image.encode()).hexdigest() for image in example["images"] if image]
    if not len(image_ids) >= 2:
        # print(f"Example {index} does not have enough images")
        return False
    image_paths = [to_save_image_dir / f"{image_id}.jpg" for image_id in image_ids]
    if not all([image_path.exists() for image_path in image_paths]):
        downloaded_images = []
        for image in example["images"]:
            if image:
                image_id = hashlib.sha256(image.encode()).hexdigest()
                image_path = to_save_image_dir / f"{image_id}.jpg"
                if not image_path.exists():
                    downloaded_image = get_http_image(image)
                    if downloaded_image:
                        downloaded_images.append(downloaded_image)
        if not len(downloaded_images) == len(image_paths):
            # print(f"Failed to download all images for example {index}")
            return False
        for image, image_path in zip(downloaded_images, image_paths):
            try:
                image.save(image_path)
            except Exception as e:
                try:
                    image.save(image_path, format="jpg")
                except Exception as e:
                    # print(f"Failed to save image {image_path}: {e}")
                    return False
    relative_image_paths = [image_path.relative_to(output_file.parent) for image_path in image_paths]
    text = ""
    image_metadata = json.loads(example["metadata"])
    for i in range(len(example["texts"])):
        if example["texts"][i]:
            text += example["texts"][i] + " "
        elif example["images"][i]:
            text += "<image> "
            if "alt_text" in image_metadata[i] and image_metadata[i]['alt_text']:
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
    
    return True   

def main(
    seed=42,
    down_sampling_size=100_000,
    output_file="./data/train.jsonl",
    to_save_image_dir="./data/images/",
    max_workers=16,
    chunk_size=5000,
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
    
    total_size = 0
    iter_dataset = iter(shuffled_dataset)
    num_processed_chunks = 0
    while total_size < down_sampling_size:
        chunk_examples = []
        for i in range(chunk_size):
            try:
                chunk_examples.append(next(iter_dataset))
            except StopIteration:
                break
        if not chunk_examples:
            break
    
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(executor.map(lambda example: process_example(example, to_save_image_dir, output_file, i), chunk_examples), total=len(chunk_examples), desc=f"Processing chunk-{num_processed_chunks} examples"))
        
        total_size += sum(results)
        num_processed_chunks += 1
        
        
        
        
        
    # with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    #     futures = []
    #     for i, example in tqdm(enumerate(shuffled_dataset), desc="Submitting tasks to executor"):
    #         futures.append(executor.submit(process_example, example, to_save_image_dir, output_file, i))
    #         if len(futures) >= chunk_size:
    #             for future in tqdm(
    #                 concurrent.futures.as_completed(futures), total=len(futures), desc="Waiting for futures to complete", leave=False
    #             ):
    #                 future.result()
    #                 total_size += 1
    #                 if total_size >= down_sampling_size:
    #                     break
    #             futures = []
    #         if total_size >= down_sampling_size:
    #             break
        
    
if __name__ == "__main__":
    fire.Fire(main)