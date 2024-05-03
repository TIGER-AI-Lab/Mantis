import fire
import json
import hashlib
import random
import regex as re
from pathlib import Path
from tqdm import tqdm


def main(
    seed: int = 42,
):
    random.seed(42)
    with open("./data/coinstruct_562k_llava_format.json", 'r') as f:
        data = json.load(f)
    image_dir = Path("./data/images")
    output_file = Path("./data/train.json")
    new_data = {}
    for item in tqdm(data):
        if isinstance(item['image'], str):
            item['image'] = [item['image']]
        num_images = len(item['image'])
        images_id = hashlib.md5("".join(item['image']).encode()).hexdigest()
        image_paths = [image_dir / img for img in item['image']]
        if images_id not in new_data:
            new_data[images_id] = {
                "id": f"coinstruct_{len(new_data)}",
                "images": [str(img.relative_to(output_file.parent)) for img in image_paths],
                "conversations": [],
            }
        new_data[images_id]["conversations"].extend(item['conversations'])
        
    for item in new_data.values():
        if random.random() < 0.1:
            for conv in item['conversations'][1:]:
                
                if conv['from'] == "human":
                    conv['value'] = conv['value'].replace("The first image:", "")
                    conv['value'] = conv['value'].replace("The second image:", "")
                    conv['value'] = conv['value'].replace("The third image:", "")
                    conv['value'] = conv['value'].replace("The fourth image:", "")
                    conv['value'] = conv['value'].replace("<image>", "")
                conv['value'] = conv['value'].strip('\n ')
        else:
            for conv in item['conversations']:
                
                if conv['from'] == "human":
                    conv['value'] = conv['value'].replace("The first image:", "")
                    conv['value'] = conv['value'].replace("The second image:", "")
                    conv['value'] = conv['value'].replace("The third image:", "")
                    conv['value'] = conv['value'].replace("The fourth image:", "")
                    conv['value'] = conv['value'].replace("<image>", "")
                conv['value'] = conv['value'].strip('\n ')
            if random.random() < 0.5:
                item['conversations'][0]['value'] += " <image>" * len(item['images'])
            else:
                item['conversations'][0]['value'] = "<image> " * len(item['images']) + item['conversations'][0]['value']
            
    with open(output_file, 'w') as f:
        json.dump(list(new_data.values()), f, indent=4)
        print(f"Saved to {len(new_data)} samples to {output_file}")

if __name__ == "__main__":
    fire.Fire(main)