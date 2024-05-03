import fire
import random
import json
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image
from io import BytesIO

def main(
    dataset_id="pixparse/docvqa-single-page-questions",
    split="train",
    to_save_image_dir="./data/docvqa/images",
    seed=42,
):
    random.seed(seed)
    dataset = load_dataset(dataset_id)
    
    to_save_image_dir = Path(to_save_image_dir)
    to_save_image_dir.mkdir(exist_ok=True, parents=True)
    new_data = []
    output_file = f"./data/docvqa/{split}.json"
    output_file = Path(output_file)
    for item in tqdm(dataset[split], desc="Processing train data"):
        
        image = item["image"]
        image_path = to_save_image_dir / item['other_metadata']['image']
        if not image_path.exists():
            image.save(image_path)
        image_path_str = str(image_path.relative_to(output_file.parent))
        new_data.append({
            "id": f"docvqa_{item['other_metadata']['ucsf_document_id']}_question_{item['question_id']}",
            "images": [image_path_str],
            "conversations": [
                {
                    "role": "human",
                    "value": "<image>" + item['question'] if random.random() < 0.5 else item['question'] + "<image>"
                },
                {
                    "role": "gpt",
                    "value": random.choice(item['answers'])
                }
            ]
        })
    with open(output_file, "w") as f:
        json.dump(new_data, f, indent=4, ensure_ascii=False)
    print(f"Saved to {output_file}")
    

if __name__ == "__main__":
    fire.Fire(main)
    