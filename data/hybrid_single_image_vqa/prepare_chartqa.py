import fire
import json
import random
from pathlib import Path
from tqdm import tqdm

def main(
    seed=42,
    split='train',
):
    random.seed(seed)
    augmented_data = json.load(open(f'./data/chartqa/ChartQA Dataset/{split}/{split}_augmented.json'))
    human_data = json.load(open(f'./data/chartqa/ChartQA Dataset/{split}/{split}_human.json'))
    output_file = Path(f"./data/chartqa/{split}.json")
    image_dir = Path(f"./data/chartqa/{split}_images")
    all_data = augmented_data + human_data
    new_data = []
    for i, item in tqdm(enumerate(all_data), desc="Processing data", total=len(all_data)):
        image_path = image_dir / item['imgname']
        if not Path(image_path).exists():
            print(f"Image not found: {image_path}")
            continue
        new_data.append({
            "id": f"chartqa_{i}",
            "images": [str(image_path.relative_to(output_file.parent))],
            "conversations": [
                {
                    "role": "human",
                    "value": "<image>" + item['query'] if random.random() < 0.5 else item['query'] + "<image>"
                },
                {
                    "role": "gpt",
                    "value": item['label']
                }
            ]
        })
    with open(output_file, "w") as f:
        json.dump(new_data, f, indent=4, ensure_ascii=False)
    print(f"Saved to {output_file}")
    
if __name__ == "__main__":
    fire.Fire(main)