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
    if split == 'train':
        with open("./data/dvqa/train_qa.json", "r") as f:
            data = json.load(f)
    elif split == 'val':
        with open("./data/dvqa/val_easy_qa.json", "r") as f:
            easy_data = json.load(f)
        with open("./data/dvqa/val_hard_qa.json", "r") as f:
            hard_data = json.load(f)
            
        data = easy_data + hard_data
    else:
        raise ValueError(f"Invalid split: {split}")
    
    output_file = Path(f"./data/dvqa/{split}.json")
    new_data = {}
    for item in tqdm(data, desc="Processing data"):
        image_path = Path(f"./data/dvqa/images/{item['image']}")
        if not image_path.exists():
            print(f"Image not found: {image_path}")
            continue
        if not item['image'] in new_data:
            new_data[item['image']] = {
                "id": f"dvqa_{item['template_id']}_question",
                "images": [str(image_path.relative_to(output_file.parent))],
                "conversations": [
                    {
                        "role": "human",
                        "value": "<image>" + item['question'] if random.random() < 0.5 else item['question'] + "<image>"
                    },
                    {
                        "role": "gpt",
                        "value": item['answer']
                    }
                ]
            }
        else:
            new_data[item['image']]["id"] += f"_{item['question_id']}"
            new_data[item['image']]["conversations"].append({
                "role": "human",
                "value": item['question']
            })
            new_data[item['image']]["conversations"].append({
                "role": "gpt",
                "value": item['answer']
            })
        
    with open(output_file, "w") as f:
        json.dump(list(new_data.values()), f, indent=4, ensure_ascii=False)
        print(f"Saved {len(new_data)} examples to {output_file}")
        
if __name__ == "__main__":
    fire.Fire(main)