import datasets
import fire
import hashlib
from PIL import Image
from pathlib import Path


def main(
    dataset_name="TIGER-Lab/VisualWebInstruct",
    save_dir="data"
):
    save_dir = Path(save_dir)
    image_dir = save_dir / "image"
    image_dir.mkdir(parents=True, exist_ok=True)
    def map_subset(item):
        question = item['question']
        answer = item['answer']
        item_id = hashlib.md5((question+answer).encode()).hexdigest()
        image_path = image_dir / f"{item_id}.jpg"
        if not image_path.exists(): 
            item['image'].save(image_path)
        return {
            "id": f"{item['dataset']}_{item_id}",
            "conversation": [
                {
                    "from": "human",
                    "value": question
                },
                {
                    "from": "gpt",
                    "value": answer
                }
            ],
            "images": [
                str(image_path.relative_to(save_dir))
            ]
        }
    
    subset_names = ['forum', 'geometry', 'stemez']
    all_dataset = []
    for subset_name in subset_names:
        subset_dataset = datasets.load_dataset(dataset_name, subset_name, split='train')
        subset_dataset = subset_dataset.map(map_subset, remove_columns=subset_dataset.column_names)
        all_dataset.append(subset_dataset)
    final_dataset = datasets.concatenate_datasets(all_dataset)
    output_file = save_dir/"train.jsonl"
    final_dataset.to_json(output_file)
    print(f"Saved final dataset to {output_file}")
        

if __name__ == "__main__":
    fire.Fire(main)
