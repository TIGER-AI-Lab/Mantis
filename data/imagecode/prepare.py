import fire
import random
import json
import datasets
from tqdm import tqdm
from pathlib import Path

def main(
    seed=42,
    split='train',
):
    random.seed(seed)
    output_file = Path(f"./data/{split}.json")
    all_data = []
    dataset = datasets.load_dataset('BennoKrojer/ImageCoDe')
    for item in tqdm(dataset[split], desc=f"Processing {split} data", total=len(dataset[split])):
        image_dir = Path(f"./data/image-sets/{item['image_set']}")
        images = list(image_dir.glob("*.jpg"))
        image_ids = [int(image.stem[len("img"):]) for image in images]
        sorted_images = sorted(zip(image_ids, images), key=lambda x: x[0])
        image_paths = [str(image.relative_to(output_file.parent)) for _, image in sorted_images]
        
        if random.random() < 0.5:
            question = "Given a detailed description, retrieve the target image among 10 minimally contrastive images"
            question += "\nDescription:\n" + item["description"]
        else:
            question = f"Given this detailed description:\n{item['description']}\nWhich image provided best matches the description?"
        
        if random.random() < 0.5:
            question = "<image>"*len(image_paths) + question if random.random() < 0.5 else question + "<image>"*len(image_paths)
            answer = "Answer: Image " + str(int(item['image_index']) + 1)
        else:
            choices = "\n".join([f"{chr(65 + i)}. <image>" for i in range(len(image_paths))])
            question += f"\n{choices}"
            answer = f"Answer: {chr(65 + int(item['image_index']))}"
        all_data.append({
            "id": f"ImageCoDe-{item['image_set']}-{item['image_index']}",
            "images": image_paths,
            "conversations": [
                {
                    "role": "human",
                    "content": question
                },
                {
                    "role": "gpt",
                    "content": answer
                }
            ]
        })
        
        

    with open(output_file, "w") as f:
        json.dump(all_data, f, indent=4, ensure_ascii=False)
    print(f"Saved {len(all_data)} to {output_file}")
    
if __name__ == '__main__':
    fire.Fire(main)