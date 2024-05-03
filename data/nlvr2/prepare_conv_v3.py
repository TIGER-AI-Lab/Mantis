import json
import random
import fire
from pathlib import Path
from tqdm import tqdm


def load_data(path:str):
    if path.endswith(".json"):
        return json.loads(open(path).read())
    elif path.endswith(".jsonl"):
        return [json.loads(i) for i in open(path).readlines()]
    else:
        raise NotImplementedError
    
def main(
    input_file: str,
    output_file: str,
    image_dir: str,
    seed:int=42,
):
    random.seed(seed)
    input_data = load_data(input_file)
    
    input_file_dir = Path(input_file).parent
    output_file = Path(output_file)
    image_dir = Path(image_dir)
    
    new_data = []
    for item in tqdm(input_data, desc="Processing data"):
        pair_id = item['identifier'][:item['identifier'].rfind("-")]
        images = [image_dir / f"{pair_id}-img0.png", image_dir / f"{pair_id}-img1.png"]
        images = [i.relative_to(input_file_dir) for i in images]
        if not all([(input_file_dir / i).exists() for i in images]):
            print(f"Missing images for {item['identifier']}")
            continue
        images = [str(i) for i in images]
        
        conversation = [
            {
                "role": "human",
                "value": "Here is a statement about the images. Is it true or false?\n" + item['sentence'],
            },
            {
                "role": "gpt",
                "value": "It is true." if item['label'].lower() == "true" else "It is false.",
            }
        ]
        new_data.append({
            "id": item['identifier'],
            "images": images,
            "conversations": conversation,
        })
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        # save to json
        json.dump(new_data, f, indent=4)
        print(f"Saved to {output_file}")
    
if __name__ == "__main__":
    fire.Fire(main)
