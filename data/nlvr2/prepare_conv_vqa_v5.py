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
    

multiple_choice_template1 = """\
Answer the following multiple-choice question:
Given these 2 images, <image> and <image>. Here is a statement describing them: {sentence} Is it true or false?
Options:
{options}\
"""

multiple_choice_template2 = """\
Answer the following multiple-choice question:
<image> <image> Here is a statement describing these 2 images: {sentence} Is it true or false? 
Options:
{options}\
"""

multiple_choice_template3 = """\
Answer the following multiple-choice question:
Here is a statement describing 2 images: {sentence} Is it true or false? <image> <image> 
Options:
{options}\
"""


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
            
        options = ["True", "False"]
        random.shuffle(options)
        for i, option in enumerate(options):
            options[i] = f"({chr(65+i)}) {option}"
        true_option = "A" if options[0].endswith("True") else "B"
        false_option = "A" if options[0].endswith("False") else "B"
        assert true_option != false_option
        answer = true_option if item['label'].lower() == "true" else false_option
        multiple_choice_template = random.choice([multiple_choice_template1, multiple_choice_template2, multiple_choice_template3])
        
        conversation = [
            {
                "role": "human",
                "value": multiple_choice_template.format(sentence=item['sentence'], options="\n".join(options))
            },
            {
                "role": "gpt",
                "value": answer
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
