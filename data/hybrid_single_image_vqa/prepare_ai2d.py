import fire
from pathlib import Path
from tqdm import tqdm
import json
import random

multiple_choice_template = """\
Answer the following multiple choice question based on the given image.
{question}
{options}
"""

def main(
    seed=42,
):
    random.seed(seed)
    
    question_dir = Path(f"./data/ai2d/questions")
    image_dir = Path(f"./data/ai2d/images")
    output_file = Path(f"./data/ai2d/train.json")
    new_data = []
    for question_file in question_dir.glob("*.json"):
        item = json.load(open(question_file))
        image_path = image_dir / item['imageName']
        if not image_path.exists():
            print(f"Image not found: {image_path}")
            continue
        for question, content in item['questions'].items():
            question = "<image>" + question if random.random() < 0.5 else question + "<image>"
            options_str = "\n".join([f"({chr(65 + i)}) {option}" for i, option in enumerate(content['answerTexts'])])
            answer = chr(content['correctAnswer'] + 65)
            question_after_template = multiple_choice_template.format(question=question, options=options_str)
            new_item = {
                "id": f"ai2d_{question_file.stem}",
                "images": [str(image_path.relative_to(output_file.parent))],
                "conversations": [
                    {
                        "role": "human",
                        "value": question_after_template
                    },
                    {
                        "role": "gpt",
                        "value": answer
                    }
                ]
            }
            new_data.append(new_item)
    
    with open(output_file, "w") as f:
        json.dump(new_data, f, indent=4, ensure_ascii=False)
        print(f"Saved {len(new_data)} examples to {output_file}")
    
if __name__ == "__main__":
    fire.Fire(main)