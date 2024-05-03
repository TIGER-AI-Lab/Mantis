import fire
import random
import json
from tqdm import tqdm
from pathlib import Path

def main(
    seed=42,
    split='train',
):
    tasks = ['choose_txt', "choose_img", "fill_in_blank"]
    split_data_dir = f"./data/iconqa_data/iconqa/{split}"
    output_file = Path(f"./data/{split}.json")
    all_data = []
    for task in tasks:
        split_task_data_dir = Path(split_data_dir) / task
        folders = [x for x in split_task_data_dir.iterdir() if x.is_dir()]
        for item_dir in tqdm(folders, desc=f"Processing {task} data", total=len(folders)):
            data_file = item_dir / "data.json"
            image = item_dir / "image.png"
            with open(data_file, "r") as f:
                item = json.load(f)
            if task == "choose_img":
                image_choices = [item_dir / x for x in item["choices"]]
                images = [image] + image_choices
                # [str(image_path.relative_to(output_file.parent))],
                images = [str(x.relative_to(output_file.parent)) for x in images]
                question = "<image>" + item['question'] if random.random() < 0.5 else item['question'] + "<image>"
                choices = [f"{chr(65 + i)}. <image>" for i in range(len(image_choices))]
                choices_str = "\n".join(choices)
                answer = f"{chr(65 + item['answer'])}"
                all_data.append({
                    "id": f"iconqa_{split}_{task}_{item['grade']}_{item['label']}_{len(all_data)}",
                    "images": images,
                    "conversations": [
                        {
                            "role": "human",
                            "value": question + "\n" + choices_str
                        },
                        {
                            "role": "gpt",
                            "value": "Answer: " + answer
                        }
                    ]
                }
                )
            elif task == "choose_txt":
                choices = [f"{chr(65 + i)}. {x}" for i, x in enumerate(item["choices"])]
                question = "<image>" + item['question'] if random.random() < 0.5 else item['question'] + "<image>"
                choices_str = "\n".join(choices)
                all_data.append({
                    "id": f"iconqa_{split}_{task}_{item['grade']}_{item['label']}_{len(all_data)}",
                    "images": [str(image.relative_to(output_file.parent))],
                    "conversations": [
                        {
                            "role": "human",
                            "value": question + "\n" + choices_str
                        },
                        {
                            "role": "gpt",
                            "value": "Answer: " + chr(65 + item['answer'])
                        }
                    ]
                }
                )
            elif task == "fill_in_blank":
                question = "<image>" + item['question'] if random.random() < 0.5 else item['question'] + "<image>"
                all_data.append({
                    "id": f"iconqa_{split}_{task}_{item['grade']}_{item['label']}_{len(all_data)}",
                    "images": [str(image.relative_to(output_file.parent))],
                    "conversations": [
                        {
                            "role": "human",
                            "value": question
                        },
                        {
                            "role": "gpt",
                            "value": "Answer: " + item['answer']
                        }
                    ]
                }
                )
    with open(output_file, "w") as f:
        json.dump(all_data, f, indent=4, ensure_ascii=False)
    print(f"Saved {len(all_data)} to {output_file}")
    
if __name__ == '__main__':
    fire.Fire(main)