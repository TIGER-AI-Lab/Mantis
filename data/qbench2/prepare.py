import fire
import json
import random
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple


# {"type": 2, "concern": 0, "question": "Compared to the first image, how is the clarity of the second image?", "img_path": "llvisionqa_compare_dev\\00079.jpg_cat_09769.jpg.jpg", "candidates": ["More blurry", "Clearer", "About the same"], "correct_ans": "Clearer", "correct_choice": "B"}


"""
{
        "id": "cw_19",
        "question_type": "multi-choice",
        "question": "<image> <image> Which image contains more advanced technology?",
        "images": [
            "images/215_0.jpeg",
            "images/215_1.jpeg"
        ],
        "options": [
            "(A) Image 1",
            "(B) Image 2",
            "(C) Both images contain the same level of technology"
        ],
        "answer": "B",
        "data_source": ".."
        "category": "technological aspects in images"
    },
"""
def main(
    seed=42,
    image_mode="pair",
    split='dev',
):
    random.seed(seed)
    
    assert image_mode in ["pair", "single"], image_mode
    with open(f"data/q-bench2-a1-{split}.jsonl", "r") as f:
        data = [json.loads(line) for line in f]
    image_dir = Path(f"data/all_single_images") if image_mode == "pair" else Path(f"data/llvisionqa_compare_{split}")
    output_file = Path(f"data/q-bench2-a1-{image_mode}-{split}.json")
    new_data = []
    for i, q in enumerate(data):
        if image_mode == "pair":
            # "llvisionqa_compare_dev\\00079.jpg_cat_09769.jpg.jpg"
            # -> ["00079.jpg", "09769.jpg"]
            img_paths = q["img_path"].split("\\")[1][:-len(".jpg")].split("_cat_")
            img_paths = [image_dir / img_path for img_path in img_paths]
            assert all(img_path.exists() for img_path in img_paths), img_paths
            question = q["question"]
            options = "\n".join([f"{chr(65 + i)}. {c}" for i, c in enumerate(q["candidates"])])
            answer = q["correct_choice"] if "correct_choice" in q else None
            
            new_data.append({
                "id": "qbench2-a1-" + str(i),
                "question_type": "multi-choice",
                "question": question,
                "images": [str(img_path.relative_to(output_file.parent)) for img_path in img_paths],
                "options": [f"({chr(65 + i)}) {c}" for i, c in enumerate(q["candidates"])],
                "answer": answer,
                "data_source": "q-bench2-a1-" + image_mode + "-" + split,
                "category": "low level visual comparison",
            })
        elif image_mode == "single":
            img_path = image_dir / q["img_path"].split("\\")[1]
            assert img_path.exists(), img_path
            question = q["question"]
            options = "\n".join([f"{chr(65 + i)}. {c}" for i, c in enumerate(q["candidates"])])
            answer = q["correct_choice"] if "correct_choice" in q else None
            
            new_data.append({
                "id": "qbench2-a1-" + str(i),
                "question_type": "multi-choice",
                "question": question,
                "images": [str(img_path.relative_to(output_file.parent))],
                "options": [f"({chr(65 + i)}) {c}" for i, c in enumerate(q["candidates"])],
                "answer": answer,
                "data_source": "q-bench2-a1-" + image_mode + "-" + split,
                "category": "low level visual comparison",
            })
        else:
            raise ValueError(image_mode)
    with open(output_file, "w") as f:
        json.dump(new_data, f, indent=4)
    print(f"Saved {len(new_data)} questions to {output_file}")
            
if __name__ == "__main__":
    fire.Fire(main)
