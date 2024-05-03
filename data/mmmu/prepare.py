import fire
import json
import regex as re
from datasets import load_dataset
from pathlib import Path

MMMU_subsets = ['Accounting', 'Agriculture', 'Architecture_and_Engineering', 'Art', 'Art_Theory', 'Basic_Medical_Science', 'Biology', 'Chemistry', 'Clinical_Medicine', 'Computer_Science', 'Design', 'Diagnostics_and_Laboratory_Medicine', 'Economics', 'Electronics', 'Energy_and_Power', 'Finance', 'Geography', 'History', 'Literature', 'Manage', 'Marketing', 'Materials', 'Math', 'Mechanical_Engineering', 'Music', 'Pharmacy', 'Physics', 'Psychology', 'Public_Health', 'Sociology']


"""
{
    "question_type": "multi-choice",
    "question": "<image> What change can be observed in the second pair of images?",
    "images": [
        ""
    ],
    "options": [
        "The person with umbrella is gone",
        "The person with red coat is moved",
        "The car up at the top left is now missing"
    ],
    "answer": "B",
    "data_source": "original dataset",
    "category": "difference description"
}
"""

def main(
    output_file="./data/test.json",
    image_output_dir="./data/images",
    split="validation"
):
    if not isinstance(output_file, Path):
        output_file = Path(output_file)
    if not isinstance(image_output_dir, Path):
        image_output_dir = Path(image_output_dir)
    image_output_dir.mkdir(exist_ok=True, parents=True)
    new_data = []
    for subset in MMMU_subsets:
        dataset = load_dataset('MMMU/MMMU', subset)
        dataset = dataset[split]
        for item in dataset:
            images = [item[f"image_{i}"] for i in range(1, 7+1) if f"image_{i}" in item and item[f"image_{i}"] is not None]
            if len(images) > 1:
                question_type = "multi-choice" if len(item['options']) > 1 else "short-answer"
                image_paths = [image_output_dir / f"{item['id']}_{i}.{images[i].format.lower()}" for i in range(len(images))]
                for image, image_path in zip(images, image_paths):
                    if not image_path.exists():
                        image.save(image_path)
                image_paths = [str(image_path.relative_to(output_file.parent)) for image_path in image_paths]
                question = re.sub(r'<image \d>', '<image>', item['question'])
                new_item = {
                    'id': "MMMU_" + item['id'],
                    'question_type': question_type,
                    'question': question,
                    'images': image_paths,
                    'options': eval(item['options']),
                    'answer': item['answer'],
                    'data_source': 'MMMU',
                    'category': "STEM reasoning"
                }
                new_data.append(new_item)
    with open(output_file, 'w') as f:
        json.dump(new_data, f, indent=4)
    
                

if __name__ == '__main__':
    fire.Fire(main)
