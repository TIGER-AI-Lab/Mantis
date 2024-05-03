import fire
import pandas as pd
import random
import json
from pathlib import Path

multiple_choice_template1 = """\
Answer the following multiple-choice question:
Here are there images: <image> <image> <image>. If {ref_image_denotation} is the reference image, which image of the other two is more similar to the reference image?
Options:
{options}\
"""

multiple_choice_template2 = """\
Answer the following multiple-choice question:
If {ref_image_denotation} is the reference image, which image of the other two is more similar to the reference image? <image> <image> <image>.
Options:
{options}\
"""

multiple_choice_template3 = """\
Answer the following multiple-choice question:
If <image> is the reference image, which image of the other two images <image> <image> is more similar to the reference image? (Assume reference image is image 1 and the other two are image 2 and image 3)
Options:
{options}\
"""

short_answer_template1 = """\
Answer the following question:
Here are there images: <image> <image> <image>. If {ref_image_denotation} is the reference image, which image of the other two is more similar to the reference image?
"""

short_answer_template2 = """\
Answer the following question:
If {ref_image_denotation} is the reference image, which image of the other two is more similar to the reference image? <image> <image> <image>. (Assume reference image is image 1 and the other two are image 2 and image 3)
"""

short_answer_template3 = """\
Answer the following question:
If <image> is the reference image, which image of the other two images <image> <image> is more similar to the reference image?
"""

def main(
    output_file="./data/train.json",
    image_dir="data/nights",
    split='train',
    seed=42
):
    random.seed(seed)
    output_file = Path(output_file)
    image_dir = Path(image_dir)
    assert image_dir.exists(), f"{image_dir} does not exist."
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv("data/nights/data.csv")
    
    all_data = []
    for i, row in df.iterrows():
        ref_path = row["ref_path"]
        left_path = row["left_path"]
        right_path = row["right_path"]
        left_vote = row["left_vote"]
        right_vote = row["right_vote"]
        prompt = row["prompt"]
        if row['split'] != split:
            continue
        images = [
            image_dir / ref_path,
            image_dir / left_path,
            image_dir / right_path
        ]
        
        if random.random() < 0.5:
            # multiple choice
            multiple_choice_template = random.choice([multiple_choice_template1, multiple_choice_template2, multiple_choice_template3])
            if multiple_choice_template == multiple_choice_template3:
                shuffled_idx = [1, 2]
                random.shuffle(shuffled_idx)
                shuffled_idx = [0] + shuffled_idx
            else:
                shuffled_idx = [0, 1, 2]
                random.shuffle(shuffled_idx)
            better_image_idx = 1 if left_vote > right_vote else 2
            
            shuffled_images = [images[i] for i in shuffled_idx]
            shuffled_images = [str(img.relative_to(output_file.parent)) for img in shuffled_images]
            
            ref_image_id = shuffled_idx.index(0)
            ref_image_denotation = f"image {ref_image_id + 1}"
            better_image_id = shuffled_idx.index(better_image_idx)
            better_image_denotation = f"image {better_image_id + 1}"
            
            options = [f"Image {i + 1}" for i in range(3) if i != ref_image_id]
            random.shuffle(options)
            answer_idx = options.index(better_image_denotation.capitalize())
            answer = chr(ord('A') + answer_idx)
            
            for i, option in enumerate(options):
                options[i] = f"({chr(65+i)}) {option}"       
            options_str = "\n".join(options)
            
            item = {
                "id": "dream_sim_nights_{}".format(i),
                "images": shuffled_images,
                "conversations": [
                    {
                        "role": "human",
                        "content": multiple_choice_template.format(ref_image_denotation=ref_image_denotation, options=options_str)
                    },
                    {
                        "role": "gpt",
                        "content": answer
                    }
                ]
            }
        
        
        else:
            
            short_answer_template = random.choice([short_answer_template1, short_answer_template2, short_answer_template3])
            if short_answer_template == short_answer_template3:
                shuffled_idx = [1, 2]
                random.shuffle(shuffled_idx)
                shuffled_idx = [0] + shuffled_idx
            else:
                shuffled_idx = [0, 1, 2]
                random.shuffle(shuffled_idx)
            better_image_idx = 1 if left_vote > right_vote else 2
            
            shuffled_images = [images[i] for i in shuffled_idx]
            shuffled_images = [str(img.relative_to(output_file.parent)) for img in shuffled_images]
            
            ref_image_id = shuffled_idx.index(0)
            ref_image_denotation = f"image {ref_image_id + 1}"
            better_image_id = shuffled_idx.index(better_image_idx)
            better_image_denotation = f"image {better_image_id + 1}"
            
            response_template_pool = [
                "The image that is more similar to the reference image is {better_image_denotation}.",
                "{better_image_denotation} is more similar to the reference image.",
                "{better_image_denotation}",
            ]
            item = {
                "id": "dream_sim_nights_{}".format(i),
                "images": shuffled_images,
                "conversations": [
                    {
                        "role": "human",
                        "content": short_answer_template.format(ref_image_denotation=ref_image_denotation)
                    },
                    {
                        "role": "gpt",
                        "content": random.choice(response_template_pool).format(better_image_denotation=better_image_denotation).capitalize()
                    }
                ]
            }
        all_data.append(item)
        
    with open(output_file, "w") as f:
        json.dump(all_data, f, indent=4)
        print(f"Saved {len(all_data)} items to {output_file}")
    
if __name__ == "__main__":
    fire.Fire(main)