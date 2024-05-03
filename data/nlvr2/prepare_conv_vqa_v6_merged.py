import json
import random
import fire
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from typing import List

def load_data(path:str):
    if path.endswith(".json"):
        return json.loads(open(path).read())
    elif path.endswith(".jsonl"):
        return [json.loads(i) for i in open(path).readlines()]
    else:
        raise NotImplementedError
    

multiple_choice_template1 = """\
Answer the following multiple-choice question:
Given these 2 images, <image>. Here is a statement describing them: {sentence} Is it true or false?
Options:
{options}\
"""

multiple_choice_template2 = """\
Answer the following multiple-choice question:
<image> Here is a statement describing these 2 images: {sentence} Is it true or false? 
Options:
{options}\
"""

multiple_choice_template3 = """\
Answer the following multiple-choice question:
Here is a statement describing 2 images: {sentence} Is it true or false? <image>
Options:
{options}\
"""

def load_images(image_paths: List[str]):
    images = []
    for image_path in image_paths:
        image = Image.open(image_path)
        images.append(image)
    return images

def merge_images(image_links: List = []):
    """Merge multiple images into one image

    Args:
        image_links (List, optional): List of image links. Defaults to [].

    Returns:
        [type]: [description]
    """
    if len(image_links) == 0:
        return None
    images = load_images(image_links)
    if len(images) == 1:
        return images[0]
    widths, heights = zip(*(i.size for i in images))
    average_height = sum(heights) // len(heights)
    for i, im in enumerate(images):
        # scale in proportion
        images[i] = im.resize((int(im.size[0] * average_height / im.size[1]), average_height))
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)
    new_im = Image.new("RGB", (total_width + 10 * (len(images) - 1), max_height))
    x_offset = 0
    for i, im in enumerate(images):
        if i > 0:
            # past a column of 1 pixel starting from x_offset width being black, 8 pixels being white, and 1 pixel being black
            new_im.paste(Image.new("RGB", (1, max_height), (0, 0, 0)), (x_offset, 0))
            x_offset += 1
            new_im.paste(Image.new("RGB", (8, max_height), (255, 255, 255)), (x_offset, 0))
            x_offset += 8
            new_im.paste(Image.new("RGB", (1, max_height), (0, 0, 0)), (x_offset, 0))
            x_offset += 1
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    return new_im



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
    merge_image_dir = image_dir.parent / image_dir.name.replace("images", "merged_images")
    merge_image_dir.mkdir(parents=True, exist_ok=True)
    new_data = []
    for item in tqdm(input_data, desc="Processing data"):
        pair_id = item['identifier'][:item['identifier'].rfind("-")]
        images = [image_dir / f"{pair_id}-img0.png", image_dir / f"{pair_id}-img1.png"]
        images = [i.relative_to(input_file_dir) for i in images]
        to_save_merged_image_path = merge_image_dir / f"{pair_id}.png"
        if not to_save_merged_image_path.exists():
            if not all([(input_file_dir / i).exists() for i in images]):
                print(f"Missing images for {item['identifier']}")
                continue
            # load images
            merged_image = merge_images([input_file_dir / i for i in images])
            # resize, make them at most 1280 * 720
            if merged_image.size[0] > 1280:
                merged_image = merged_image.resize((1280, int(1280 / merged_image.size[0] * merged_image.size[1])))
            if merged_image.size[1] > 720:
                merged_image = merged_image.resize((int(720 / merged_image.size[1] * merged_image.size[0]), 720))
            merged_image.save(merge_image_dir / f"{pair_id}.png")
        images = [str((merge_image_dir / f"{pair_id}.png").relative_to(input_file_dir))]
            
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
