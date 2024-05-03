import requests
import fire
import json
import os
import pandas as pd
import numpy as np
import regex as re
from io import BytesIO
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from typing import List
from easy_openai import openai_completions, _chatml_to_prompt
from string import Template

CONV_TEMPLATE = """
Here is a response of a question about two bird images. Please generate me one possible question based on this response. 

Response: ${response}
"""

VQA_TEMPLATE = """\
I am transforming the task type of a dataset. 
The original dataset askes models to take 2 bird images as the input and asks it to generate a text describing the difference between them. 
The targeted transformed dataset be a VQA task that takes 2 images and a question about the differnce, and the answer should be multi-choice. 

You are provided the groudth truth description of the two bird images:
${difference}

Now generate the question and options/anwer in multi-choice format:

Overall output format:
{
"question": "...",
"options": [ "A: {option A}", ...],
"answer": "{A or B or ...}" 
}
"""

def load_image(image_file):
    try:
        if image_file.startswith("http"):
            response = requests.get(image_file, timeout=1)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            import os
            image = Image.open(image_file).convert("RGB")
        return image
    except Exception as e:
        return None

def load_bird_image(url):
    image_postfixs = ["JPG", "jpg", "jpeg", "png"]
    url_template = "https://inaturalist-open-data.s3.amazonaws.com/photos/{}/medium.{}"
    url_template2 = "https://static.inaturalist.org/photos/{}/medium.{}"
    image_id = url.split("/")[-1]
    image = None
    for postfix in image_postfixs:
        image = load_image(url_template.format(image_id, postfix))
        if image is not None:
            break
    if image is None:
        for postfix in image_postfixs:
            image = load_image(url_template2.format(image_id, postfix))
            if image is not None:
                break
    if image is None:
        print("Could not load image", url)
    return image

def save_images(raw_data, image_output_dir, output_dir):
    image_paths = []
    for i, row in tqdm(enumerate(raw_data), total=len(raw_data), desc="Saving Images"):
        image1_id = row['img1ObservationURL'].split("/")[-1]
        image2_id = row['img2ObservationURL'].split("/")[-1]
        image1_path = image_output_dir / "{}.jpg".format(image1_id)
        image2_path = image_output_dir / "{}.jpg".format(image2_id)
        if image1_path.exists() and image2_path.exists():
            images = [str(image1_path.relative_to(output_dir)), str(image2_path.relative_to(output_dir))]
        else:
            image1 = load_bird_image(row['img1ObservationURL'])
            image2 = load_bird_image(row['img2ObservationURL'])
            if image1 is None or image2 is None:
                images = None
            else:
                image1.save(image1_path), image2.save(image2_path)
                images = [str(image1_path.relative_to(output_dir)), str(image2_path.relative_to(output_dir))]
        image_paths.append(images)
    
    print("{}/{} images are missing".format(len([x for x in image_paths if x is None]), len(image_paths)))
    raw_data = [x for i, x in enumerate(raw_data) if image_paths[i] is not None]
    image_paths = [x for x in image_paths if x is not None]
    return raw_data, image_paths
    
    
def save_conv_data(raw_data, image_output_dir, output_file):
    if not isinstance(output_file, Path):
        output_file = Path(output_file)
    if not isinstance(image_output_dir, Path):
        image_output_dir = Path(image_output_dir)
    output_dir = output_file.parent
    
    raw_data, image_paths = save_images(raw_data, image_output_dir, output_dir)
        
    # generate convs
    tempalte = Template(CONV_TEMPLATE)
    def map_item_to_prompt(item):
        return _chatml_to_prompt([{
            "role": "user",
            "content": tempalte.substitute(response=item['description']),
        }])
    prompts = [map_item_to_prompt(item) for item in raw_data]
    results = openai_completions(
        prompts=prompts,
        model_name="ChatGPT",
        max_tokens=256,
        temperature=0.0,
        top_p=1.0,
    )
    completions = results['completions']
    total_price = sum(results['price_per_example'])
    print(f"Total price: {total_price} for {len(completions)} examples")
    
    # save convs
    new_data = []
    for i, row in tqdm(enumerate(raw_data), total=len(raw_data), desc="Saving Conv data"):
        images = image_paths[i]
        conv_inst = completions[i]
        response = row['description']
        new_item = {
            "id": "birds-to-words-{}".format(i),
            "images": images,
            "conversations": [
                {
                    "role": "human",
                    "value": conv_inst,
                },
                {
                    "role": "gpt",
                    "value": response,
                }
            ]
        }
        new_data.append(new_item)
    
    # deduplicate by images
    dedup_vqa_data = []
    image_set = set()
    for item in new_data:
        if str(item['images']) not in image_set:
            dedup_vqa_data.append(item)
            image_set.add(str(item['images']))
    new_data = dedup_vqa_data

    with open(output_file, "w") as f:
        json.dump(new_data, f, indent=4)
        print("Saved to", output_file)

def save_vqa_data(raw_data, image_output_dir, output_file):
    if not isinstance(output_file, Path):
        output_file = Path(output_file)
    if not isinstance(image_output_dir, Path):
        image_output_dir = Path(image_output_dir)
    output_dir = output_file.parent
    
    raw_data, image_paths = save_images(raw_data, image_output_dir, output_dir)
    
    # generate vqa
    tempalte = Template(VQA_TEMPLATE)
    def map_item_to_prompt(item):
        return _chatml_to_prompt([{
            "role": "user",
            "content": tempalte.substitute(difference=item['description']),
        }])
    prompts = [map_item_to_prompt(item) for item in raw_data]
    results = openai_completions(
        prompts=prompts,
        model_name="ChatGPT",
        max_tokens=256,
        temperature=0.0,
        top_p=1.0,
    )
    completions = results['completions']
    total_price = sum(results['price_per_example'])
    print(f"Total price: {total_price} for {len(completions)} examples")
    
    new_vqa_data = []
    for i, row in tqdm(enumerate(raw_data), total=len(raw_data), desc="Saving VQA data"):
        images = image_paths[i]
        completion = completions[i]
        # multi_choice = re.search(r'(?<=Multi-choice:)((.|\n)*?)', completion, re.DOTALL)
        completion = completion[completion.find("{"):completion.rfind("}")+1]
        try:
            output_pure = json.loads(completion)
            
            assert output_pure['answer'] in ['A', 'B', 'C', 'D'], \
                "answer should be one of A, B, C, D, but got {answer}".format(answer=output_pure['answer'])
            answer_idx = ord(output_pure['answer']) - ord('A')
            for j in range(len(output_pure['options'])):
                if re.match(r'^[A-D]:', output_pure['options'][j]):
                    output_pure['options'][j] = output_pure['options'][j][2:].strip()
            
            answer = output_pure['options'][answer_idx]
            output_pure['options'] = np.random.permutation(output_pure['options']).tolist()
            answer_idx = output_pure['options'].index(answer)
            output_pure['answer'] = chr(ord('A') + answer_idx)

            new_vqa_data.append({
                "id": "birds-to-words-{}".format(i),
                "question_type": "multi-choice",
                "question": output_pure['question'],
                "images": images,
                "options": output_pure['options'],
                "answer": output_pure['answer'],
                "data_source": "birds-to-words",
                "category": "difference description"
            })
        except Exception as e:
            print(e)
            
    # deduplicate by images
    dedup_vqa_data = []
    image_set = set()
    for item in new_vqa_data:
        if str(item['images']) not in image_set:
            dedup_vqa_data.append(item)
            image_set.add(str(item['images']))
    new_vqa_data = dedup_vqa_data
            
    with open(output_file, "w") as f:
        json.dump(new_vqa_data, f, indent=4)
        print("Saved to", output_file)
    
    
def main(
    input_file: str,
    image_output_dir: str,
    seed:int=42,
):
    np.random.seed(seed)  
    data = pd.read_csv(input_file, sep="\t")
    image_output_dir = Path(image_output_dir)
    image_output_dir.mkdir(exist_ok=True, parents=True)
    
    train_raw_data = [x for i, x in data.iterrows() if x['split'] == 'train']
    val_raw_data = [x for i, x in data.iterrows() if x['split'] == 'val']
    test_raw_data = [x for i, x in data.iterrows() if x['split'] == 'test']
    
    train_file = "./train.json"
    val_file = "./val.json"
    test_file = "./test.json"
    
    print("Saving train conv data")
    save_conv_data(train_raw_data, image_output_dir, train_file)
    print("Saving val conv data")
    save_conv_data(val_raw_data, image_output_dir, val_file)
    print("Saving test vqa data")
    save_vqa_data(test_raw_data, image_output_dir, test_file)
    


if __name__ == '__main__':
    fire.Fire(main)