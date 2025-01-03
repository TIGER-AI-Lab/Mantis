import json
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset

all_data=[]

remove_image_placeholders_in_prompt = True

anno_data=load_dataset("TIGER-Lab/VideoFeedback",name="annotated",split="train")
for idx,item in tqdm(enumerate(anno_data)):
    # "p110367_22.jpg"
    item['images'] = ["images/" + item['images'][0].split("_")[0] + "/" + image for i, image in enumerate(item['images'])]
    assert all([Path(image).exists() for image in item['images']]), item['images']
    labels = [x for x in item['conversations'][1]['value'].split("\n") if x]
    labels = {label.split(":")[0].strip(' \n'): float(label.split(":")[1]) for label in labels}
    prompt = item['conversations'][0]['value']
    if remove_image_placeholders_in_prompt:
        prompt = prompt[:prompt.find("all the frames of video are as follows:")+len("all the frames of video are as follows:")].strip(' \n') + "\n"
    
    all_data.append({
        "id": item['id'],
        "images": item['images'],
        "prompt": prompt,
        "labels": labels
    })
  
    
## please comment the part below if you only need the annotated data    
real_data=load_dataset("TIGER-Lab/VideoFeedback",name="real",split="train")   
for idx,item in tqdm(enumerate(real_data)):
    # "p110367_22.jpg"
    item['images'] = ["images/" + item['images'][0].split("_")[0] + "/" + image for i, image in enumerate(item['images'])]
    assert all([Path(image).exists() for image in item['images']]), item['images']
    labels = [x for x in item['conversations'][1]['value'].split("\n") if x]
    labels = {label.split(":")[0].strip(' \n'): float(label.split(":")[1]) for label in labels}
    prompt = item['conversations'][0]['value']
    if remove_image_placeholders_in_prompt:
        prompt = prompt[:prompt.find("all the frames of video are as follows:")+len("all the frames of video are as follows:")].strip(' \n') + "\n"
    all_data.append({
        "id": item['id'],
        "images": item['images'],
        "prompt": prompt,
        "labels": labels
    })

with open("./train_regression.json", "w") as f:
    json.dump(all_data, f, indent=4)