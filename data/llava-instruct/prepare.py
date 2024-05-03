import json
import fire
import random
from tqdm import tqdm
from pathlib import Path
from transformers import AutoProcessor
from mantis.models.conversation import conv_mllava_v1_mmtag as default_conv_tempalte

def main(
    seed=42,
    input_file="data/llava_v1_5_mix665k.json",
    output_file="data/llava_v1_5_mix665k_multi.json",
    content_length=4096,
    num_image_patches=256,
):
    random.seed(seed)
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    
    with open(input_file, 'r') as f:
        data = json.load(f)

    if not isinstance(input_file, Path):
        input_file = Path(input_file)
    names = {
        "2 images": [
            ["left image", "first image", "image 1", "image one", "image on the left", "image on the left side", "image on the left hand side", "image on the left-hand side"],
            ["right image", "second image", "image 2", "image two", "image on the right", "image on the right side", "image on the right hand side", "image on the right-hand side"]
        ],
        "3 images": [
            ["left image", "first image", "image 1", "image one", "image on the left", "image on the left side", "image on the left hand side", "image on the left-hand side", "initial image"],
            ["middle image", "second image", "image 2", "image two", "image on the middle", "image on the middle side", "central image"],
            ["right image", "third image", "image 3", "image three", "image on the right", "image on the right side", "image on the right hand side", "image on the right-hand side", "final image"]
        ],
    }

    new_data = []
    data_with_image = [x for x in data if 'image' in x]
    data_without_image = [x for x in data if 'image' not in x]
    conv_template = default_conv_tempalte
    roles = {"human": conv_template.roles[0], "gpt": conv_template.roles[1], "user": conv_template.roles[0], "assistant": conv_template.roles[1]}
    i = 0
    with tqdm(total=len(data_with_image)) as pbar:
        while i < len(data_with_image):
            if random.random() < 0.5:
                num_items_merged = 1
            else:
                num_items_merged = random.randint(2, 4)
                
            items_to_merge = data_with_image[i:i+num_items_merged]
            num_items_merged = len(items_to_merge)
            
            if num_items_merged == 1:
                convs = items_to_merge[0]['conversations']
                for c in convs:
                    if c['from'] == "human":
                        c['value'] = c['value'].replace("<image>", "").strip(" \n")
                add_image_token = True 
            else:
                if random.random() < 0.03:
                    convs_pairs = []
                    for item in items_to_merge:
                        for j in range(0, len(item['conversations']), 2):
                            convs_pairs.append(item['conversations'][j:j+2])
                    convs = []
                    for convs_pair in convs_pairs:
                        convs += convs_pair
                    add_image_token = False # use preexisting image tokens 
                else:
                    for j, item in enumerate(items_to_merge):
                        if len(items_to_merge) == 2:
                            to_replace_names = names["2 images"][j]
                        elif len(items_to_merge) == 3:
                            to_replace_names = names["3 images"][j]
                        else:
                            to_replace_names = [f"image {j+1}"]
                        for c in item['conversations']:
                            if c['from'] == "human":
                                c['value'] = c['value'].replace("<image>", "")
                                if "image" in c:
                                    to_replace_name = random.choice(to_replace_names)
                                    c['value'] = c['value'].replace("image", to_replace_name)
                                else:
                                    to_replace_name = random.choice(to_replace_names)
                                    # make the first letter lowercase
                                    c['value'] = c['value'].strip(" \n")
                                    for k, char in enumerate(c['value']):
                                        if char.isalpha():
                                            break
                                    c['value'] = c['value'][:k] + c['value'][k].lower() + c['value'][k+1:]
                                    c['value'] = "For the {}, {}".format(to_replace_name, c['value'])
                                    
                    convs_pairs = []
                    for item in items_to_merge:
                        for j in range(0, len(item['conversations']), 2):
                            convs_pairs.append(item['conversations'][j:j+2])
                    convs = []
                    random.shuffle(convs_pairs)
                    for convs_pair in convs_pairs:
                        convs += convs_pair
                    add_image_token = True
                    
                    
            
            if not all([(input_file.parent / x['image']).exists() for x in items_to_merge]):
                print("Cannot find image files {}".format([x['image'] for x in items_to_merge]))
            else:
                conv_template.messages = []
                for j, sentence in enumerate(convs):
                    role = roles[sentence.get("from", sentence.get("role"))]
                    assert role == conv_template.roles[j % 2], f"{i}"
                    conv_template.append_message(role, sentence.get("content", sentence.get("text", sentence.get("value", ""))))
                    
                prompt = conv_template.get_prompt()
                prompt_len = len(processor.tokenizer.encode(prompt))
                prompt_len_with_image = prompt_len + num_image_patches * len(items_to_merge)
                if prompt_len_with_image > content_length:
                    num_parts = prompt_len_with_image // content_length + 1
                    num_convs_per_part = len(convs) // num_parts + 1
                    # make sure num_convs_per_part is even
                    num_convs_per_part = num_convs_per_part + 1 if num_convs_per_part % 2 == 1 else num_convs_per_part
                    convs_parts = [convs[i:i+num_convs_per_part] for i in range(0, len(convs), num_convs_per_part)]
                    print(f"Warning: the context length is {prompt_len_with_image}, which is larger than {content_length}, dividing the conversation into {num_parts} parts")
                    for j, convs_part in enumerate(convs_parts):
                        if add_image_token:
                            if random.random() < 0.5:
                                convs_part[0]['value'] = "<image> " * num_items_merged + convs_part[0]['value']
                            else:
                                convs_part[0]['value'] += " <image>" * num_items_merged
                        new_item = {
                            "id": "llava_665k_multi_{}_part{}".format("-".join([str(x["id"]) for x in items_to_merge]), j),
                            "images": [x["image"] for x in items_to_merge],
                            "conversations": convs_part,
                        }
                        new_data.append(new_item)
                else:
                    if add_image_token:
                        if random.random() < 0.5:
                            convs[0]['value'] = "<image> " * num_items_merged + convs[0]['value']
                        else:
                            convs[0]['value'] += " <image>" * num_items_merged
                    new_item = {
                        "id": "llava_665k_multi_{}".format("-".join([str(x["id"]) for x in items_to_merge])),
                        "images": [x["image"] for x in items_to_merge],
                        "conversations": convs,
                    }
                    new_data.append(new_item)
            i += num_items_merged
            pbar.update(num_items_merged)
            
    for i, item in enumerate(data_without_image):
        del item['model']
        item['id'] = "llava_665k_no_image_{}".format(item['id'])
        item['images'] = []
    new_merged_data = new_data + data_without_image
    
    # postprocess the data to ensure the context length be in 4096
    
    random.shuffle(new_merged_data)
    
    with open(output_file, 'w') as f:
        json.dump(new_data, f, indent=4)
        
if __name__ == "__main__":
    fire.Fire(main)