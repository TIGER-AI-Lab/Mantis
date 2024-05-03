import json
import os
import random
import fire
from collections import Counter


def read_data(with_corrdinates=True, with_chart_data=True) -> dict[str: list[tuple[str, str]]]:
    '''
    read in the data and return it in a dictionary where:
    
    key: the image id, ex: "12345", this corresponds to "12345.jpg" in the data/image forlder
    value: a list of tuple in the form of (question, answer)
    '''
    if with_corrdinates:
        input_lst = ["data/train_1.json",  "data/train_3.json"]
    else:
        input_lst = ["data/train_2.json", "data/train_4.json"]
    
    
    for i in input_lst:
        if not os.path.exists(i):
            raise Exception(f"Need to have file {i}, you can acquire it by running prepare.sh")
    
    raw_data = []
    for file in input_lst:
        with open(file, "r") as f:
            ram = json.loads(f.readline())
            for item in ram:
                item['image'] = f"images/inst_image/{item['image_id']}.jpg"
                assert os.path.exists("data/"+item['image']), f"File {item['image']} does not exist"
            raw_data += ram

    if with_chart_data:
        with open("data/train_5.json", "r") as f:
            ram = json.loads(f.readline())
            for item in ram:
                item['image'] = f"images/chart_image/{item['image_id']}" # image_id is the same as the file name with png extension 
                assert os.path.exists("data/"+item['image']), f"File {item['image']} does not exist"
            raw_data += ram
    
    print("Example of raw data:\n", raw_data[0])
    return raw_data


    
def helper(input: list[tuple[str, str]], position : int) -> list[dict[str:str]]:
    '''this function turns a list of (question, answer) into the required format,
    the position variable is for use when we want to include multiple'''
    if position == 1:
        out = [{"from": "human", "value":"<image> " + input[0][0]}, {"from": "gpt", "value": input[0][1]}]
    else:
        out = [{"from": "human", "value":"<image> we not look at another image. " + input[0][0]}, {"from": "gpt", "value": input[0][1]}]
    for i in range(1, len(input)):
        out += [{"from": "human", "value": input[i][0]}, {"from": "gpt", "value": input[i][1]}]
    return out

def num_tokens(input: str):
    return len(input) // 4

def get_conv_num_tokens(input: list[dict[str, str]]):
    return sum([num_tokens(i['value']) for i in input])

def agg_conv_with_same_image(raw_data: list[dict[str, str]]) -> list[dict[str, str]]:
    new_data_image_id_map = {}
    for item in raw_data:
        if item['image_id'] not in new_data_image_id_map:
            new_data_image_id_map[item['image_id']] = {
                "id": item['image_id'].replace(".png", ""),
                "images": [item['image']],
                "conversations": [],
            }
        new_data_image_id_map[item['image_id']]['conversations'].append({
            "from": "human",
            "value": item['question'],
        })
        new_data_image_id_map[item['image_id']]['conversations'].append({
            "from": "gpt",
            "value": item['answer'],
        })
    new_data = list(new_data_image_id_map.values())
    # print("Example of new data:\n", new_data[0])
    return new_data


def agg_multiple_image_conv(conv_single_image_data: list[dict[str, str]]) -> list[dict[str, str]]:
    """ agg conversations from multiple images into one conversation based on the estimated number of tokens, 
        the maximum number of tokens is 2048, and it's better to make sure there are at least 2 images in a single data
    """
    output_lst = []
    random.shuffle(conv_single_image_data)
    

    for i, item in enumerate(conv_single_image_data):
        if len(output_lst) == 0:
            conv_local_id = 1
            for j in range(0, len(item['conversations']), 2):
                item['conversations'][j]['value'] = f"Consider image {conv_local_id}. " + item['conversations'][j]['value']
            output_lst.append(item)
            continue
        if get_conv_num_tokens(output_lst[-1]['conversations']) + get_conv_num_tokens(item['conversations']) < 2048 and i < len(conv_single_image_data) - 1:
            conv_local_id += 1
            for j in range(0, len(item['conversations']), 2):
                item['conversations'][j]['value'] = f"Consider image {conv_local_id}. " + item['conversations'][j]['value']
            output_lst[-1]['conversations'] += item['conversations']
            output_lst[-1]['images'] += item['images']
        else:
            conv_pairs = [output_lst[-1]['conversations'][i:i+2] for i in range(0, len(output_lst[-1]['conversations']), 2)]
            random.shuffle(conv_pairs)
            output_lst[-1]['conversations'] = [m for conv_pair in conv_pairs for m in conv_pair]
            if random.random() < 0.5:
                output_lst[-1]['conversations'][0]['value'] = "<image> " * len(output_lst[-1]['images']) + output_lst[-1]['conversations'][0]['value']
            else:
                output_lst[-1]['conversations'][0]['value'] += " <image>" * len(output_lst[-1]['images'])
            
            conv_local_id = 1
            for j in range(0, len(item['conversations']), 2):
                item['conversations'][j]['value'] = f"Consider image {conv_local_id}. " + item['conversations'][j]['value']
            output_lst.append(item)
    print("Distribution of number of images in each data:\n", Counter([len(i['images']) for i in output_lst]))
    # print("Example of output_lst:\n", output_lst[0])
    return output_lst

def main(
    seed:int=42,
):
    random.seed(seed)
    data = read_data()

    conv_single_image_data = agg_conv_with_same_image(data)
    
    conv_agg_image_data = agg_multiple_image_conv(conv_single_image_data)
    
    with open("data/train_conv.json", "w") as f:
        json.dump(conv_agg_image_data, f, indent=4)
        print("Saved #{} data to data/train_conv.json".format(len(conv_agg_image_data))) 

if __name__ == "__main__":
    fire.Fire(main)