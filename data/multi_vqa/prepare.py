import fire
import json
import random
import regex as re
from datasets import load_dataset
from tqdm import tqdm
from pathlib import Path
from easy_openai import openai_completions

template = """Here are {num_of_images} captions:
{captions}
Please generate 10 independent QA pairs. Each question shall involve at least 2 images to answer. Try to cover different ability like reasoning, planning, common sense understanding, etc. Be creative with your questions and **make sure the answers require integration of information from multiple images**. Use "image i" to refer to the i-th image in your questions.

Output format:
Question: First question?
Answer: The answer to the first question.
Question: Second question?
Answer: The answer to the second question.
...
"""

def map_dataset_to_prompts(item):
    captions = item["captions"]
    num_of_images = len(captions)
    caption_str = ""
    for i, caption in enumerate(captions):
        caption_str += f"Image {i+1}: {caption}\n"
    return template.format(num_of_images=num_of_images, captions=caption_str)
    
def parse_completion(completion: str):
    qa_pairs = []
    
    # Define the pattern to match question and answer pairs
    pattern = r'Question ?(\d+)?: (.*?)(?=Answer ?(\d+)?:)|Answer ?(\d+)?: (.+?)(?=Question ?(\d+)?:|$)'
    
    # Extract question and answer pairs using regular expressions
    matches = re.findall(pattern, completion, re.DOTALL)
    # Iterate through matches and create QA pairs
    try:
        for i in range(0, len(matches), 2):
            question = matches[i][1].strip(' \n')
            answer = matches[i+1][-2].strip(' \n')
            qa_pairs.append((question, answer))
    except:
        return None
    
    if len(qa_pairs) == 0:
        return None
    return qa_pairs

def main(
    seed=42,
    image_dir="./data/sharegpt4v",
    max_size=5000,
    model_name="gpt-4"
):
    random.seed(seed)
    dataset = load_dataset("Lin-Chen/ShareGPT4V", "ShareGPT4V-PT")
    dataset = dataset['train']
    
    # shuffle the dataset
    dataset = dataset.shuffle(seed=seed)
    
    image_dir = Path(image_dir)
    output_file = Path("./data/train.json")
    
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory {image_dir} does not exist.")
    
    cur_idx = 0
    new_data = []
    while cur_idx < len(dataset):
        num_items_to_merge = random.randint(2, 6)
        
        cur_items = []
        while len(cur_items) < num_items_to_merge and cur_idx < len(dataset):
            image_path = image_dir / dataset[cur_idx]['image']
            if not image_path.exists():
                print(f"Image path {image_path} does not exist.")
            else:
                cur_items.append(dataset[cur_idx])
            cur_idx += 1
        
        if len(cur_items) <= 1:
            break
        
        items_to_merge = cur_items
        num_items_to_merge = len(items_to_merge)
        image_paths = [image_dir / item['image'] for item in items_to_merge]
        image_paths = [str(image_path.relative_to(output_file.parent)) for image_path in image_paths]
        image_captions = [item['conversations'][1]['value'] for item in items_to_merge]
        new_data.append({
            "id": f"sharegpt4v-pt_{cur_idx}",
            "images": image_paths,
            "captions": image_captions
        })
        if max_size and len(new_data) >= max_size:
            break
    
    prompts = list(map(map_dataset_to_prompts, new_data))
    results = openai_completions(prompts, model_name=model_name, max_tokens=1024, temperature=0.7, top_p=1.0)
    price_per_example = results['price_per_example']
    print(f"Total price: ${sum(price_per_example)}")
    completions = results['completions']
    
    QA_pairs = [parse_completion(completion) for completion in completions]
    
    final_data = []
    for i, item in enumerate(new_data):
        conversations = []
        if not QA_pairs[i]:
            print(f"Failed to generate QA pairs for item {item['id']}")
            continue
        for qa_pairs in QA_pairs[i]:
            conversations.append({
                "role": "human",
                "value": qa_pairs[0]
            })
            conversations.append({
                "role": "gpt",
                "value": qa_pairs[1]
            })
        if random.random() < 0.5:
            conversations[0]['value'] = "<image>"*len(item['images']) + conversations[0]['value']
        else:
            conversations[0]['value'] = conversations[0]['value'] + "<image>"*len(item['images'])
        final_data.append({
            "id": item['id'],
            "images": item['images'],
            "conversations": conversations
        })
    with open(output_file, "w") as f:
        json.dump(final_data, f, indent=4, ensure_ascii=False)
        print(f"Saved {len(final_data)} items to {output_file}")

if __name__ == "__main__":
    fire.Fire(main)
    
    