import json
import random
import fire
import regex as re
from mantis.easy_openai import openai_completions, _chatml_to_prompt
from string import Template
from pathlib import Path
from typing import List


PROMPT_TEMPLATE = """\
You are given a description of two images (left and right). You need to transform it to a question and response format used to train a model to answer questions about images. 
${label}

${demos}
"""

PROMPT_TEMPLATE_FOR_FALSE = """\
You are given an incorrect description of two images (left and right). You need to transform it to a question and response in a natural chat style used to train a model to answer questions about images. 
Since the description is incorrect, the question should be asking the content of the description, and the response should be opposite to the description.
Note that the question and response cannot see the description, please rephrase the description in your own words if you mention the description in the question or response.
If the description is not specifying which image, then you should assume the description is about the both images as a whole.

${demos}
"""

def load_data(path:str):
    if path.endswith(".json"):
        return json.loads(open(path).read())
    elif path.endswith(".jsonl"):
        return [json.loads(i) for i in open(path).readlines()]
    else:
        raise NotImplementedError

def map_item_to_prompt(item:dict, demos:List[dict]):
    # randomly select 10 demos
    demos = random.sample(demos, 10) if len(demos) > 10 else demos
    demos = [x for x in demos if x['label'] == item['label']][:2] # 2 shot
    template = Template(PROMPT_TEMPLATE)
    demo_sentences = [i['sentence'] for i in demos]
    demo_labels = [item['label']] * len(demo_sentences)
    demo_questions = [i['question'] for i in demos]
    demo_responses = [i['response'] for i in demos]
    
    demo_strs = []
    # label = "a correct" if item['label'].lower() == "true" else "an incorrect"
    if item['label'].lower() == "true":
        for i in range(len(demo_sentences)):
            demo_strs.append(f"Description: {demo_sentences[i]}\nQuestion: {demo_questions[i]}\nResponse: {demo_responses[i]}")
        demo_strs.append(f"Description: {item['sentence']}")
        demo_strs = "\n\n".join(demo_strs)
        label = "The following given descriptions are all totally correct. Therefore, Your generated questions and responses should express same the meaning with the given descriptions."
    else:
        for i in range(len(demo_sentences)):
            demo_strs.append(f"Incorrect Description: {demo_sentences[i]}\n\nQuestion: {demo_questions[i]}\nResponse: {demo_responses[i]}")
        demo_strs.append(f"Incorrect Description: {item['sentence']}")
        demo_strs = "\n\n".join(demo_strs)
        # label = "The following given descriptions all contain incorrect information. Therefore, Your generated question and response should express the opposite meaning with the given descriptions."
        template = Template(PROMPT_TEMPLATE_FOR_FALSE)
        label = None
    prompt = template.substitute(
        demos=demo_strs,
        label=label,
    )
    messages = [
        {
            "role": "user",
            "content": prompt,
        },
    ]
    return _chatml_to_prompt(messages)

def parse_completion_to_conv(completion:str):
    completion = completion.strip()
    question = re.search(r"Question: (.*)", completion)
    response = re.search(r"Response: (.*)", completion)
    if not question or not response:
        return None
    question = question.group(1).strip()
    response = response.group(1).strip()
    conv = [
        {
            "role": "human",
            "value": question,
        },
        {
            "role": "gpt",
            "value": response,
        }
    ]
    return conv
    
def main(
    input_file: str,
    output_file: str,
    image_dir: str,
    demo_file: str,
    demo_update_freq: int=1000,
    model_name:str="ChatGPT",
    seed:int=42,
):
    random.seed(seed)
    input_data = load_data(input_file)[:50]
    demos = load_data(demo_file)
    new_data = []
    for i in range(0, len(input_data), demo_update_freq):
        batch_data = input_data[i:i+demo_update_freq]
        batch_data = [x for x in batch_data if x['label'] == "False"]
        propmts = [map_item_to_prompt(i, demos) for i in batch_data]
        results = openai_completions(
            prompts=propmts,
            model_name=model_name,
            max_tokens=256,
            temperature=0.0,
            top_p=1.0,
        )
        completions = results['completions']
        total_price = sum(results['price_per_example'])
        print(f"Total price: {total_price} for {len(completions)} examples")
        image_dir = Path(image_dir)
        input_file_dir = Path(input_file).parent
        assert len(batch_data) == len(completions), f"Batch size mismatch {len(batch_data)} != {len(completions)}"
        for i, item, completion in zip(range(len(batch_data)), batch_data, completions):
            pair_id = item['identifier'][:item['identifier'].rfind("-")]
            images = [image_dir / f"{pair_id}-img0.png", image_dir / f"{pair_id}-img1.png"]
            images = [i.relative_to(input_file_dir) for i in images]
            if not all([(input_file_dir / i).exists() for i in images]):
                print(f"Missing images for {item['identifier']}")
                continue
            images = [str(i) for i in images]
            conversation = parse_completion_to_conv(completion)
            if not conversation:
                print(f"Failed to parse completion for {item['identifier']}")
                continue
            new_data.append({
                "id": item['identifier'],
                "images": images,
                "conversations": conversation,
            })
        # randomly sample 5% of the results to the demos
        # new_demos_idxs = random.sample(range(len(new_data)), int(len(propmts) * 0.05))
        new_demos_idxs = random.sample(range(len(new_data)), int(len(new_data) * 0.05))
        new_demos = [input_data[i] for i in new_demos_idxs]
        for i in range(len(new_demos)):
            new_demos[i]['question'] = new_data[new_demos_idxs[i]]['conversations'][0]['value']
            new_demos[i]['response'] = new_data[new_demos_idxs[i]]['conversations'][1]['value']
        print(f"Updated size of demonstrations pool from {len(demos)} to {len(demos) + len(new_demos)}")
        # demos.extend(new_demos)
            
    with open(output_file, "w") as f:
        # save to json
        json.dump(new_data, f, indent=4)
        print(f"Saved to {output_file}")
    to_save_demo_file = demo_file.replace(".json", f"_updated_{model_name}.json")
    with open(to_save_demo_file, "w") as f:
        # save to json
        json.dump(demos, f, indent=4)
        print(f"Saved to {to_save_demo_file}")
    
if __name__ == "__main__":
    fire.Fire(main)
