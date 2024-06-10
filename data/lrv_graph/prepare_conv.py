import json
import os
from mantis.easy_openai import openai_completions, _chatml_to_prompt
from string import Template
import fire

PROMPT_TEMPLATE = """
Here are some questions and answers related to the first image:

${image1}

Here are some questions and answers related to the second image:

${image2}

Now, please give me ${question_ct} question and its answer which compares the two images. Each question should ask about the details in both image. The output should be in the following json format, do not include extra spacing or new line character:
[{"question": "...", "answer": "..."}]
"""

def read_data() -> dict[str: list[tuple[str, str]]]:
    '''
    read in the data and return it in a dictionary where:
    
    key: the image id, ex: "12345", this corresponds to "12345.jpg" in the data/image forlder
    value: a list of tuple in the form of (question, answer)
    '''
    input_file = "data/train.json"
    if not os.path.exists(input_file):
        raise Exception(f"Need to have file {input_file}, you can acquire it by running prepare.sh")
    output = dict()
    with open(input_file, "r") as f:
        ram = json.loads(f.readline())
        for i in ram:
            k = i['image_id']
            if k in output:
                output[k].append((i['question'], i['answer']))
            else:
                output[k] = [(i['question'], i['answer'])]
    return output

    
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


def map_item_to_prompt(input1 : list[tuple[str, str]], input2 : list[tuple[str, str]], question_ct:int = 3) -> list[dict[str:str]]:
    '''
    receive the questions and answers of 2 images, then produce an additional question that 
    '''
    template = Template(PROMPT_TEMPLATE)
        
    prompt = template.substitute(
        image1="\n".join([f"Question: {i[0]} Answer: {i[1]}" for i in input1]),
        image2="\n".join([f"Question: {i[0]} Answer: {i[1]}" for i in input2]),
        question_ct=str(question_ct)
    )
    
    messages = [
        {
            "role": "user",
            "content": prompt,
        },
    ]
    
    return _chatml_to_prompt(messages)


def main(process_limit: int=9999999, model_name:str="chatGPT") -> None:
    process_limit = int(process_limit)
    output_lst = []
    data = read_data()
    ids = list(data.keys())
    input_for_gpt = [map_item_to_prompt(data[ids[i]],data[ids[i+1]]) for i in range(0, min(len(ids), process_limit) - 1, 2)]
    gpt_answer = openai_completions(
        prompts=input_for_gpt,
        model_name=model_name,
        max_tokens=256,
        temperature=0.0,
        top_p=1.0,
    )['completions']
    for i in range(0, min(len(ids), process_limit) - 1, 2):
        try:
            ram2 = []
            for ram in json.loads(gpt_answer[int(i / 2)]):
                ram2 += [{"from": "human", "value": ram['question']}, {"from": "gpt", "value": ram['answer']}]
            output_lst.append({
                "id": str(int(i / 2)), 
                "images": [f"data/image/{ids[i]}.jpg", f"data/image/{ids[i+1]}.jpg"], 
                "conversations": helper(data[ids[i]], 1) + helper(data[ids[i+1]], 2) + ram2
                })
        except:
            print(f"json parse error {gpt_answer[int(i / 2)]}")
    with open("data/train_conv.json", "w") as f:
        f.write(json.dumps(output_lst))

if __name__ == "__main__":
    fire.Fire(main)