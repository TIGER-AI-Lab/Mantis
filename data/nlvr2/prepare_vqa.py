import json
from easy_openai import openai_completions, _chatml_to_prompt
from string import Template
import fire
from pathlib import Path

PROMPT_TEMPLATE = """\
Give a statement about 2 images along with a label saying whether it's a correct statement (true) or a wrong statement (false ), you need to come up with a proper multiple choice question with different choices and only 1 correct answer. Please create reasonable questions based on the limited information provided, no additional hallucination should be added.

Some rules to notice:
1. If the statement does not mention which image it is referring to, the question might be about both images. For example, "There are an empty glass" means that "there is at least one empty glass in the two images". "There are six bottles" means that "there are totally six bottles in the two images".
2. If the statement mentions "An image" or "One image" but does not specify whether it is the left or right image, it means "there is at least one" in the two images. For example, "An image shows a curving walkway of dark glass circles embedded in dirt and flanked by foliage." means that "there is at least one image shows a curving walkway of dark glass circles embedded in dirt and flanked by foliage."
3. If the statement is labeled as "True", then it's good for you to take the statement as a fact and create a question based on it. 
4. If the statement is labeled as "False", then the only thing you can be sure of is that the statement is wrong, and you should create a question based on the opposite of the statement. For example, if the statement is "There are six bottles", then the question should be "How many bottles are there?" and the answer should be "Not six" or other numbers. 
5. Note that the generated options should only have one correct answer. For example, if the statement is "None of the gloves pictured are black", the bad question and options would be "Which of the following colors are not present in the gloves pictured?" with options "a) black b) white c) red d) blue" because you don't know whether other colors like white, red, or blue are present in the gloves. The good question and options would be "Which image shows a black glove?" with options "a) left image b) right image c) both images d) neither image" because you know that there is at least one black glove in the two images. In short, the options should not hallucinate any information that is not in the statement. You can ask about which image shows a certain thing, in cases like that.

Here are a few examples:
Statement: "The right image shows a curving walkway of dark glass circles embedded in dirt and flanked by foliage."
Label: "True"
Transformed Question: 
{"question": "Select the answer that best describes the two images", "options": [ "a) The right image shows a curving walkway of dark glass circles embedded in dirt and flanked by foliage.", "b) The left image shows a curving walkway of dark glass circles embedded in dirt and flanked by foliage.", "c) The right image shows a straight walkway of dark glass circles embedded in dirt and flanked by foliage.", "d) The left image shows a straight walkway of dark glass circles embedded in dirt and flanked by foliage."], "answer": "A"}

Statement: "There are exactly six bottles in the right image."
Label: "True"
Transformed Question: 
{"question": "How many bottles are in the right image?", "options": [ "a) four", "b) five", "c) six", "d) seven"], "answer": "C"}

Statement: "In at least one image there are five bottles of beer."
Label: "False"
Transformed Question: 
{"question": "Select the answer that best describes the two images", "options": [ "a) Both images have five or more bottles of beer", "b) Only the left image has five or more bottles of beer", "c)  Only the right image has five or more bottles of beer", "d)  None of the images have five or more bottles of beer"], "answer": "D"}

Now it's your turn:
Statement: "${sentence}"
Label: "${answer}"
Transformed Question: 

please give me a multiple choice question with 4 choices. The output should be in the following json format, do not include extra spacing or new line character:
{"question": "...", "options": [ "...", ...], "answer": "{A or B or ...}"}
"""

def load_data(path:str):
    if path.endswith(".json"):
        return [json.loads(open(path).read())]
    elif path.endswith(".jsonl"):
        return [json.loads(i) for i in open(path).readlines()]
    else:
        raise NotImplementedError
    
def map_item_to_prompt(input: dict):
    # randomly select 10 demos
    template = Template(PROMPT_TEMPLATE)
    
    if input['label'].lower() == "true":
        answer = "True"
    else:
        answer = "False"
        
    prompt = template.substitute(
        sentence=input['sentence'],
        answer=answer,
    )
    
    messages = [
        {
            "role": "user",
            "content": prompt,
        },
    ]
    
    return _chatml_to_prompt(messages)

def main(
    input_file: str,
    output_file: str,
    image_dir: str,
    process_limit: int=-1,
    model_name:str="chatGPT"
):
    input_data = load_data(input_file)
    input_data = [i for i in input_data if i['label'] == "True"] # only process true samples
    if process_limit <= 0:
        process_limit = len(input_data)
        
    prompts = [map_item_to_prompt(input_data[i]) for i in range(process_limit)]
    results = openai_completions(
        prompts=prompts,
        model_name=model_name,
        max_tokens=256,
        temperature=0.0,
        top_p=1.0,
        )['completions']
    
    output_data = []
    if not isinstance(output_file, Path):
        output_file = Path(output_file)
    for i in range(process_limit):
        dicc = json.loads(results[i])
        dicc["id"] = f"nlvr2_{i}"
        dicc["question_type"] = "multi-choice"
        dicc["question"] = "<image><image>" + dicc["question"]
        image_identifier = input_data[i]['identifier'][:input_data[i]['identifier'].rfind("-")]
        dicc["images"] = [Path(image_dir) / f"{image_identifier}-img0.png", Path(image_dir) / f"{image_identifier}-img1.png"]
        dicc["images"] = [str(i.relative_to(output_file.parent)) for i in dicc["images"]]
        dicc["data_source"] = "original dataset"
        dicc["category"] = "difference description"
        dicc["sentence"] = input_data[i]['sentence']
        dicc["label"] = input_data[i]['label']
        output_data.append(dicc)
    json.dump(output_data, open(output_file, "w"), indent=4)
    
    print("Finished!")
            
if __name__ == "__main__":
    fire.Fire(main)
