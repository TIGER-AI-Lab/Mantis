import json
import fire
import os
import regex as re
import numpy as np
from string import Template
from pathlib import Path
from easy_openai import openai_completions, _chatml_to_prompt


PROMPT_TEMPLATE = """\
I am transforming the task type of a dataset. 
The original dataset askes models to take input 2 images and asks it to generate a text describing the difference between them. 
The targeted transformed dataset be a VQA task that takes 2 images and a question about the differnce in, and the answer is either short answer or multi-choice. 

You are provided the groudth truth differences description:
${difference}

Now generate a single question for in both multi-choice and short-answer format, you may consider the editing requests as a whole.

Overall output format:
Multi-choice:
{
"question": "...",
"options": [ "A: {option A}", ...],
"answer": "{A or B or ...}" 
}
Short-answer:
{
"question": "...",
"options": [],
"answer": "{short answer, less than 3 words}"
}
"""
def main(
    input_file: str="test.json",
    output_file: str="test_vqa.json",
    shuffle: bool=False,
    seed: int=42,
    sample_size: int=500,
):
    with open(input_file) as f:
        data = json.load(f)[:5]
    template = Template(PROMPT_TEMPLATE)
    prompts = [template.substitute(
        difference="\n".join([
            f"{i+1}. " + sent for i, sent in enumerate(x['sentences'])
        ])) for x in data]
    messages_list = [
        [{
            "content": prompt,
            "role": "user"
        }]
        for prompt in prompts
    ]
    prompts = [_chatml_to_prompt(messages) for messages in messages_list]
    results = openai_completions(prompts, 'gpt-3.5-turbo-1106')
    completions = results['completions']
    questions = []
    cur_dir_path = os.path.dirname(os.path.realpath(__file__))
    cur_dir_path = Path(cur_dir_path)
    work_dir = cur_dir_path.parent.parent # we set the url of images to be relative to the root working directory
    for i, completion in enumerate(completions):
        data[i]['transformed'] = {}
        multi_choice = re.search(r'(?<=Multi-choice:)((.|\n)*)(?=Short-answer:)', completion, re.DOTALL)
        if multi_choice:
            try:
                question = json.loads(multi_choice.group(1))
                image1_path = "./resized_images/{image_id}.jpg".format(image_id=data[i]['img_id'])
                image2_path = "./resized_images/{image_id}_2.jpg".format(image_id=data[i]['img_id'])
                image1_path = Path(image1_path).absolute()
                image2_path = Path(image2_path).absolute()
                if not image1_path.exists() or not image2_path.exists():
                    image1_path = "./resized_images/{image_id}.png".format(image_id=data[i]['img_id'])
                    image2_path = "./resized_images/{image_id}_2.png".format(image_id=data[i]['img_id'])
                    image1_path = Path(image1_path).absolute()
                    image2_path = Path(image2_path).absolute()
                assert image1_path.exists(), "{image1_path} does not exist".format(image1_path=image1_path)
                assert image2_path.exists(), "{image2_path} does not exist".format(image2_path=image2_path)
                image_1 = image1_path.relative_to(work_dir).as_posix()
                image_2 = image2_path.relative_to(work_dir).as_posix()
                assert question['answer'] in ['A', 'B', 'C', 'D'], "answer should be one of A, B, C, D, but got {answer}".format(answer=question['answer'])
                answer_idx = ord(question['answer']) - ord('A')
                for i in range(len(question['options'])):
                    if re.match(r'^[A-D]:', question['options'][i]):
                        question['options'][i] = question['options'][i][2:].strip()
                answer = question['options'][answer_idx]
                question['options'] = np.random.permutation(question['options']).tolist()
                answer_idx = question['options'].index(answer)
                question['answer'] = chr(ord('A') + answer_idx)
                questions.append({
                    "question_type": "multi-choice",
                    "question": question['question'],
                    "images": [image_1, image_2],
                    "options": question['options'],
                    "answer": question['answer'],
                    "data_source": "spot-the-diff",
                    "categoty": "difference description"
                })
            except Exception as e:
                print(multi_choice.group(1))
                raise e
            
        short_answer = re.search(r'(?<=Short-answer:)((.|\n)*)', completion, re.DOTALL)
        if short_answer:
            try:
                question = json.loads(short_answer.group(1))
                image1_path = "./resized_images/{image_id}.jpg".format(image_id=data[i]['img_id'])
                image2_path = "./resized_images/{image_id}_2.jpg".format(image_id=data[i]['img_id'])
                image1_path = Path(image1_path).absolute()
                image2_path = Path(image2_path).absolute()
                if not image1_path.exists() or not image2_path.exists():
                    image1_path = "./resized_images/{image_id}.png".format(image_id=data[i]['img_id'])
                    image2_path = "./resized_images/{image_id}_2.png".format(image_id=data[i]['img_id'])
                    image1_path = Path(image1_path).absolute()
                    image2_path = Path(image2_path).absolute()
                assert image1_path.exists(), "{image1_path} does not exist".format(image1_path=image1_path)
                assert image2_path.exists(), "{image2_path} does not exist".format(image2_path=image2_path)
                image_1 = image1_path.relative_to(work_dir).as_posix()
                image_2 = image2_path.relative_to(work_dir).as_posix()
                questions.append({
                    "question_type": "short-answer",
                    "question": question['question'],
                    "images": [image_1, image_2],
                    "options": [],
                    "answer": question['answer'],
                    "data_source": "spot-the-diff",
                    "categoty": "difference description"
                })
            except Exception as e:
                print(short_answer.group(1))
                raise e
            
        
    with open(output_file, 'w') as f:
        json.dump(questions, f, indent=4)
if __name__ == '__main__':
    fire.Fire(main)
    