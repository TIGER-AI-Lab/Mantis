import fire
import json5
import json
import random
from easy_openai import openai_completions
from string import Template
from typing import List
from pathlib import Path



PROMPT = """\
I want to generate images and the associated conversation texts to help train a multimodal language model. 

For each time, you are allowed to generate 2 to 5 image prompts, (text only), which will then be used to prompt my local stable diffusion model to generate these images by myself. So you only need to control the diversity of the images through the prompts. The prompt should be as specific as possible so that the generated image does not differ a lot from the prompt.

What's more, you should also generate conversations by asking and answering questions that can **only** be solved by combining the information of these images. 
The turns of the QA conversation can be at most 10 turns.

Since I am using these data as synthetic data to train a model, the image prompts and the conversations should be as diverse as possible. And since you also need to generate the conversations, the image prompts should somehow be related to each other so that you can generate high-quality conversations.

To make your generated more focused, please first generate a few knowledges aspects that you want to cover.  Examples include: ${demos}. You are encouraged to brainstorm more knowledge aspects other than using the demo ones. 

Then generate the image prompts and conversation.

Finally, generate one or two sentences summarizing what knowledge these image prompts and conversation have covered and why they are important to gain a better understanding of the images.

Output format:
{
    "knowledge_aspects": ["aspect1", "aspect2", ...],
    "image_prompts": ["prompt1", "prompt2", ...],
    "conversation": ["question1", "answer1", "question2", "answer2", ...],
    "summary": "summary sentence"
}

Now try to generate one example.
"""

PROMPT_BY_GPT4 = """\
Your task is to generate content that will assist in training a multimodal language model. This involves creating text prompts for image generation and accompanying conversation texts. The specifics of the task are as follows:

1. **Image Prompts Creation**: You need to create 2 to 5 detailed text prompts for image generation. These prompts should be explicit and unique to ensure minimal variance in the resultant images when generated using a stable diffusion model. Each image prompt should be at least 2 sentences describing as many details as possible. Remember, you're only crafting the prompts; the actual image generation will be done externally.

2. **Conversation Generation**: Alongside the image prompts, you need to create a dialogue consisting of questions and answers. This conversation should be designed in a way that the questions can only be answered by combining the interpreted information of at least 2 images. For example, one extreme example is to ask the common points of two images. But we encourgage more complex question related to more than 1 images, insteading of asking for 1 image in a question. The dialogue should not exceed 10 exchanges (a question and an answer count as one exchange).

3. **Diversity and Relevance**: The image prompts and the conversations should be diverse in content. However, they must be interconnected to some extent to enable the creation of coherent and high-quality dialogues.

4. **Knowledge Aspects Identification**: Before generating the prompts and conversation, identify a few key knowledge aspects you aim to cover. This can include a variety of subjects or themes. Here are some aspects and their example questions:
${demos}
But you're encouraged to brainstorm more original and complex aspects beyond the provided examples, and focus not only on image 1 and image 2, but also other images.

5. **Summary**: After creating the prompts and conversation, provide a brief summary explaining the knowledge aspects covered by the images and the dialogue. Explain why this knowledge is important for a deeper understanding of the images.

Your output should be structured as follows:
{
    "knowledge_aspects": {
        "aspect1": "example question 1",
        "aspect2": "example question 2",
        ...
    },
    "image_prompts": ["prompt1", "prompt2", ...],
    "conversation": [
        {
            "role": "human",
            "content": "...question 1..."
        },
        {
            "role": "gpt",
            "content": "...answer 1..."
        },
        {
            "role": "human",
            "content": "...question 2..."
        },
        {
            "role": "gpt",
            "content": "...answer 2..."
        },
    ],
    "summary": "summary sentence"
}
"""

PROMPT_BY_GPT4_VQA = """\
Your task is to generate content as an VQA question to evaluate a multimodal language model. This involves creating text prompts for image generation and accompanying QA pairs. The specifics of the task are as follows:

1. **Image Prompts Creation**: You need to create 2 to 5 detailed text prompts for image generation. These prompts should be explicit and unique to ensure minimal variance in the resultant images when generated using a stable diffusion model. Each image prompt should be at least 2 sentences describing as many details as possible. Remember, you're only crafting the prompts; the actual image generation will be done externally.

2. **QA pair Generation**: Alongside the image prompts, you need to create a question and answer in a way that the questions can only be answered by combining the interpreted information of at least 2 images. For example, one extreme example is to ask the common points of two images. But we encourgage more complex question related to more than 1 images, insteading of asking for 1 image in a question. Try generate hard QA pairs. And the wrong options should be well designed to be confusing.

3. **Diversity and Relevance**: The image prompts and the QA pair should be diverse in content. However, they must be interconnected to some extent to enable the creation of coherent and high-quality dialogues.

4. **Knowledge Aspects Identification**: Before generating the prompts and QA pair, here you are given a key knowledge aspect and example question that the QA pair should cover. 
${demos}
More complex questions related to this aspect are encouraged.

5. **Summary**: After creating the prompts and QA pair, provide a brief summary explaining the knowledge aspects covered by the images and QA pair. Explain why this knowledge is important for a deeper understanding of the images.

Your output should be structured as follows:
{
    "knowledge_aspect": "...",
    "image_prompts": ["prompt1", "prompt2", ...],
    "question": "...",
    "options": ["(A) ...", "(B) ...", "(C) ...", "(D) ..."],
    "answer": "A/B/C/D", 
    "summary": "summary sentence"
}
"""    


def load_json_file(file_path):
    with open(file_path, "r") as f:
        return json.load(f)
    
def main(
    seed_demo_file: str = "./seed_demos.v2.json",
    output_file: str = "./data/generated_examples.json",
    num_examples: int = 15000,
    seed: int = 31,
    mode="vqa",
    model_name="ChatGPT"
):
    print(seed_demo_file, output_file, num_examples, seed, mode, model_name)
    if not isinstance(seed_demo_file, Path):
        seed_demo_file = Path(seed_demo_file)
    if not isinstance(output_file, Path):
        output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    random.seed(seed)
    # load seed demo file
    seed_demos = load_json_file(seed_demo_file)
    
    
    # generate examples
    prompts = []
    for i in range(num_examples):
        if mode == "vqa":
            demos_to_use = random.sample(seed_demos, k=1)
        else:
            demos_to_use = random.sample(seed_demos, k=4)
        # demos_to_use = [{"aspect": ..., "question": ...}, ...]
        demos_str = "\n".join([f"- {demo['aspect']}: {demo['question']}" for demo in demos_to_use])
        
        if mode == "vqa":
            prompt = Template(PROMPT_BY_GPT4_VQA).substitute(demos=demos_str)
        else:
            prompt = Template(PROMPT_BY_GPT4).substitute(demos=demos_str)
        prompts.append([{
            "role": "user",
            "content": prompt,
        }])

    # generate completions
    results = openai_completions(
        prompts,
        model_name=model_name,
        max_tokens=1536,
        temperature=0.7,
        top_p=1,
        use_cache=True,
    )
    completions = results["completions"]
    total_price = sum(results['price_per_example'])
    print(f"Total price: {total_price} USD")
    
    
    
    final_data = []
    for i in range(num_examples):
        try:
            completions[i] = completions[i].strip(' \n')
            if completions[i].startswith("```json") and completions[i].endswith("```"):
                completions[i] = completions[i][7:-3].strip(' \n')
            parsed_data = json5.loads(completions[i])
        
            # conversation post processing
            # conversations = []
            # for j in range(0, len(parsed_data["conversation"]), 2):
            #     conversations.append({
            #         "role": "human",
            #         "content": parsed_data["conversation"][j],
            #     })
            #     conversations.append({
            #         "role": "gpt",
            #         "content": parsed_data["conversation"][j+1],
            #     })
            # conversations = parsed_data["conversation"]
        except Exception as e:
            print(e)
            print(completions[i])
            print(f"Failed to parse completion {i}")
            continue
        # final_data.append({
        #     "knowledge_aspects": parsed_data["knowledge_aspects"],
        #     "image_prompts": parsed_data["image_prompts"],
        #     "conversation": parsed_data["conversation"],
        #     "summary": parsed_data["summary"],
        # })
        final_data.append(parsed_data)
    
    if mode != 'vqa':
        for item in final_data:
            for aspect, question in item["knowledge_aspects"].items():
                seed_demos.append({
                    "aspect": aspect,
                    "question": question,
                })
        # save seed demos
        with open("./data/seed_demos.new.json", "w") as f:
            json.dump(seed_demos, f, indent=4)
            print(f"Saved new seed demos to ./seed_demos.new.json")
    
    # save to file
    with open(output_file, "w") as f:
        json.dump(final_data, f, indent=4)
        print(f"Saved generated examples to {output_file}")
if __name__ == "__main__":
    fire.Fire(main)