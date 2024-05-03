import fire
import json
import fire
import torch
import random
from PIL import Image
from diffusers import AutoPipelineForText2Image
from functools import partial
from tqdm import tqdm
from pathlib import Path

def synthesize_image(text, pipeline, generator,  **kwargs):
    image = pipeline(text, generator=generator, **kwargs).images[0]
    return image


def main(
    input_file:str ="./generated_examples.json",
    output_file:str ="./data/train.json",
    image_dir:str ="./data/images",
    seed: int = 31,
    diffuser="stabilityai/sdxl-turbo",
    start_idx: int = None,
    end_idx: int = None,
    mode="conv",
):
    """
    stabilityai/sdxl-turbo
    stabilityai/stable-diffusion-xl-base-1.0
    """
    random.seed(seed)
    # how to use this 
    if not isinstance(input_file, Path):
        input_file = Path(input_file)
    if not isinstance(output_file, Path):
        output_file = Path(output_file)
    if not isinstance(image_dir, Path):
        image_dir = Path(image_dir)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    with open(input_file, "r") as f:
        examples = json.load(f)
        
    output_file_postfix = ""
    if end_idx is not None:
        examples = examples[:end_idx]
        print("Truncated to first [:{}] examples".format(end_idx))
        output_file_postfix += "{}".format(end_idx)
    if start_idx is not None:
        examples = examples[start_idx:]
        print("Truncated to last [{}:] examples".format(start_idx))
        output_file_postfix = ".{}".format(start_idx) + "-" + output_file_postfix
    if output_file_postfix:
        output_file = output_file.parent / (output_file.stem + output_file_postfix + output_file.suffix)
        
    diffuser_pipeline = AutoPipelineForText2Image.from_pretrained(
        diffuser, torch_dtype=torch.float16, variant="fp16"
    ).to("cuda")
    generator = torch.Generator("cuda").manual_seed(seed)
    partial_kwargs = {
        "pipeline": diffuser_pipeline,
        "generator": generator,
    }
    if diffuser == "stabilityai/sdxl-turbo":
        partial_kwargs["num_inference_steps"] = 1
        partial_kwargs["guidance_scale"] = 0.0
    synthesize_image_from_text = partial(synthesize_image, **partial_kwargs) # synthesize_image_from_text(text)
    
    
    # resolutions_pool = [(256, 256), (320, 240), (480, 320), (640,480), (512, 512), (1024, 1024), (1280, 720), (1920, 1080), (800, 600)]
    resolutions_pool = [(512, 512)]
    new_data = []
    for i, example in tqdm(enumerate(examples), total=len(examples), desc="Synthesizing images"):
        item_id = "synthetic_{}".format(i)
        image_paths = []
        for j, image_prompt in enumerate(example["image_prompts"]):
            image_path = image_dir / "{}_img-{}.png".format(item_id, j)
            image_paths.append(image_path)
            if image_path.exists():
                flag = True
                try:
                    existing_image = Image.open(image_path)
                except Exception as e:
                    print("Failed to open image {}".format(image_path))
                    flag = False
                if flag and existing_image.size in resolutions_pool:
                    continue
            width, height = random.choice(resolutions_pool)
            image = synthesize_image_from_text(image_prompt, width=width, height=height)
            image.save(image_path)
        
        image_paths = [str(image_path.relative_to(output_file.parent)) for image_path in image_paths]    
        if mode == "conv":
            new_data.append({
                "id": item_id,
                "images": image_paths,
                "conversations": example["conversation"],
            })
        else:
            new_data.append({
                "id": item_id,
                "question_type": "multi-choice",
                "images": image_paths,
                "question": example["question"],
                "options": example["options"],
                "answer": example["answer"],
                "data_source": "synthetic",
                "category": example['knowledge_aspect']
            })
        
    with open(output_file, "w") as f:
        json.dump(new_data, f, indent=4)
        print("Saved to {}".format(output_file))
    
    
    
if __name__ == "__main__":
    fire.Fire(main)