from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm
import json
import fire


def save_subset_images(subset, image_save_dir):
    data = load_dataset("pufanyi/MIMICIT", f"{subset}_Images")
    image_save_dir = Path(image_save_dir)
    image_save_dir.mkdir(exist_ok=True)
    ID_to_image_file = {}
    for item in tqdm(data['train'], desc="Saving {} images".format(subset)):
        image_file = image_save_dir / f"{item['id']}.{item['image'].format.lower()}"
        if not image_file.exists():
            item['image'].save(image_file)
        ID_to_image_file[item['id']] = str(image_file)
    return ID_to_image_file


def get_subset_convs(subset, ID_to_image_file):
    data = load_dataset("pufanyi/MIMICIT", f"{subset}_Instructions")
    convs = {}
    missing_count = 0
    for item in tqdm(data['train'], desc="Processing {} instructions".format(subset)):
        related_instruction_ids = item['related instructions']
        try:
            images = [ID_to_image_file[image_id] for image_id in item['images']]
        except KeyError:
            missing_count += 1
            continue
        if not any([id in convs for id in related_instruction_ids]):
            convs[item['id']] = {
                "id": "MIMICIT-" + item['id'],
                "images": [ID_to_image_file[image_id] for image_id in item['images']],
                "conversations": [
                    {
                        "role": "human",
                        "value": item['instruction']
                    },
                    {
                        "role": "gpt",
                        "value": item['answer']
                    }
                ]
            }
        else:
            processed_related_instruction_id = [_id for _id in related_instruction_ids if _id in convs]
            assert len(processed_related_instruction_id) == 1, f"{processed_related_instruction_id}"
            processed_related_instruction_id = processed_related_instruction_id[0]
            convs[processed_related_instruction_id]['conversations'].append({
                "role": "human",
                "value": item['instruction']
            })
            convs[processed_related_instruction_id]['conversations'].append({
                "role": "gpt",
                "value": item['answer']
            })
    print(f"Missing {missing_count} instructions")
    return list(convs.values())

def main(
    mimicit_subsets=["SD", "VST"]
):

    all_subset_data = []
    for subset in mimicit_subsets:
        image_save_dir = Path(f"./{subset}_images")
        ID_to_image_file = save_subset_images(subset, image_save_dir)
        convs = get_subset_convs(subset, ID_to_image_file)
        with open(f"./{subset}.json", "w") as f:
            json.dump(convs, f, indent=4)
        all_subset_data.extend(convs)
    with open("./train.json", "w") as f:
        json.dump(all_subset_data, f, indent=4)
    
if __name__ == "__main__":
    fire.Fire(main)