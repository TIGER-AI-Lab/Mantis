import fire
import json
import os
from tqdm import tqdm
def main(
    subset_name:str
):
    with open(f"./data/{subset_name}/{subset_name}_cap_processed.json", "r") as f:
        data = json.load(f)
    
    new_data = []
    for item in tqdm(data):
        item['conversations'][0]['value'] = item['conversations'][0]['value'].replace("<image>", "<video>")
        item['video'] = "videos/" + item['video']
        if not os.path.exists(f"./data/{subset_name}/{item['video']}"):
            continue
        new_data.append(item)
        assert os.path.exists(f"./data/{subset_name}/{item['video']}"), f"./data/{subset_name}/{item['video']}"
    data = new_data
    with open(f"./data/{subset_name}/{subset_name}_cap_processed_train.json", "w") as f:
        json.dump(data, f, indent=4)
    print(f"Processed {len(data)} items")
    print(f"Saved to ./data/{subset_name}/{subset_name}_cap_processed_train.json")
              
              
if __name__ == "__main__":
    fire.Fire(main)