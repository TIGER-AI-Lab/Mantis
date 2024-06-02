import json
from pathlib import Path
from tqdm import tqdm
# file_list = ["data_insf.json", "data_lab.json", "data_real.json", "data_static.json", "data_worsen_gen.json"]
file_list = list([str(x) for x in Path("./").glob("data*.json")])
print(file_list)
all_data = []
for file_name in file_list:
    with open(file_name, "r") as f:
        data = json.load(f)
        for item in tqdm(data, desc=file_name):
            # video_id = item['images'][0].split("_")[0]
            # item['images'] = ["images/" + video_id + "/" + f"{video_id}_{i+1:02d}.jpg" for i, image in enumerate(item['images'])]
            item['images'] = ["images/" + item['images'][0].split("_")[0] + "/" + image for i, image in enumerate(item['images'])]
            assert all([Path(image).exists() for image in item['images']]), item['images']
            labels = [x for x in item['conversations'][1]['value'].split("\n") if x]
            labels = {label.split(":")[0].strip(' \n'): float(label.split(":")[1]) for label in labels}
            all_data.append({
                "id": item['id'],
                "images": item['images'],
                "prompt": item['conversations'][0]['value'],
                "labels": labels
            })
with open("train.json", "w") as f:
    json.dump(all_data, f, indent=4)