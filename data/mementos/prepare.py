import fire
import json
import pandas as pd
from pathlib import Path
from collections import defaultdict

def main(
    data_dir="./data"
):
    data_dir = Path(data_dir)
    image_dir = data_dir / "images"
    new_data_split_map = defaultdict(list)
    csv_files = ["cmc_description.csv", "dl_description.csv", "robo_description.csv"]
    csv_datas = {}
    for csv_file in csv_files:
        print(data_dir / csv_file)
        csv_datas[csv_file] = pd.read_csv(data_dir / csv_file, encoding="latin-1")
    for folder, csv_data in zip(["image_cmc", "image_dl", "image_robo"], csv_datas.values()):
        folder = image_dir / folder
        assert folder.exists(), f"{folder} does not exist"
        assert (folder / ".done").exists(), f"{folder} is not downloaded fully"
        
        for i, row in csv_data.iterrows():
            try:
                image_id = row["image_name"].split(".")[0]
            except:
                image_id = row["image"].split(".")[0]
            try:
                human_description = row["gt_description"]
            except:
                human_description = row["description"]
            img = folder / f"{image_id}.jpg"
            if not img.exists():
                img = folder / f"{image_id}.png"
            assert img.exists(), f"{img} does not exist"
        
            new_data_split_map[folder.name].append({
                "id": img.stem,
                "question_type": "description",
                "question": "Write a description for the given image sequence in a single paragraph, what is happening in this episode?",
                "images": [f"images/{folder.name}/{img.name}"],
                "options": [],
                "answer": human_description,
                "data_source": "mementos",
                "category": "image sequence description"
            })
        
            
    # for folder in ["single_image_cmc", "single_image_dl", "single_image_robo"]:
    for folder, csv_data in zip(["single_image_cmc", "single_image_dl", "single_image_robo"], csv_datas.values()):
        folder = image_dir / folder
        assert folder.exists(), f"{folder} does not exist"
        assert (folder / ".done").exists(), f"{folder} is not downloaded fully"
        
        for i, row in csv_data.iterrows():
            try:
                image_id = row["image_name"].split(".")[0]
            except:
                image_id = row["image"].split(".")[0]
            try:
                human_description = row["gt_description"]
            except:
                human_description = row["description"]
            sub_image_folder = folder / image_id
            assert sub_image_folder.exists(), f"{sub_image_folder} does not exist"
            images = []
            for img in sub_image_folder.iterdir():
                if not img.is_file():
                    continue
                assert img.suffix in [".jpg", ".png"], f"Ecountered {img} with suffix {img.suffix}"
                images.append(f"images/{folder.name}/{sub_image_folder.name}/{img.name}")
            images.sort(key=lambda x: int(x.split("/")[-1].split(".")[0]))
            
            new_data_split_map[folder.name].append({
                "id": image_id,
                "question_type": "description",
                "question": "Write a description for the given image sequence in a single paragraph, what is happening in this episode?",
                "images": images,
                "options": [],
                "answer": human_description,
                "data_source": "mementos",
                "category": "image sequence description"
            })
                
    for k, v in new_data_split_map.items():
        with open(f"data/{k}.json", "w") as f:
            json.dump(v, f, indent=4)
            print(f"Saved to data/{k}.json")
    print("Done")
    
if __name__ == '__main__':
    fire.Fire(main)