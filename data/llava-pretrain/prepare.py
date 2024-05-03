import json
from tqdm import tqdm
with open("./data/blip_laion_cc_sbu_558k.json", 'r') as f:
    data = json.load(f)
    
for item in tqdm(data):
    item['image'] = "images/" + item['image']

with open('./data/train.json', 'w') as f:
    json.dump(data, f, indent=4)