import json
with open("./data/train.jsonl", 'r') as f:
    data = [json.loads(line) for line in f.readlines()]
final_data = [item for item in data if item['id'].startswith('datikz') and item['conversation'][0]['content'].startswith("Synthesize TikZ code for this figure.")]
with open("./data/datikz.jsonl", 'w') as f:
    for item in final_data:
        f.write(json.dumps(item) + '\n')
