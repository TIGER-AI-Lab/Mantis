
import fire
import datasets
from pathlib import Path
from tqdm import tqdm

def convert_conversations(old_conv):
    new_conv = []
    for turn in old_conv:
        roles = ['user', 'assistant']
        contents = [turn['user'], turn['assistant']]
        new_conv.append({
            "role": roles[0],
            "content": contents[0]
        })
        new_conv.append({
            "role": roles[1],
            "content": contents[1]
        })
    return new_conv
            
all_subsets = ['ai2d', 'aokvqa', 'chart2text', 'chartqa', 'clevr', 'clevr_math', 'cocoqa', 'datikz', 'diagram_image_to_text', 'docvqa', 'dvqa', 'figureqa', 'finqa', 'geomverse', 'hateful_memes', 'hitab', 'iam', 'iconqa', 'infographic_vqa', 'intergps', 'localized_narratives', 'mapqa', 'mimic_cgd', 'multihiertt', 'nlvr2', 'ocrvqa', 'okvqa', 'plotqa', 'raven', 'rendered_text', 'robut_sqa', 'robut_wikisql', 'robut_wtq', 'scienceqa', 'screen2words', 'spot_the_diff', 'st_vqa', 'tabmwp', 'tallyqa', 'tat_qa', 'textcaps', 'textvqa', 'tqa', 'vistext', 'visual7w', 'visualmrc', 'vqarad', 'vqav2', 'vsr', 'websight']
def main(
    dataset_name: str="HuggingFaceM4/the_cauldron",
    sample_size=1000,
    save_dir="./data",
    num_proc=8
):
    save_dir = Path(save_dir)
    image_dir = save_dir / 'images'
    image_dir.mkdir(parents=True, exist_ok=True)
    selected_subsets = ['ai2d', 'aokvqa', 'chart2text', 'chartqa', 'clevr', 'cocoqa', 'datikz', 'diagram_image_to_text', 'docvqa', 'dvqa', 'figureqa', 'finqa', 'geomverse', 'hateful_memes', 'hitab', 'iam', 'iconqa', 'infographic_vqa', 'intergps', 'localized_narratives', 'mapqa', 'mimic_cgd', 'multihiertt', 'nlvr2', 'ocrvqa', 'plotqa', 'raven', 'rendered_text', 'robut_sqa', 'robut_wikisql', 'robut_wtq', 'scienceqa', 'screen2words', 'spot_the_diff', 'st_vqa', 'tabmwp', 'tallyqa', 'tat_qa', 'textcaps', 'textvqa', 'tqa', 'vistext', 'visual7w', 'visualmrc', 'vqarad', 'vqav2', 'vsr']
    all_datasets = datasets.Dataset.from_dict({"subset": selected_subsets})
    
    def generate_subset(item):
        subset_name = item['subset']
        dataset = datasets.load_dataset(dataset_name, subset_name, split='train', streaming=True)
        all_items = []
        for i, item in tqdm(enumerate(dataset), desc=f"Processing subset {subset_name}", total=sample_size):
            if i >= sample_size:
                break
            item['source'] = subset_name
            item['ori_images'] = item['images']
            del item['images']
            all_items.append(item)
        return {
            "subset": subset_name,
            "items": all_items
        }
    all_datasets = all_datasets.map(generate_subset, num_proc=num_proc)
    all_items = []
    for item in all_datasets:
        all_items.extend(item['items'])
    dataset = datasets.Dataset.from_list(all_items)
    dataset = dataset.cast_column('ori_images', datasets.Sequence(datasets.Image()))
    # dataset = datasets.concat_datasets([item['dataset'] for item in all_datasets])
    
    def map_save(item, index):
        if not item['ori_images']:
            images = []
        else:
            image_paths = []
            for i, image in enumerate(item['ori_images']):
                image_format = image.format if image.format else 'jpg'
                image_path = image_dir / f"{index}_{i}.{image_format}"
                if not image_path.exists():
                    image.save(image_path)
                image_paths.append(image_path)
            images = [str(image_path.relative_to(save_dir)) for image_path in image_paths]
        return {
            "id": f"{item['source']}_{index}",
            "conversation": convert_conversations(item['texts']),
            "images": images
        }
    dataset = dataset.map(map_save, with_indices=True, remove_columns=dataset.column_names, num_proc=num_proc)
    output_file = save_dir / 'train.jsonl'

    dataset.to_json(output_file, orient='records', lines=True)
    
    print(f"Saved to {output_file}")
    
    
if __name__ == '__main__':
    fire.Fire(main)