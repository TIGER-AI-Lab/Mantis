mkdir -p data
python get_seed_examples.py


# ChatGPT new data items: 2 second per item
# Image Synthesizing: 
# 10 second per image, 40 second per item, 1 hour per 100 items
# SD turbo will be way more faster, 2 minutes per 100 items

# 34 USD for 15000 examples, about 6 hours

CUDA_VISIBLE_DEVICES=4 python prepare.py --input_file ./data/generated_examples.json --output_file ./data/train.json --image_dir ./data/images --seed 31 --diffuser stabilityai/sdxl-turbo --start_idx 0 --end_idx 3750 &
CUDA_VISIBLE_DEVICES=5 python prepare.py --input_file ./data/generated_examples.json --output_file ./data/train.json --image_dir ./data/images --seed 31 --diffuser stabilityai/sdxl-turbo --start_idx 3750 --end_idx 7500 &
CUDA_VISIBLE_DEVICES=6 python prepare.py --input_file ./data/generated_examples.json --output_file ./data/train.json --image_dir ./data/images --seed 31 --diffuser stabilityai/sdxl-turbo --start_idx 7500 --end_idx 11250 &
CUDA_VISIBLE_DEVICES=7 python prepare.py --input_file ./data/generated_examples.json --output_file ./data/train.json --image_dir ./data/images --seed 31 --diffuser stabilityai/sdxl-turbo --start_idx 11250 --end_idx 15000 &


CUDA_VISIBLE_DEVICES=3 python prepare.py --input_file ./data/generated_examples.json --output_file ./data/train.json --image_dir ./data/images_sdturbo-1 --seed 31 --diffuser stabilityai/sdxl-turbo 


python get_seed_examples.py --seed_demo_file "./seed_demos.v2.clean.json" --output_file ./data/generated_eval_examples.json --num_examples 500 --seed 31 --model_name ChatGPT --mode "vqa"