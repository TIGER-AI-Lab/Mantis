mkdir -p logs
# python prepare.py --dataset_path "laion/gpt4v-dataset"

num_shards=8
shared_size=1000
for i in $(seq 0 $((num_shards-1))); do
    start_idx=$((i*shared_size))
    end_idx=$(((i+1)*shared_size))
    echo "start_idx: $start_idx, end_idx: $end_idx"
    python prepare.py --dataset_path "laion/gpt4v-dataset" --image_save_dir "data/LaionGPT4V_images" --output_file "data/LaionGPT4V_train.json" --start_idx $start_idx --end_idx $end_idx > logs/prepare_gpt4v_v2_${start_idx}_${end_idx}.log 2>&1 &
done


num_shards=8
shared_size=5000
for i in $(seq 0 $((num_shards-1))); do
    start_idx=$((i*shared_size))
    end_idx=$(((i+1)*shared_size))
    echo "start_idx: $start_idx, end_idx: $end_idx"
    python prepare.py --dataset_path "laion/220k-GPT4Vision-captions-from-LIVIS" --image_save_dir "data/LaionGPT4V-LIVIS_images" --output_file "data/LaionGPT4V-LIVIS_train.json" --start_idx $start_idx --end_idx $end_idx > logs/prepare_220k_${start_idx}_${end_idx}.log 2>&1 &
done

# # sharegptv sam
# gdown https://drive.google.com/uc?id=16_5S-l4X01Kv7AUPFzfJ2LZjpw1-lLVF -O data/sharegpt4v/sam_images_share-sft.zip
# mkdir -p data/sam
# mkdir -p data/sam/images
# mv data/sam_images_share-sft.zip data/sam
# unzip data/sam/sam_images_share-sft.zip -d data/sam/images

# gdown https://drive.google.com/uc?id=1FxB2Nw-vWUcTUSI_dBpPIykb-uGYoEqV -O data/sharegpt4v/wikiart.zip
# mkdir -p data/sharegpt4v/wikiart
# mv data/sharegpt4v/wikiart.zip data/sharegpt4v/wikiart
# unzip data/sharegpt4v/wikiart/wikiart.zip -d data/sharegpt4v/wikiart/images
# mv data/sharegpt4v/wikiart/images/data/wikiart/images/* data/sharegpt4v/wikiart/images
# rm -rf data/sharegpt4v/wikiart/images/data

# gdown https://drive.google.com/uc?id=1JpJkN7ZMA50xAhMx9O-rVb5yLhfGm3_o -O data/sharegpt4v/web-landmakr.zip
# mkdir -p data/sharegpt4v/web-landmark
# mv data/sharegpt4v/web-landmakr.zip data/sharegpt4v/web-landmark
# unzip data/sharegpt4v/web-landmark/web-landmakr.zip -d data/sharegpt4v/web-landmark/images
# mv data/sharegpt4v/web-landmark/images/data/web-landmark/images/* data/sharegpt4v/web-landmark/images
# rm -rf data/sharegpt4v/web-landmark/images/data

# gdown https://drive.google.com/uc?id=1-SB71C3j1mVg0kDDXwj2IWGEoBoRUD-J -O data/sharegpt4v/web-celebrity.zip
# mkdir -p data/sharegpt4v/web-celebrity
# mv data/sharegpt4v/web-celebrity.zip data/sharegpt4v/web-celebrity
# unzip data/sharegpt4v/web-celebrity/web-celebrity.zip -d data/sharegpt4v/web-celebrity/images
# mv data/sharegpt4v/web-celebrity/images/data/web-celebrity/images/* data/sharegpt4v/web-celebrity/images
# rm -rf data/sharegpt4v/web-celebrity/images/data

# gdown https://drive.google.com/uc?id=1f4v_3e1OJtyYqam1CEp6RenCNTU5_mG2 -O data/sharegpt4v/share_textvqa.zip
# mkdir -p data/sharegpt4v/share_textvqa
# mv data/sharegpt4v/share_textvqa.zip data/sharegpt4v/share_textvqa
# unzip data/sharegpt4v/share_textvqa/share_textvqa.zip -d data/sharegpt4v/share_textvqa/images
# mv data/sharegpt4v/share_textvqa/images/data/share_textvqa/images/* data/sharegpt4v/share_textvqa/images
# rm -rf data/sharegpt4v/share_textvqa/images/data


python prepare_sharegpt4v.py --dataset_path "Lin-Chen/ShareGPT4V" --image_save_dir "data/sharegpt4v" --output_file "data/ShareGPT4V_train.json"
