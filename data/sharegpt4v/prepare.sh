mkdir -p data/sharegpt4v
# sharegptv sam
gdown https://drive.google.com/uc?id=16_5S-l4X01Kv7AUPFzfJ2LZjpw1-lLVF -O data/sharegpt4v/sam_images_share-sft.zip
mkdir -p data/sam
mkdir -p data/sam/images
mv data/sam_images_share-sft.zip data/sam
unzip data/sam/sam_images_share-sft.zip -d data/sam/images

gdown https://drive.google.com/uc?id=1FxB2Nw-vWUcTUSI_dBpPIykb-uGYoEqV -O data/sharegpt4v/wikiart.zip
mkdir -p data/sharegpt4v/wikiart
mv data/sharegpt4v/wikiart.zip data/sharegpt4v/wikiart
unzip data/sharegpt4v/wikiart/wikiart.zip -d data/sharegpt4v/wikiart/images
mv data/sharegpt4v/wikiart/images/data/wikiart/images/* data/sharegpt4v/wikiart/images
rm -rf data/sharegpt4v/wikiart/images/data

gdown https://drive.google.com/uc?id=1JpJkN7ZMA50xAhMx9O-rVb5yLhfGm3_o -O data/sharegpt4v/web-landmakr.zip
mkdir -p data/sharegpt4v/web-landmark
mv data/sharegpt4v/web-landmakr.zip data/sharegpt4v/web-landmark
unzip data/sharegpt4v/web-landmark/web-landmakr.zip -d data/sharegpt4v/web-landmark/images
mv data/sharegpt4v/web-landmark/images/data/web-landmark/images/* data/sharegpt4v/web-landmark/images
rm -rf data/sharegpt4v/web-landmark/images/data

gdown https://drive.google.com/uc?id=1-SB71C3j1mVg0kDDXwj2IWGEoBoRUD-J -O data/sharegpt4v/web-celebrity.zip
mkdir -p data/sharegpt4v/web-celebrity
mv data/sharegpt4v/web-celebrity.zip data/sharegpt4v/web-celebrity
unzip data/sharegpt4v/web-celebrity/web-celebrity.zip -d data/sharegpt4v/web-celebrity/images
mv data/sharegpt4v/web-celebrity/images/data/web-celebrity/images/* data/sharegpt4v/web-celebrity/images
rm -rf data/sharegpt4v/web-celebrity/images/data

gdown https://drive.google.com/uc?id=1f4v_3e1OJtyYqam1CEp6RenCNTU5_mG2 -O data/sharegpt4v/share_textvqa.zip
mkdir -p data/sharegpt4v/share_textvqa
mv data/sharegpt4v/share_textvqa.zip data/sharegpt4v/share_textvqa
unzip data/sharegpt4v/share_textvqa/share_textvqa.zip -d data/sharegpt4v/share_textvqa/images
mv data/sharegpt4v/share_textvqa/images/data/share_textvqa/images/* data/sharegpt4v/share_textvqa/images
rm -rf data/sharegpt4v/share_textvqa/images/data


python prepare.py --dataset_path "Lin-Chen/ShareGPT4V" --image_save_dir "data/sharegpt4v" --output_file "data/ShareGPT4V_train.json"
