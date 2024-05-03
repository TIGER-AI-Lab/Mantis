mkdir -p data
cd ./data
# llava-150k-instruct
wget https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/llava_v1_5_mix665k.json?download=true -O llava_v1_5_mix665k.json

# COCO
echo "Downloading COCO images..."
mkdir -p coco
cd coco
wget http://images.cocodataset.org/zips/train2017.zip -O coco_train2017.zip
unzip coco_train2017.zip
rm coco_train2017.zip
cd ..
echo "Finished COCO images."

# GQA images
echo "Downloading GQA images..."
mkdir -p gqa
cd gqa
wget https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip -O GQA_images.zip
unzip GQA_images.zip
rm GQA_images.zip
cd ..
echo "Finished GQA images."

# OCR VQA
echo "Downloading OCR VQA images..."
mkdir -p ocr_vqa
cd ocr_vqa
gdown https://drive.google.com/uc?id=1r0tyZUwGCc4wIG4RkiglCGNL_nFJjR6Q
gdown https://drive.google.com/uc?id=16eqkNbgARX1aLM4q0l5WBiPPSFbK0Elp
python loadDataset.py # download images, might need to remove the pdb.set_trace() in the script
cd ..
echo "Finished OCR VQA images."

# Text VQA
echo "Downloading TextVQA images..."
mkdir -p textvqa
cd textvqa
wget https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip
unzip train_val_images.zip
rm train_val_images.zip
cd ..
echo "Finished TextVQA images."

# VisualGenome
echo "Downloading VisualGenome images..."
mkdir -p vg
cd vg
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip -O VG_part1.zip
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip -O VG_part2.zip
unzip VG_part1.zip
unzip VG_part2.zip
rm VG_part1.zip
rm VG_part2.zip
cd ..
echo "Finished VisualGenome images."


cd ..
# prepare the data
python prepare.py # -> llava_v1_5_mix665k_merged.json
cd data
ln -s llava_v1_5_mix665k_merged.json train.json