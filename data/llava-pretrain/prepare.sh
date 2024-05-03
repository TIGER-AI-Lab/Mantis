mkdir data
cd data

wget https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/blip_laion_cc_sbu_558k.json?download=true -O blip_laion_cc_sbu_558k.json

mkdir images
cd images
wget https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/images.zip?download=true -O images.zip
unzip images.zip
cd ../..

python prepare.py