mkdir -p data
cd data

wget "https://huggingface.co/datasets/q-future/Co-Instruct-DB/resolve/main/coinstruct_562k_llava_format.json?download=true" -O coinstruct_562k_llava_format.json

wget "https://huggingface.co/datasets/q-future/Co-Instruct-DB/resolve/main/co-instruct-images.tar?download=true" -O co-instruct-images.tar
mkdir -p images
cd images && tar -xvf co-instruct-images.tar && cd ..
rm co-instruct-images.tar