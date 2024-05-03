mkdir -p data
cd data

wget https://huggingface.co/datasets/q-future/q-bench2/raw/main/q-bench2-a1-dev.jsonl
wget https://huggingface.co/datasets/q-future/q-bench2/raw/main/q-bench2-a1-test.jsonl
wget https://huggingface.co/datasets/q-future/q-bench2/raw/main/q-bench2-a2.jsonl   


wget "https://huggingface.co/datasets/q-future/q-bench2/resolve/main/all_single_images.zip?download=true" -O all_single_images.zip
unzip all_single_images.zip
rm all_single_images.zip

wget "https://huggingface.co/datasets/q-future/q-bench2/resolve/main/lldescribe_compare.zip?download=true" -O lldescribe_compare.zip
unzip lldescribe_compare.zip
rm lldescribe_compare.zip

wget "https://huggingface.co/datasets/q-future/q-bench2/resolve/main/llvisionqa_compare_dev.zip?download=true" -O llvisionqa_compare_dev.zip
unzip llvisionqa_compare_dev.zip
rm llvisionqa_compare_dev.zip

wget "https://huggingface.co/datasets/q-future/q-bench2/resolve/main/llvisionqa_compare_test.zip?download=true" -O llvisionqa_compare_test.zip
unzip llvisionqa_compare_test.zip
rm llvisionqa_compare_test.zip

cd ..
python prepare.py --image_mode "pair" --split "dev"
python prepare.py --image_mode "single" --split "dev"
python prepare.py --image_mode "pair" --split "test"
python prepare.py --image_mode "single" --split "test"
