mkdir -p data
cd data
if [ ! -f birds-to-words-v1.0.tsv ]; then
    echo "Downloading birds-to-words-v1.0.tsv"
    wget https://github.com/google-research-datasets/birds-to-words/raw/master/birds-to-words-v1.0.tsv
else
    echo "birds-to-words-v1.0.tsv already exists"
fi

echo "Reformatting birds-to-words-v1.0.tsv with ChatGPT and saving images"
python ../prepare.py --input_file ./birds-to-words-v1.0.tsv --image_output_dir ./images