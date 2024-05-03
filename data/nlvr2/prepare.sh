#! /bin/bash

mkdir -p data
cd ./data

if [ ! -f "train.json" ]; then
    wget -O train.jsonl https://raw.githubusercontent.com/lil-lab/nlvr/master/nlvr2/data/train.json 
fi

if [ ! -f "test.json" ]; then
    wget -O test.jsonl https://raw.githubusercontent.com/lil-lab/nlvr/master/nlvr2/data/test1.json 
fi

if [ ! -f "dev.json" ]; then
    wget -O dev.jsonl https://raw.githubusercontent.com/lil-lab/nlvr/master/nlvr2/data/dev.json
fi

if [ ! -f "./dev_images/.done" ]; then
    wget https://lil.nlp.cornell.edu/resources/NLVR2/dev_img.zip
    unzip dev_img.zip
    rm dev_img.zip
    mv dev dev_images
    touch ./dev_images/.done
else
    echo "dev_images already downloaded"
fi

if [ ! -f "./test_images/.done" ]; then
    wget -O test_img.zip https://lil.nlp.cornell.edu/resources/NLVR2/test1_img.zip
    unzip test_img.zip
    rm test_img.zip
    mv test1 test_images
    touch ./test_images/.done
else
    echo "test_images already downloaded"
fi

if [ ! -f "./train_images/.done" ]; then
    wget https://lil.nlp.cornell.edu/resources/NLVR2/train_img.zip
    unzip train_img.zip
    rm train_img.zip
    mv images/train train_images
    rm -rf images
    for i in {0..99};
    do
        mkdir -p train_images/train$i
        mv train_images/$i/* train_images/
        rm -d train_images/$i
    done
    touch ./train_images/.done
else
    echo "train_images already downloaded"
fi


cd ..

# Using ChatGPT to augment the data
# python prepare_conv.py --input_file data/train.jsonl --output_file data/train.json --image_dir data/train_images --demo_file seed_demos.json
# python prepare_conv.py --input_file data/dev.jsonl --output_file data/dev.json --image_dir data/dev_images --demo_file seed_demos.json
# python prepare_conv.py --input_file data/dev.jsonl --output_file data/dev.json --image_dir data/dev_images --demo_file seed_demos.json


# python prepare_conv_v3.py --input_file data/train.jsonl --output_file data/train_v3.json --image_dir data/train_images
# python prepare_conv_v3.py --input_file data/dev.jsonl --output_file data/dev_v3.json --image_dir data/dev_images
# python prepare_vqa_v3.py --input_file data/test.jsonl --output_file data/test_v3.json --image_dir data/test_images

# python prepare_conv_v4.py --input_file data/train.jsonl --output_file data/train_v4.json --image_dir data/train_images
# python prepare_conv_v4.py --input_file data/dev.jsonl --output_file data/dev_v4.json --image_dir data/dev_images

# python prepare_conv_vqa_v5.py --input_file data/train.jsonl --output_file data/train_v5.json --image_dir data/train_images
# python prepare_conv_vqa_v5.py --input_file data/dev.jsonl --output_file data/dev_v5.json --image_dir data/dev_images

python prepare_conv_vqa_v6_merged.py --input_file data/train.jsonl --output_file data/train_v6.json --image_dir data/train_images
python prepare_conv_vqa_v6_merged.py --input_file data/dev.jsonl --output_file data/dev_v6.json --image_dir data/dev_images