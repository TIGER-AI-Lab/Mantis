#! /bin/bash

mkdir -p data
cd ./data

if [ ! -f "train.json" ]; then
    gdown https://drive.google.com/uc?id=13j2U-ectsYGR92r6J5hPdhT8T5ezItHF
    mv chart_release_update.json train.json
fi
echo "train.json downloaded"

if [ ! -f "images.done" ]; then
    gdown https://drive.google.com/uc?id=1Dey-undzW2Nl21CYLFSkP_Y4RrfRJkYd
    unzip chart_image.zip
    rm chart_image.zip
    rm -r __MACOSX
    mv chart_image image
    if [ -d "image" ]; then
        touch images.done
    fi
fi
echo "images downloaded"

cd ..

# python prepare_conv.py 