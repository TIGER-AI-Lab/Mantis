#!/bin/bash
mkdir -p ./data
cd data

if [ -d 'nights' ]; then
    echo "NIGHTS dataset already exists"
else
    # Download NIGHTS dataset
    wget -O nights.zip https://data.csail.mit.edu/nights/nights.zip

    unzip nights.zip
    rm nights.zip
fi


cd ..
python prepare.py --split train --output_file "./data/train.json"
python prepare.py --split val --output_file "./data/val.json"