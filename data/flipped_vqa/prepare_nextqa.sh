mkdir -p data
cd ./data

# obtain NExTQA dataset
mkdir -p nextqa
cd ./nextqa

# get train split
if [ ! -f "train.csv" ]; then
    echo "Downloading NExTQA training dataset from Google Drive"
    wget https://drive.google.com/file/d/1Ok_5UcDVmkFfGhtLdXfIMS0ShX3E_15c/view?usp=sharing -O train.csv
else
    echo "NExTQA training dataset already exists"
fi

# get validation split
if [ ! -f "val.csv" ]; then
    echo "Downloading NExTQA validation dataset from Google Drive"
    wget https://drive.google.com/file/d/1cO989o4tpPwkyAZF9pjW6IoKmZNS1tha/view?usp=sharing -O val.csv
else
    echo "NExTQA validation dataset already exists"
fi

# get mapping file
if [ ! -f "map_vid_vidorID.json" ];  then
    echo "Downloading NExTQA video-to-path mapping file from Google Drive"
    wget https://drive.google.com/file/d/1NFAOQYZ-D0LOpcny8fm0fSzLvN0tNR2A/view?usp=sharing -O map_vid_vidorID.json
else
    echo "NExTQA mapping file already exists"
fi

cd ..

python ../prepare_nextqa.py --output_file "./train_example_nextqa.json"
