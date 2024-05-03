mkdir -p data
cd ./data

mkdir -p images
mkdir -p resized_images
mkdir -p cluster_images
if [ ! -d "resized_images" ]; then
    echo "Downloading resized_images from Google Drive"
    gdown 'https://drive.google.com/uc?id=1OVb4_3Uec_xbyUk90aWC6LFpKsIOtR7v' # resized_images.zip
    unzip resized_images.zip # resized_images
    rm resized_images.zip
fi
if [ ! -f "params_eps_20min_samples_9__allrectangles_corrected.obj" ]
then
    echo "Downloading params_eps_20min_samples_9__allrectangles_corrected.obj"
    wget https://github.com/harsh19/spot-the-diff/raw/master/data/params_eps_20min_samples_9__allrectangles_corrected.obj

fi

if [ ! -f "train.json" ]
then
    echo "Downloading train.json"
    wget https://github.com/harsh19/spot-the-diff/raw/master/data/annotations/train.json
fi
# if [ ! -f "val.json" ]
# then
#     echo "Downloading val.json"
#     wget https://github.com/harsh19/spot-the-diff/raw/master/data/annotations/val.json
# fi
# if [ ! -f "test.json" ]
# then
#     echo "Downloading test.json"
#     wget https://github.com/harsh19/spot-the-diff/raw/master/data/annotations/test.json
# fi

python ../prepare.py --input_file "test.json" --output_file "test_vqa.json"