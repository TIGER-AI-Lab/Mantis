#! /bin/bash

mkdir -p data
cd ./data

# instruction data
if [ ! -f "train_1.json" ]; then
    wget "https://drive.google.com/uc?id=1OlNxY1eC9Fg10LEcEo1xS6cyOWYera_G" -O filter_cap.json
    mv filter_cap.json train_1.json
fi
echo "train_1.json downloaded"

# instruction data without corrdinates
if [ ! -f "train_2.json" ]; then
    wget https://drive.google.com/uc?id=1pWkxE2kqpys1VdwBi99ZXN6-XY5SqhwU -O filter_cap1.json
    mv filter_cap1.json train_2.json
fi
echo "train_2.json downloaded"

# more instruction data
if [ ! -f "train_3.json" ]; then
    wget https://drive.google.com/uc?id=1E83vpCY8ofEDGFZLjpIjPWHHJvocrWVj -O filter_cap_more.json
    mv filter_cap_more.json train_3.json
fi
echo "train_3.json downloaded"

# more instruction data without corrdinates
if [ ! -f "train_4.json" ]; then
    wget https://drive.google.com/uc?id=1NTxkuRPlvDn7aWaJpK_yb0p5r0cxPLNZ -O filter_cap_more1.json
    mv filter_cap_more1.json train_4.json
fi
echo "train_4.json downloaded"

# chart instruction data
if [ ! -f "train_5.json" ]; then
    wget https://drive.google.com/uc?id=13j2U-ectsYGR92r6J5hPdhT8T5ezItHF -O chat.json
    mv chat.json train_5.json
fi
echo "train_5.json downloaded"

mkdir -p images
# instruction images
if [ ! -f "./images/image/.done" ]; then
    gdown https://drive.google.com/uc?id=1k9MNV-ImEV9BYEOeLEIb4uGEUZjd3QbM
    tar -zxvf image.tar.gz
    rm image.tar.gz
    mv image ./images/inst_image
    if [ -d "./images/inst_image" ]; then
        touch ./images/inst_image/.done
    fi
fi
echo "instruction images downloaded"

# chart instruction images
if [ ! -f "./images/chart_image/.done" ]; then
    gdown https://drive.google.com/uc?id=1Dey-undzW2Nl21CYLFSkP_Y4RrfRJkYd
    unzip chart_image.zip
    rm chart_image.zip
    rm -rf __MACOSX
    mv chart_image ./images/chart_image
    if [ -d "./images/chart_image" ]; then
        touch ./images/chart_image/.done
    fi  
fi
echo "chart instruction images downloaded"

cd ..
python prepare_conv_v2.py 