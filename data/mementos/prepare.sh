mkdir -p data
cd data

mkdir -p images
cd images
# download images

if [ -f "image_cmc/.done" ]; then
    echo "image_cmc already downloaded"
else
    gdown https://drive.google.com/uc?id=1vRZxH2NbDwj9tw-YfcrJF_EkZ0kUGj0K -O image_comics.zip
    unzip image_comics.zip # -> image_cmc
    touch image_cmc/.done
    rm image_comics.zip
fi

if [ -f "image_dl/.done" ]; then
    echo "image_dl already downloaded"
else
    gdown https://drive.google.com/uc?id=1d3Cux3n6nPfJ2HH_yTqtHNYr7iRVLe9a -O image_dl.zip
    unzip image_dl.zip # -> image
    mv image image_dl
    touch image_dl/.done
    rm image_dl.zip
fi

if [ -f "image_robo/.done" ]; then
    echo "image_robotics already downloaded"
else
    gdown https://drive.google.com/uc?id=162hyoZNHf12b-X1bjHV8zlIBd6pvvrYD -O image_robotics.zip
    unzip image_robotics.zip # -> image_robo
    touch image_robo/.done
    rm image_robotics.zip
fi

if [ -f "single_image_cmc/.done" ]; then
    echo "single_image_comics already downloaded"
else
    gdown https://drive.google.com/uc?id=1L-N7NFYy-nyv6sIkF5RSKM1lhAT7WC6O -O single_image_comics.zip
    unzip single_image_comics.zip # -> single_image_cmc
    touch single_image_cmc/.done
    rm single_image_comics.zip
fi

if [ -f "single_image_dl/.done" ]; then
    echo "single_image_dl already downloaded"
else
    gdown https://drive.google.com/uc?id=1oyMzmjudqUzL9pi5pChl0RTU1eaxzR2o -O single_image_dl.zip
    unzip single_image_dl.zip # -> single_image
    mv single_image single_image_dl
    touch single_image_dl/.done
    rm single_image_dl.zip
fi

if [ -f "single_image_robo/.done" ]; then
    echo "single_image_robotics already downloaded"
else
    gdown https://drive.google.com/uc?id=1s1gni6lpWlxtSqBM6qbvm98bh1u4caSy -O single_image_robotics.zip
    unzip single_image_robotics.zip # -> single_image_robo
    touch single_image_robo/.done
    rm single_image_robotics.zip
fi

cd ..

wget https://github.com/si0wang/Mementos/raw/main/cmc_description.csv -O cmc_description.csv
wget https://github.com/si0wang/Mementos/raw/main/dl_description.csv -O dl_description.csv
wget https://github.com/si0wang/Mementos/raw/main/robo_description.csv -O robo_description.csv

cd ..
python prepare.py


# gdown https://drive.google.com/uc?id=1vRZxH2NbDwj9tw-YfcrJF_EkZ0kUGj0K -O image_comics.zip
# unzip image_comics.zip # -> image_cmc

# gdown https://drive.google.com/uc?id=1d3Cux3n6nPfJ2HH_yTqtHNYr7iRVLe9a -O image_dl.zip
# unzip image_dl.zip

# gdown https://drive.google.com/uc?id=162hyoZNHf12b-X1bjHV8zlIBd6pvvrYD -O image_robotics.zip
# unzip image_robotics.zip

# gdown https://drive.google.com/uc?id=1L-N7NFYy-nyv6sIkF5RSKM1lhAT7WC6O -O single_image_comics.zip
# unzip single_image_comics.zip

# gdown https://drive.google.com/uc?id=1oyMzmjudqUzL9pi5pChl0RTU1eaxzR2o -O single_image_dl.zip
# unzip single_image_dl.zip

# gdown https://drive.google.com/uc?id=1s1gni6lpWlxtSqBM6qbvm98bh1u4caSy -O single_image_robotics.zip
# unzip single_image_robotics.zip