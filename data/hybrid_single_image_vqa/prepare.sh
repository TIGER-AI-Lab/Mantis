
mkdir data

# ai2d
wget https://ai2-public-datasets.s3.amazonaws.com/diagrams/ai2d-all.zip -O data
python prepare_ai2d.py

# chartqa
wget "https://huggingface.co/datasets/ahmed-masry/ChartQA/resolve/main/ChartQA%20Dataset.zip?download=true" -O data/chartqa.zip
mv data/chartqa.zip data/chartqa
unzip data/chartqa/chartqa.zip
ln -s data/chartqa/ChartQA\ Dataset data/train/png data/chartqa/train_images
ln -s data/chartqa/ChartQA\ Dataset data/val/png data/chartqa/val_images
ln -s data/chartqa/ChartQA\ Dataset data/test/png data/chartqa/test_images
python prepare_chartqa.py --split train
python prepare_chartqa.py --split val
python prepare_chartqa.py --split test

# docvqa
python prepare_docvqa.py

# dvqa
mkdir data/dvqa
gdown https://drive.google.com/uc?id=1iKH2lTi1-QxtNUVRxTUWFvUvRHq6HAsZ -O data/dvqa/images.tar.gz
gdown https://drive.google.com/uc?id=1VKYd3kaiCFziSsSv4SgQJ2T5m7jxuh5u -O data/dvqa/qa.tar.gz
tar -xvf data/dvqa/images.tar.gz -C data/dvqa
tar -xvf data/dvqa/qa.tar.gz -C data/dvqa
python prepare_dvqa.py