mkdir data
cd data

wget https://iconqa2021.s3.us-west-1.amazonaws.com/iconqa_data.zip
unzip iconqa_data.zip

cd ..
python prepare.py