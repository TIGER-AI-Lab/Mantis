mkdir -p data
cd ./data

# obtain STAR dataset
mkdir -p star
cd ./star

if [ ! -f "STAR_train.json" ]; then
    echo "Downloading STAR training dataset from Google Drive" 
    wget "https://drive.google.com/file/d/1pTdGZFzy51gOYKS9M3NPBWOzYs_FhMcc/view?usp=sharing" -O STAR_train.json
else
    echo "STAR training dataset already exists"
fi
if [ ! -f "STAR_val.json" ]; then
    echo "Downloading STAR validation dataset from Google Drive" 
    wget "https://drive.google.com/file/d/1yHToKmnpCW-ejqbpBEN7wehionNX8YZx/view?usp=sharing" -O STAR_val.json
else
    echo "STAR validation dataset already exists"
fi
cd ..

python ../prepare_star.py --output_file "./train_example_star.json"



