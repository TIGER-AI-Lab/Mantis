mkdir -p data
cd data

subset_name=0_30_s_youtube_v0_1

huggingface-cli download --repo-type dataset lmms-lab/LLaVA-Video-178K ${subset_name}/${subset_name}_cap_processed.json --local-dir .

for i in {1..20}; do
    echo "Downloading ${subset_name}_videos_$i.tar.gz"
    huggingface-cli download --repo-type dataset lmms-lab/LLaVA-Video-178K ${subset_name}/${subset_name}_videos_$i.tar.gz --local-dir .
done

cd ${subset_name}
mkdir videos
for i in {1..20}; do
    echo "Extracting ${subset_name}_videos_$i.tar.gz"
    tar -xvf ${subset_name}_videos_$i.tar.gz -C videos
done

cd ../..
python prepare_cap.py --subset_name ${subset_name} # not used for qwen2 vl vae
python prepare_cap_conv.py --subset_name ${subset_name} # only use this for qwen2 vl vae
