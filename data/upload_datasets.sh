train_repo_id="TIGER-Lab/Mantis-Instruct"
val_repo_id="TIGER-Lab/Mantis-Instruct"
test_repo_id="TIGER-Lab/MIQA-Eval"
train_mode="zip" # "parquet" or "zip"
val_mode="zip"
test_mode="parquet"
# split: train, val, test


# python upload_conv_dataset_to_hf.py --repo_id $train_repo_id --image_upload_mode $train_mode --split train --dataset_name nlvr2 --dataset_file nlvr2/data/train_v3.json --image_dir nlvr2/data/train_images
# python upload_conv_dataset_to_hf.py --repo_id $val_repo_id --image_upload_mode $val_mode --split val --dataset_name nlvr2 --dataset_file nlvr2/data/dev_v3.json --image_dir nlvr2/data/dev_images
# python upload_test_dataset_to_hf.py --repo_id $test_repo_id --image_upload_mode $test_mode --split test --dataset_name nlvr2 --dataset_file nlvr2/data/test_v3.json --image_dir nlvr2/data/test_images

# python upload_conv_dataset_to_hf.py --repo_id $train_repo_id --image_upload_mode $train_mode --split train --dataset_name spot-the-diff --dataset_file mimicit/data/SD.json --image_dir mimicit/data/SD_images

# python upload_conv_dataset_to_hf.py --repo_id $train_repo_id --image_upload_mode $train_mode --split train --dataset_name visual_story_telling --dataset_file mimicit/data/VST.json --image_dir mimicit/data/VST_images

# python upload_conv_dataset_to_hf.py --repo_id $train_repo_id --image_upload_mode $train_mode --split train --dataset_name birds-to-words --dataset_file birds-to-words/data/train.json --image_dir birds-to-words/data/images
# python upload_conv_dataset_to_hf.py --repo_id $val_repo_id --image_upload_mode $val_mode --split val --dataset_name birds-to-words --dataset_file birds-to-words/data/val.json --image_dir birds-to-words/data/images
# python upload_test_dataset_to_hf.py --repo_id $test_repo_id --image_upload_mode $test_mode --split test --dataset_name birds-to-words --dataset_file birds-to-words/data/test.json --image_dir birds-to-words/data/images

# python upload_conv_dataset_to_hf.py --repo_id $train_repo_id --image_upload_mode $train_mode --split train --dataset_name synthetic --dataset_file synthetic/data/train.json --image_dir synthetic/data/images_sdturbo-1

# python upload_conv_dataset_to_hf.py --repo_id $train_repo_id --image_upload_mode $train_mode --split train --dataset_name llava_665k_multi --dataset_file llava-instruct/data/train.json --image_dir llava-instruct/data

# python upload_conv_dataset_to_hf.py --repo_id $train_repo_id --image_upload_mode $train_mode --split train --dataset_name lrv --dataset_file lrv/data/train_conv.json --image_dir lrv/data/images

# mementos test evaluation datasets
# for dataset in "image_cmc" "image_dl" "image_robo" "single_image_cmc" "single_image_dl" "single_image_robo"
# do
#     python upload_test_dataset_to_hf.py --repo_id $test_repo_id --image_upload_mode $test_mode --split $dataset --dataset_name mementos --dataset_file mementos/data/$dataset.json --image_dir mementos/data/images
# done

# python upload_test_dataset_to_hf.py --repo_id $test_repo_id --image_upload_mode $test_mode --split test --dataset_name mantis_eval --dataset_file human_collected/merged/data.json --image_dir human_collected/merged/images

# python upload_test_dataset_to_hf.py --repo_id $test_repo_id --image_upload_mode $test_mode --split test --dataset_name imagenhub --dataset_file human_collected/"wenhu chen"/data_imagenhub.json --image_dir human_collected/"wenhu chen"/images

# python upload_test_dataset_to_hf.py --repo_id $test_repo_id --image_upload_mode $test_mode --split test --dataset_name synthetic_eval --dataset_file synthetic/data/test.json --image_dir synthetic/data/test_images


# python upload_conv_dataset_to_hf.py --repo_id $train_repo_id --image_upload_mode $train_mode --split train --dataset_name nlvr2_true --dataset_file nlvr2/data/train_v4.json --image_dir nlvr2/data/train_images
# python upload_conv_dataset_to_hf.py --repo_id $val_repo_id --image_upload_mode $val_mode --split val --dataset_name nlvr2_true --dataset_file nlvr2/data/dev_v4.json --image_dir nlvr2/data/dev_images

# python upload_conv_dataset_to_hf.py --repo_id $train_repo_id --image_upload_mode $train_mode --split train --dataset_name contrastive_caption --dataset_file contrastive_caption/data/train.json --image_dir contrastive_caption/data

# python upload_conv_dataset_to_hf.py --repo_id $train_repo_id --image_upload_mode $train_mode --split train --dataset_name dreamsim --dataset_file dream_sim/data/train.json --image_dir dream_sim/data/
# python upload_conv_dataset_to_hf.py --repo_id $train_repo_id --image_upload_mode $train_mode --split train --dataset_name dreamsim --dataset_file dream_sim/data/val.json --image_dir dream_sim/data



# python upload_conv_dataset_to_hf.py --repo_id $train_repo_id --image_upload_mode $train_mode --split train --dataset_name nlvr2 --dataset_file nlvr2/data/train_v5.json --image_dir nlvr2/data/train_images --upload_zip_images False
# python upload_conv_dataset_to_hf.py --repo_id $val_repo_id --image_upload_mode $val_mode --split val --dataset_name nlvr2 --dataset_file nlvr2/data/dev_v5.json --image_dir nlvr2/data/dev_images --upload_zip_images False

# python upload_conv_dataset_to_hf.py --repo_id $train_repo_id --image_upload_mode $train_mode --split train --dataset_name dreamsim --dataset_file dream_sim/data/train.json --image_dir dream_sim/data/ --upload_zip_images False
# python upload_conv_dataset_to_hf.py --repo_id $train_repo_id --image_upload_mode $train_mode --split val --dataset_name dreamsim --dataset_file dream_sim/data/val.json --image_dir dream_sim/data --upload_zip_images False

# python upload_conv_dataset_to_hf.py --repo_id $train_repo_id --image_upload_mode $train_mode --split train --dataset_name llava_665k_multi --dataset_file llava-instruct/data/train.json --image_dir llava-instruct/data

# python upload_conv_dataset_to_hf.py --repo_id $train_repo_id --image_upload_mode $train_mode --split train --dataset_name lrv --dataset_file lrv/data/train_conv.json --image_dir lrv/data/images --upload_zip_images False

# python upload_conv_dataset_to_hf.py --repo_id $train_repo_id --image_upload_mode $train_mode --split train --dataset_name nlvr2_merged_img --dataset_file nlvr2/data/train_v6.json --image_dir nlvr2/data/train_merged_images --upload_zip_images True
# python upload_conv_dataset_to_hf.py --repo_id $val_repo_id --image_upload_mode $val_mode --split val --dataset_name nlvr2_merged_img --dataset_file nlvr2/data/dev_v6.json --image_dir nlvr2/data/dev_merged_images --upload_zip_images True


# python upload_conv_dataset_to_hf.py --repo_id $train_repo_id --image_upload_mode $train_mode --split train --dataset_name ai2d --dataset_file hybrid_vqa/data/ai2d/train.json --image_dir hybrid_vqa/data/ai2d/images

# python upload_conv_dataset_to_hf.py --repo_id $train_repo_id --image_upload_mode $train_mode --split train --dataset_name chartqa --dataset_file hybrid_vqa/data/chartqa/train.json --image_dir hybrid_vqa/data/chartqa/train_images
# python upload_conv_dataset_to_hf.py --repo_id $val_repo_id --image_upload_mode $val_mode --split val --dataset_name chartqa --dataset_file hybrid_vqa/data/chartqa/val.json --image_dir hybrid_vqa/data/chartqa/val_images
# python upload_test_dataset_to_hf.py --repo_id $test_repo_id --image_upload_mode $test_mode --split test --dataset_name chartqa --dataset_file hybrid_vqa/data/chartqa/test.json --image_dir hybrid_vqa/data/chartqa/test_images

# python upload_conv_dataset_to_hf.py --repo_id $train_repo_id --image_upload_mode $train_mode --split train --dataset_name docvqa --dataset_file hybrid_vqa/data/docvqa/train.json --image_dir hybrid_vqa/data/docvqa/images

# python upload_conv_dataset_to_hf.py --repo_id $train_repo_id --image_upload_mode $train_mode --split train --dataset_name dvqa --dataset_file hybrid_vqa/data/dvqa/train.json --image_dir hybrid_vqa/data/dvqa/images

# python upload_conv_dataset_to_hf.py --repo_id $train_repo_id --image_upload_mode $train_mode --split train --dataset_name coinstruct --dataset_file coinstruct/data/train.json --image_dir coinstruct/data/images

# python upload_conv_dataset_to_hf.py --repo_id $train_repo_id --image_upload_mode $train_mode --split train --dataset_name multi_vqa --dataset_file multi_vqa/data/train.json --image_dir multi_vqa/data

# python upload_conv_dataset_to_hf.py --repo_id $train_repo_id --image_upload_mode $train_mode --split train --dataset_name iconqa --dataset_file iconqa/data/train.json --image_dir iconqa/data

# python upload_conv_dataset_to_hf.py --repo_id $train_repo_id --image_upload_mode $train_mode --split train --dataset_name imagecode --dataset_file imagecode/data/train.json --image_dir imagecode/data

# python upload_conv_dataset_to_hf.py --repo_id "TIGER-Lab/OBELICS-min" --image_upload_mode $train_mode --split train --dataset_name obelics_100k_multi --dataset_file obelics/data/train.jsonl --image_dir obelics/data/images --max_zip_size "10G"

# python upload_conv_dataset_to_hf.py --repo_id "TIGER-Lab/llava-data" --image_upload_mode $train_mode --split train --dataset_name llava_pretrain --dataset_file llava-pretrain/data/train.json --image_dir llava-pretrain/data

# python upload_conv_dataset_to_hf.py --repo_id "TIGER-Lab/llava-data" --image_upload_mode $train_mode --split train --dataset_name llava_instruct --dataset_file llava-instruct/data/llava_v1_5_mix665k.json --image_dir llava-instruct/data
