# Set maximum number of parallel processes
MAX_JOBS=4

# Function to wait until we're below MAX_JOBS
wait_for_jobs() {
    while [ $(jobs -p | wc -l) -ge $MAX_JOBS ]; do
        sleep 1
    done
}

huggingface-cli download --repo-type dataset lmms-lab/LLaVA-Video-178K --local-dir llava-video-data

# for llava-video-data/llava_hound
cd llava-video-data/llava_hound
# if already downloaded, skip this step
echo "Downloading ShareGPTVideo/train_video_and_instruction/train_300k"
# huggingface-cli download --repo-type dataset ShareGPTVideo/train_video_and_instruction --include "train_300k/*" --local-dir .
cd train_300k
video_output_dir="../shareVideoGPTV/frames/all_frames/"
mkdir -p $video_output_dir
for file in *.tar.gz; do
    echo "Extracting $file"
    maker=".${file}_extracted"
    if [ -f $maker ]; then
        echo "Already extracted $file"
        continue
    else
        echo "Extracting $file"
        tar -xvf $file -C $video_output_dir && touch $maker &
    fi
    wait_for_jobs
done
if [ ! -f "llava_hound_oe_qa_processed.json" ]; then
    ln -s sharegptvideo_qa_255k_processed.json llava_hound_oe_qa_processed.json # symbolic link for uniform processing in prepare_all_conv.py
fi
cd ../.. # return to llava-video-data



# Main loop
for folder in *; do
    echo "Processing $folder"
    if [ -d "$folder" ]; then
        (
            cd "$folder" || exit
            # break if no tar.gz files found
            num_files=$(ls -1 *.tar.gz 2>/dev/null | wc -l)
            if [ $num_files -eq 0 ]; then
                echo "No tar.gz files found in $folder"
                continue
            fi
            for file in *.tar.gz; do
                maker=".${file}_extracted"
                # Check if files exist (avoid "*.tar.gz" literal match)
                if [ -f "$file" ]; then
                    echo "Extracting $file in $folder"
                    tar -xf "$file" && touch $maker &
                    # break
                fi
            done
            wait  # Wait for tar to complete in this subshell
        ) &  # Run each folder processing in background
        
        # Wait if we've reached max parallel jobs
        wait_for_jobs
    fi
done

# Wait for all remaining background jobs to complete
wait

echo "All extractions complete"

# prepare_all.py
# ["oe_qa", "mc_qa", "cap"],
cd ..
python prepare_all_conv.py --data_dir "./llava-video-data" --output_dir "./llava-video-data" --qa_types "oe_qa,mc_qa,cap"