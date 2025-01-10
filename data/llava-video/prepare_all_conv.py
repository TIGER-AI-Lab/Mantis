import fire
import json
import os
import datasets
from tqdm import tqdm
import av
import decord
import math
import numpy as np
import time
import threading
from PIL import Image
from pathlib import Path
from collections import defaultdict
def read_video_decord(video_path, indices):
    '''
    Decode the video with Decord decoder.
    
    Args:
        video_path (str): Path to the video file.
        indices (List[int]): List of frame indices to decode.
        
    Returns:
        np.ndarray: np array of decoded frames of shape (num_frames, height, width, 3).
    '''
   
    
    # Set Decord to use CPU for decoding
    decord.bridge.set_bridge('numpy')
    
    if len(indices) == 0:
        indices = [0]
        print("No indices to decode, might be an empty video please check")
    
    # Load video with Decord
    vr = decord.VideoReader(video_path)
    
    # Decode frames at specified indices
    frames = vr.get_batch(indices).asnumpy()
    
    # Decord returns frames in (N,H,W,C) format by default, same as PyAV
    return frames

def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.

    Args:
        container (av.container.input.InputContainer): PyAV container.
        indices (List[int]): List of frame indices to decode.

    Returns:
        np.ndarray: np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    if len(indices) == 0:
        # to debug
        indices = [0]
        print("No indices to decode, might be an empty video please check")
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

def timeout_handler(signum, frame):
    raise TimeoutError("Function call timed out")

def with_timeout(timeout):
    def decorator(func):
        def wrapper(*args, **kwargs):
            result = [TimeoutError(f"Function call timed out (timeout={timeout})")]
            stop_event = threading.Event()

            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    result[0] = e

            thread = threading.Thread(target=target)
            thread.start()
            thread.join(timeout)
            if thread.is_alive():
                stop_event.set()
                raise TimeoutError(f"Function call timed out (timeout={timeout})")
            if isinstance(result[0], Exception):
                raise result[0]
            return result[0]
        return wrapper
    return decorator

def main(
    data_dir="./llava-video-data",
    qa_types="oe_qa,mc_qa,cap",
    output_dir="./llava-video-data"
):
    qa_types = qa_types.split(",") if isinstance(qa_types, str) else qa_types
    all_data = []
    oe_qa_post_fix = "_oe_qa_processed.json"
    mc_qa_post_fix = "_mc_qa_processed.json"
    cap_post_fix = "_cap_processed.json"
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    for subset_folder in data_dir.iterdir():
        print(f"Processing {subset_folder}")
        if not subset_folder.is_dir():
            continue
        subset_name = subset_folder.name
        if subset_name in ['gpt4o_qa_prompt', 'gpt4o_caption_prompt']:
            continue
        num_json_files = len(list(subset_folder.glob("*.json")))
        if subset_name.endswith("_v0_1"):
            cap_file = subset_folder / f"{subset_name}{cap_post_fix}"
            oe_qa_file = subset_folder / f"{subset_name.replace('_v0_1', '')}_oe_v0_1_qa_processed.json"
            mc_qa_file = subset_folder / f"{subset_name.replace('_v0_1', '')}_mc_v0_1_qa_processed.json"
        else:
            cap_file = subset_folder / f"{subset_name}{cap_post_fix}"
            oe_qa_file = subset_folder / f"{subset_name}{oe_qa_post_fix}"
            mc_qa_file = subset_folder / f"{subset_name}{mc_qa_post_fix}"
        if subset_name == "llava_hound":
            assert oe_qa_file.exists(), f"oe_qa_file: {oe_qa_file}. exists: {oe_qa_file.exists()}"
        else:
            assert sum([cap_file.exists(), oe_qa_file.exists(), mc_qa_file.exists()]) == num_json_files, f"cap_file: {cap_file}. exists: {cap_file.exists()}. oe_qa_file: {oe_qa_file}. exists: {oe_qa_file.exists()}. mc_qa_file: {mc_qa_file}. exists: {mc_qa_file.exists()}"
        qa_type_file_map = {
            "oe_qa": oe_qa_file,
            "mc_qa": mc_qa_file,
            "cap": cap_file
        }
        for qa_type in qa_types:
            file = qa_type_file_map[qa_type]
            if not file.exists():
                continue
            with open(file, "r") as f:
                data = json.load(f)
            for item in tqdm(data, desc=f"Processing {subset_name} {qa_type}"):
                item['conversations'][0]['value'] = item['conversations'][0]['value'].replace("<image>", "<video>")
                assert "<video>" in item['conversations'][0]['value']
                # item['video'] = "videos/" + item['video']
                item['video'] = cap_file.parent / item['video']
                if not item['video'].exists():
                    print(f"Video not found: {item['video']}")
                    continue
                all_data.append(item)
    # post
    output_file = output_dir / "all_conv.json" if set(qa_types) == set(["oe_qa", "mc_qa", "cap"]) else output_dir / f"all_conv_{'_'.join(qa_types)}.json"
    for item in all_data:
        item['video'] = str(item['video'].relative_to(output_file.parent))
    # count the number of data_source
    count_data_source = defaultdict(int)
    for item in all_data:
        count_data_source[item['data_source']] += 1
    count_data_source = dict(sorted(count_data_source.items(), key=lambda x: x[1], reverse=True))
    print("Data source count:")
    for data_source, count in count_data_source.items():
        print(f"{data_source}: {count}")
        
    with open(output_file, "w") as f:
        json.dump(all_data, f, indent=4)
    print(f"Processed {len(all_data)} items")
    print(f"Saved to {output_file}")        
    
if __name__ == "__main__":
    fire.Fire(main)