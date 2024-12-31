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
    subset_name:str,
):
    input_file = Path(f"./data/{subset_name}/{subset_name}_cap_processed.json")
    with open(f"./data/{subset_name}/{subset_name}_cap_processed.json", "r") as f:
        data = json.load(f)
    
    new_data = []
    for item in tqdm(data):
        item['conversations'][0]['value'] = item['conversations'][0]['value'].replace("<image>", "<video>")
        item['video'] = "videos/" + item['video']
        if not os.path.exists(f"./data/{subset_name}/{item['video']}"):
            continue
        new_data.append(item)
        assert os.path.exists(f"./data/{subset_name}/{item['video']}"), f"./data/{subset_name}/{item['video']}"
    data = new_data
    with open(f"./data/{subset_name}/{subset_name}_cap_processed_train.conv.json", "w") as f:
        json.dump(data, f, indent=4)
    print(f"Processed {len(data)} items")
    print(f"Saved to ./data/{subset_name}/{subset_name}_cap_processed_train.conv.json")
if __name__ == "__main__":
    fire.Fire(main)