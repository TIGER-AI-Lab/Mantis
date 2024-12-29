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
    max_num_frames:int=64,
    fps=2,
    video_reader_engine="pyav"
):
    input_file = Path(f"./data/{subset_name}/{subset_name}_cap_processed.json")
    with open(f"./data/{subset_name}/{subset_name}_cap_processed.json", "r") as f:
        data = json.load(f)
    
    new_data = []
    for item in tqdm(data):
        # item['conversations'][0]['value'] = item['conversations'][0]['value'].replace("<image>", "<video>")
        item['text'] = item['conversations'][1]['value']
        item['video'] = "videos/" + item['video']
        if not os.path.exists(f"./data/{subset_name}/{item['video']}"):
            continue
        del item['conversations']
        new_data.append(item)
        assert os.path.exists(f"./data/{subset_name}/{item['video']}"), f"./data/{subset_name}/{item['video']}"
    data = new_data
    with open(f"./data/{subset_name}/{subset_name}_cap_processed_train.json", "w") as f:
        json.dump(data, f, indent=4)
    print(f"Processed {len(data)} items")
    print(f"Saved to ./data/{subset_name}/{subset_name}_cap_processed_train.json")
    
    
    frames_folder = f"./data/{subset_name}/videos_frames_{fps}fps_{max_num_frames}frames"
    
    
    def process_func(item):
        video_file = f"data/{subset_name}/" + item['video']
        # if video_file != "data/0_30_s_youtube_v0_1/videos/liwei_youtube_videos/videos/youtube_video_2024/ytb_mvGBEtO-HI8.mp4":
        #     return None
        timeout = 3
        @with_timeout(timeout)
        def read_video():
            start = time.time()
            video_file = f"data/{subset_name}/" + item['video']
            # print(f"Reading video {video_file}")
            if video_reader_engine == "decord":
                video_reader = decord.VideoReader(str(video_file))
                total_frames = len(video_reader)
                video_fps = video_reader.get_avg_fps()
            elif video_reader_engine == "pyav":
                container = av.open(video_file, timeout=timeout)
                total_frames = container.streams.video[0].frames
                video_fps = container.streams.video[0].average_rate
                video_fps = video_fps.numerator / video_fps.denominator
            
            if max_num_frames and total_frames > max_num_frames:
                if fps:
                    interval = math.ceil(video_fps / fps)
                    indices = np.arange(0, total_frames, interval).astype(int)
                    if len(indices) > max_num_frames:
                        indices = indices[:max_num_frames]
                else:
                    indices = np.arange(0, total_frames, total_frames / max_num_frames).astype(int)
                # print(f"Sample {len(indices)} frames from {total_frames} frames")
            else:
                indices = np.arange(total_frames)
            # print(f"Decoding video {video_file} with indices {indices}")
            if video_reader_engine == "decord":
                try:
                    video_frames = video_reader.get_batch(indices).asnumpy()
                except:
                    # If batch decoding fails, try one by one
                    print("Batch decoding failed, trying sequential decoding")
                    video_frames = []
                    for idx in indices:
                        try:
                            frame = video_reader[idx].asnumpy()
                            video_frames.append(frame)
                        except:
                            print(f"Failed to decode frame at index {idx}")
                            continue
                            
                    if not video_frames:
                        print("Failed to decode any frames")
                        return None
                    video_frames = np.stack(video_frames)
                # video_frames = video_reader.get_batch(indices).asnumpy()
            elif video_reader_engine == "pyav":
                video_frames = read_video_pyav(container, indices)
            end = time.time()
            # print(f"Decoded {len(video_frames)} frames from {video_file} in {end-start:.2f}s")
            # save_frames
            images = []
            _frames_folder = f"{frames_folder}/{item['video'].split('/')[-1]}"
            os.makedirs(_frames_folder, exist_ok=True)
            for i, frame in enumerate(video_frames):
                frame_path = f"{_frames_folder}/{i}.jpg"
                if os.path.exists(frame_path):
                    images.append(frame_path)
                    continue
                frame = frame[:, :, ::-1]
                frame = Image.fromarray(frame)
                frame.save(frame_path)
                images.append(frame_path)
            # print(f"Saved frames to {_frames_folder}")
            return images, video_fps
        # item['images'], video_fps = read_video()
        try:
            result = read_video()
            item['images'], video_fps = result
            item['images'] = [str(Path(x).relative_to(input_file.parent)) for x in item['images']]
            item['fps'] = video_fps
            item['video_file'] = item.pop('video')
        except TimeoutError as e:
            item['images'] = None
            item['fps'] = None
            item['video_file'] = item.pop('video')
            print(video_file)
            print(e)
        except av.error.InvalidDataError as e:
            item['images'] = None
            item['fps'] = None
            item['video_file'] = item.pop('video')
            print("InvalidDataError")
            print(video_file)
            print(e)
        except av.error.ExitError as e:
            item['images'] = None
            item['fps'] = None
            item['video_file'] = item.pop('video')
            print("Immediate exit requested")
            print(video_file)
            print(e)
        except Exception as e:
            item['images'] = None
            item['fps'] = None
            item['video_file'] = item.pop('video')
            print(video_file)
            print(e)
        item['sampled_fps'] = fps
        return item

    data = data[:10000]        
    subdatasets = []
    subdataset_size = 1000
    for i in tqdm(range(0, len(data), subdataset_size), desc="Processing subdatasets"):
        subdataset = data[i:i+subdataset_size]
        subdataset = datasets.Dataset.from_list(subdataset)
        subdataset = subdataset.map(process_func, desc="Processing videos", num_proc=16)
        subdataset = subdataset.filter(lambda x: x['images'] is not None)
        subdatasets.append(subdataset)
    dataset = datasets.concatenate_datasets(subdatasets)
    
    # dataset = dataset.map(process_func, desc="Processing videos", num_proc=16)
    # dataset = dataset.filter(lambda x: x['images'] is not None)
    
    with open(f"./data/{subset_name}/{subset_name}_cap_processed_train.frames.json", "w") as f:
        json.dump([x for x in dataset], f, indent=4)
    print(f"Saved to ./data/{subset_name}/{subset_name}_cap_processed_train.frames.json")
if __name__ == "__main__":
    fire.Fire(main)