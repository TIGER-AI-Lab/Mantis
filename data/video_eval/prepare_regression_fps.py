import json
import cv2
import datasets
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset

all_data=[]

remove_image_placeholders_in_prompt = True
fps=24
num_proc=16
frames_save_dir = Path(f"f{fps}_frames")
frames_save_dir.mkdir(parents=True, exist_ok=True)

def extract_frames_from_video(video_path, fps):
    """
    Extract frames from a video file at a specified FPS rate.
    
    Args:
        video_path (str): Path to the video file
        fps (float): Number of frames to extract per second
        
    Returns:
        list: List of numpy arrays containing the extracted frames
        
    Raises:
        ValueError: If fps is not positive or video_path is invalid
        ImportError: If required libraries are not installed
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        raise ImportError("This function requires opencv-python (cv2) to be installed. "
                         "Install it using: pip install opencv-python")

    if not isinstance(fps, (int, float)) or fps <= 0:
        raise ValueError("FPS must be a positive number")
    
    if isinstance(video_path, Path):
        video_path = str(video_path)
        
    if not isinstance(video_path, str) or not video_path:
        raise ValueError("video_path must be a non-empty string")
        
    # Open the video file
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    frames = []
    frame_interval = 1.0 / fps
    video_fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / video_fps
    
    # Calculate which frames to extract
    target_frame_numbers = []
    current_time = 0
    while current_time < duration:
        frame_number = int(current_time * video_fps)
        if frame_number < frame_count:
            target_frame_numbers.append(frame_number)
        current_time += frame_interval
    
    # Extract the frames
    for frame_number in target_frame_numbers:
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = video.read()
        if ret:
            frames.append(frame)
        else:
            print(f"Warning: Could not read frame {frame_number}")
    
    # Release the video object
    video.release()
    
    return frames


def map_to_train_data(item, video_dir):
    # "p110367_22.jpg"
    # video_path = f"VideoFeedback/annotated/videos_train/{item['id']}.mp4"
    video_path = video_dir / f"{item['id']}.mp4"
    video_frames = extract_frames_from_video(video_path, fps)
    
    # save freams
    frames_save_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = [ frames_save_dir / f"{item['id']}_{i}.jpg" for i in range(len(video_frames)) ]
    for frame, frame_path in zip(video_frames, frame_paths):
        if not Path(frame_path).exists():
            cv2.imwrite(str(frame_path), frame)
    assert all([Path(image).exists() for image in frame_paths]), frame_paths
    labels = [x for x in item['conversations'][1]['value'].split("\n") if x]
    labels = {label.split(":")[0].strip(' \n'): float(label.split(":")[1]) for label in labels}
    prompt = item['conversations'][0]['value']
    if remove_image_placeholders_in_prompt:
        prompt = prompt[:prompt.find("all the frames of video are as follows:")+len("all the frames of video are as follows:")].strip(' \n') + "\n"
    
    return {
        "id": item['id'],
        "images": [str(frame_path) for frame_path in frame_paths],
        "prompt": prompt,
        "labels": labels
    }
    

anno_data=load_dataset("TIGER-Lab/VideoFeedback",name="annotated",split="train")
video_dir = Path("VideoFeedback/annotated/videos_train")
new_anno_data = anno_data.map(map_to_train_data, num_proc=num_proc, desc="Processing annotated data", remove_columns=anno_data.column_names, fn_kwargs={"video_dir": video_dir})

real_data=load_dataset("TIGER-Lab/VideoFeedback",name="real",split="train")   
video_dir = Path("VideoFeedback/real/videos_train")
new_real_data = real_data.map(map_to_train_data, num_proc=num_proc, desc="Processing real data", remove_columns=real_data.column_names, fn_kwargs={"video_dir": video_dir})

all_data = datasets.concatenate_datasets([new_anno_data, new_real_data])

with open("./train_regression.json", "w") as f:
    json.dump([item for item in all_data], f, indent=4)
    print(f"Saved {len(all_data)} items to train_regression.json")