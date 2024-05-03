import json
import fire
import os
import math
import cv2
import pandas as pd
from tqdm import tqdm

class NExTQA():
    def __init__(self, split='train'):
        self.split=split
        self.data = pd.read_csv(f'./data/nextqa/{split}.csv')
        self.vid_mapping=json.load(open(f'./data/nextqa/map_vid_vidorID.json', "r"))

    def _get_frame(self, idx, num_frames=8):
        vid=str(self.data["video"].values[idx])
        video_path=f"./data/nextqa/nextvideo/{self.vid_mapping[vid]}.mp4"

        images_folder=f"./data/nextqa/images"
        os.makedirs(images_folder, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = math.floor(total_frames / num_frames)

        for i in range(num_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
            ret, frame = cap.read()
            if ret:
                frame_filename = f"{images_folder}/{vid}_{i}.jpg"
                cv2.imwrite(frame_filename, frame)
            else:
                break
        cap.release()
        return None
    
    def _check_images(self,idx):
        vid=str(self.data["video"].values[idx])
        not_exist_frame=[]
        images_folder=f"data/nextqa/images"
        for i in range(8):
            frame_filename = f"{images_folder}/{vid}_{i}.jpg"
            if not os.path.exists(frame_filename):
                print(f"{images_folder}/{vid}_{i}.jpg not exist!\n")
                not_exist_frame.append(f"{vid}_{i}.jpg")
        return not_exist_frame

class STAR():
    def __init__(self, split="train"):
        self.split = split
        self.data = json.load(open(f'data/star/STAR_{split}.json', 'r'))
        self.da=1
    
    def _get_frame(self,idx):
        num_frames=8
        vid=self.data[idx]["video_id"]
        video_path=f"data/star/Charades_v1_480/{vid}.mp4"

        images_folder=f"data/star/images"
        os.makedirs(images_folder, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = math.floor(total_frames / num_frames)

        for i in range(num_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
            ret, frame = cap.read()
            if ret:
                frame_filename = f"{images_folder}/{vid}_{i}.jpg"
                cv2.imwrite(frame_filename, frame)
            else:
                break
        cap.release()
        return None
    
    def _check_images(self,idx):
        vid=self.data[idx]["video_id"]
        not_exist_frame=[]
        images_folder=f"data/star/images"
        for i in range(8):
            frame_filename = f"{images_folder}/{vid}_{i}.jpg"
            if not os.path.exists(frame_filename):
                print(f"{images_folder}/{vid}_{i}.jpg not exist!\n")
                not_exist_frame.append(f"{vid}_{i}.jpg")
        return not_exist_frame
    
def main(
        dset,
):
    if dset=="nextqa":
        clsname=NExTQA
    elif dset == "star":
        clsname=STAR
    else:
        print("Args Error")
        exit(0)

    curr_set=clsname(split="train")
    for i in tqdm(range(len(curr_set.data))):
        curr_set._get_frame(idx=i)
    curr_set=clsname(split="val")
    for i in tqdm(range(len(curr_set.data))):
        curr_set._get_frame(idx=i)
    
            
if __name__ == '__main__':
    fire.Fire(main)
