import torch
import datasets
import yaml
import numpy as np
import bisect
import PIL
import regex as re
import time
import os
import math
import random
import av
import decord
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datasets.config import HF_DATASETS_OFFLINE, HF_DATASETS_CACHE
from mantis.train.train_utils import (
    load_images,
    load_json_data,
)
from mantis.train.conversation import SeparatorStyle
from typing import List, Dict
IGNORE_INDEX = -100
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_TOKEN_ID = None # should be set when loading the processor
DEFAULT_VIDEO_TOKEN = "<video>"
DEFAULT_VIDEO_TOKEN_ID = None # should be set when loading the processor

DATASET_IMAGE_TOKENS = "<image>"
DATASET_VIDEO_TOKENS = "<video>"

def set_ignore_index(new_ignore_index=-100):
    global IGNORE_INDEX
    IGNORE_INDEX = new_ignore_index

def set_default_image_token(new_default_image_token="<image>"):
    global DEFAULT_IMAGE_TOKEN
    DEFAULT_IMAGE_TOKEN = new_default_image_token
    print("setting default image token to", new_default_image_token)

def set_default_image_token_id(new_default_image_token_id=None):
    global DEFAULT_IMAGE_TOKEN_ID
    DEFAULT_IMAGE_TOKEN_ID = new_default_image_token_id
    print("setting default image token id to", new_default_image_token_id)
    
def set_default_video_token(new_default_video_token="<video>"):
    global DEFAULT_VIDEO_TOKEN
    DEFAULT_VIDEO_TOKEN = new_default_video_token
    print("setting default video token to", new_default_video_token)
    
def set_default_video_token_id(new_default_video_token_id=None):
    global DEFAULT_VIDEO_TOKEN_ID
    DEFAULT_VIDEO_TOKEN_ID = new_default_video_token_id
    print("setting default video token id to", new_default_video_token_id)

def read_local_cached_dataset(data_path, name, split, offline_sha):
    assert offline_sha is not None, "offline_sha must be provided when HF_DATASETS_OFFLINE is True"
    repo, repo_dataset_name = data_path.split("/")
    repo_dataset_name = repo_dataset_name.lower()
    local_cache_path = HF_DATASETS_CACHE / f"{repo}___{repo_dataset_name}"
    datafile_path = local_cache_path / f"{name}/0.0.0/{offline_sha}/{repo_dataset_name}-{split}.arrow" # MIQA commit id
    image_dir = local_cache_path / f"{name}/0.0.0/{offline_sha}/{split}_images"
    assert local_cache_path.exists(), f"{local_cache_path} does not exist"
    assert image_dir.exists(), f"{image_dir} does not exist"
    if datafile_path.exists():
        dataset = datasets.Dataset.from_file(str(datafile_path))
    else:
        files = []
        pattern = f"{repo_dataset_name}-{split}-\d+-of-\d+.arrow"
        for file in datafile_path.parent.iterdir():
            if file.is_file() and re.match(pattern, file.name):
                files.append(file)
        files.sort(key=lambda x: int(x.name.split("-")[-3]))
        assert len(files) > 0, f"No files found for {datafile_path}"
        all_datasets = []
        for file in files:
            all_datasets.append(datasets.Dataset.from_file(str(file)))
        dataset = datasets.concatenate_datasets(all_datasets)
        print(f"Loading dataset '{name}' {split} from offline cached huggingface datasets")
             
    # map image path to absolute path
    def map_image_path_to_abs(item):
        if item['images']:
            for image in item['images']:
                image["path"] = str(image_dir / image["path"])
        return item
    dataset = dataset.map(map_image_path_to_abs)
    return dataset

class ChatDataset(torch.utils.data.Dataset):
    """
    conv format:
    <s> {system}\n USER: {}<0x04>ASSISTANT: {}</s> ...
    """
    def __init__(
        self, processor, data_path, dataset_type, name, split, max_seq_len, conv_format,
        is_master_worker=True, 
        max_size=None, 
        shuffle=False, 
        max_num_images=None, 
        vl_only=False,
        offline_sha=None,
        sample_ratio=1.0,
        revision="script",
        num_proc=8,
        max_image_size=None,
    ):
        self.num_proc = num_proc
        self.processor = processor
        self.data_path = Path(data_path)
        self.dataset_type = dataset_type
        self.name = name
        self.split = split
        self.is_master_worker = is_master_worker
        self.max_size = max_size
        self.max_num_images = max_num_images
        self.max_image_size = max_image_size
        # print("Sleeping for", int(os.environ.get("LOCAL_RANK", 0)) * 5, "seconds")
        # time.sleep(int(os.environ.get("LOCAL_RANK", 0)) * 5) # avoid error when multiple processes try to access the same file
        if self.data_path.exists() and self.dataset_type != "huggingface":
            self.print(f"Loading dataset '{name}' from {data_path}")
            self.data = load_json_data(data_path)
            self.image_dir = self.data_path.parent
            if shuffle:
                random.seed(42)
                random.shuffle(self.data)
            if self.max_size:
                print(f"Truncating dataset to from {len(self.data)} to {self.max_size}")
                self.data = self.data[:self.max_size]
            self.data = datasets.Dataset.from_list(self.data)
        else:
            # load from huggingface datasets
            if HF_DATASETS_OFFLINE:
                # when export HF_DATASETS_OFFLINE=1
                print(f"Loading dataset '{name}' {split} from offline cached huggingface datasets")
                self.data = read_local_cached_dataset(data_path, name, split, offline_sha)
            else:
                self.print(f"Loading dataset '{data_path}' {name} {split} from online huggingface datasets")
                max_retry = 5
                retried = 0
                while retried < max_retry:
                    try:
                        self.data = datasets.load_dataset(data_path, name, split=split, trust_remote_code=True, revision=revision, num_proc=num_proc)
                        break
                    except json.decoder.JSONDecodeError as e:
                        retried += 1
                        time.sleep(5)
                        if retried > max_retry:
                            raise e
                    
            _max_num_images = max([len(x) for x in self.data['images'] if x])
            print(f"Max number of images per sample: {_max_num_images}, limit: {max_num_images}")
            if max_num_images and _max_num_images > max_num_images:
                print(f"Filtering dataset to images <= {max_num_images}")
                self.filtered_data = self.data.filter(lambda x: len(x['images']) <= max_num_images if ('images' in x and x['images']) else True) # max 5 images
                print(f"Filtered dataset size changed from {len(self.data)} to {len(self.filtered_data)}")
                self.data = self.filtered_data
            if vl_only:
                print("Filtering dataset with images only")
                self.data = self.data.filter(lambda x: "images" in x and x['images']) # debug
                print("filter out images, now {}".format(len(self.data)))
            self.image_dir = Path("/")
            if shuffle:
                self.data = self.data.shuffle(seed=42)
            if self.max_size:
                print(f"Truncating dataset to from {len(self.data)} to {self.max_size}")
                self.data = self.data.select(range(self.max_size))
                    
        # filtering examples with image more than max_num_images
        if isinstance(max_num_images, int) and max_num_images > 0:
            self.data = self.data.filter(
                lambda x: len(x['images']) <= max_num_images if 'images' in x else True, 
                num_proc=num_proc, desc="Filtering examples with more than max_num_images")

        # for debugging    
        # new_data = []
        # for i, x in enumerate(self.data):
        #     new_data.append(x)
        #     if i > 100:
        #         break
        # self.data = new_data
        
        # self.conv = default_conversation.copy()
        self.conv = conv_format.copy()
        self.conversations, self.all_images = self.preprocess()

        if sample_ratio < 1.0:
            print(f"Down sampling {sample_ratio} of the data")
            num_samples = int(len(self.conversations) * sample_ratio)
            self.conversations = self.conversations[:num_samples]
            self.all_images = self.all_images[:num_samples]
        elif sample_ratio > 1.0:
            additional_samples = int(len(self.conversations) * (sample_ratio - 1))
            print(f"Adding {additional_samples} samples for dataset {name}")
            added_conversations, added_images = [], []
            while additional_samples > len(self.conversations):
                added_conversations.extend(self.conversations)
                added_images.extend(self.all_images)
                additional_samples -= len(self.conversations)
            random.seed(42)
            added_conversations.extend(random.sample(self.conversations, additional_samples))
            added_images.extend(random.sample(self.all_images, additional_samples))
            self.conversations += added_conversations
            self.all_images += added_images

        self.max_seq_len = max_seq_len
    
    def print(self, *args, **kwargs):
        if self.is_master_worker:
            print(*args, **kwargs)

    def preprocess(self):
        
        # process formats
        conv = self.conv
        image_dir = self.image_dir
        roles = {"human": conv.roles[0], "gpt": conv.roles[1], "user": conv.roles[0], "assistant": conv.roles[1]}
        conversations = []
        all_images = []
        
        def preprocess_func(item, i):
            # phd
            source_key = "conversation" if "conversation" in item else "conversations"
            source = item[source_key]
            if roles[source[0].get("from", source[0].get("role"))] != conv.roles[0]:
                # Skip the first one if it is not from human
                source = source[1:]

            conv.messages = []
            for j, sentence in enumerate(source):
                role = roles[sentence.get("from", sentence.get("role"))]
                assert role == conv.roles[j % 2], f"{i}"
                conv.append_message(role, sentence.get("content", sentence.get("text", sentence.get("value", ""))))
            # prompt = conv.get_prompt()
            conv_messages = conv.messages.copy()
            if "image" in item and item['image']:
                if isinstance(item['image'], str):
                    image_file = image_dir / item['image']
                elif isinstance(item['image'], PIL.Image.Image):
                    image_file = item['image']
                elif isinstance(item['image'], dict):
                    image_file = image_dir / item['image']['path']
                else:
                    raise ValueError(f"Unknown image format {item['image']}")
                image_file = [image_file]
            elif "images" in item and item['images'] and len(item['images']) > 0:
                if isinstance(item['images'][0], str):
                    image_file = [image_dir / image for image in item['images']]
                elif isinstance(item['images'][0], dict):
                    image_file = [image_dir / image['path'] for image in item['images']]
                elif isinstance(item['images'][0], PIL.Image.Image):
                    image_file = item['images']
            else:
                image_file = None
            try:
                if image_file:
                    if isinstance(image_file, list) and all([isinstance(image, Path) for image in image_file]):
                        assert all([image.exists() for image in image_file]), f"{image_file} does not exist"
                        image_file = [str(image) for image in image_file]
                    elif isinstance(image_file, Path):
                        assert image_file.exists(), f"{image_file} does not exist"
                        image_file = str(image_file)
                else:
                    image_file = None
                return {"conv_messages": conv_messages, "image_file": image_file}
            except Exception as e:
                print(f"Error at {i}")
                return {"conv_messages": None, "image_file": None}
        
        def filter_none(item):
            return item["conv_messages"] is not None
        
        new_dataset = self.data.map(preprocess_func, with_indices=True, desc="Format conversations and load images", remove_columns=self.data.column_names, num_proc=self.num_proc)
        new_dataset = new_dataset.filter(filter_none, num_proc=self.num_proc)
        conversations = new_dataset["conv_messages"]
        all_images = new_dataset["image_file"]
        
        return conversations, all_images
        
    def __len__(self):
        return len(self.conversations)
    
    def __getitem__(self, idx):
        conv_messages = self.conversations[idx]
        sub_images = self.all_images[idx]
        sub_images = load_images(sub_images, max_image_size=self.max_image_size)
        # resize sub_images to be at least 16 * 16 if image is too small, to avoid errors in clip image processor
        if sub_images:
            for i, image in enumerate(sub_images):
                if image.size[0] < 16 or image.size[1] < 16:
                    scale_factor = max(16 / image.size[0], 16 / image.size[1])
                    sub_images[i] = image.resize((int(image.size[0] * scale_factor), int(image.size[1] * scale_factor))).convert("RGB")
                    
        if self.conv.sep_style == SeparatorStyle.PLAIN:
            # NOTE: this is for the pretraining, where we only use the pure text or interleaved text and images
            source = conv_messages
            assert len(source) >= 2, "we only use the text in the second message for pretraining."
            # assert DEFAULT_IMAGE_TOKEN in source[0][1]
            # assert len(sub_images) == 1 if isinstance(sub_images, list) else isinstance(sub_images, PIL.Image.Image)
            if isinstance(sub_images, PIL.Image.Image):
                sub_images = [sub_images]
            text = source[1][1]
            image_token_count = source[1][1].count(DEFAULT_IMAGE_TOKEN)
            if image_token_count < len(sub_images):
                text = f"{DEFAULT_IMAGE_TOKEN} " * (len(sub_images) - image_token_count) + text
            conv_str = text + self.conv.sep
            encoding = self.processor(text=conv_str, images=sub_images, return_tensors="pt", truncation=True, max_length=self.max_seq_len)
        else:
            # NOTE: this is for the conversation style finetuning
            # check the number of images
            image_token_count = sum([message[1].count(DEFAULT_IMAGE_TOKEN) for message in conv_messages])
            if isinstance(sub_images, list):
                if image_token_count < len(sub_images):
                    conv_messages[0][1] = DEFAULT_IMAGE_TOKEN * (len(sub_images) - image_token_count) + conv_messages[0][1]
            self.conv.messages = conv_messages
            conv_str = self.conv.get_prompt()
            
            if self.conv.sep_style == SeparatorStyle.QWEN2VL:
                from qwen_vl_utils.vision_process import fetch_image
                sub_images = [fetch_image({"type": "image", "image": image}) for image in sub_images]
                encoding = self.processor(text=conv_str, images=sub_images, videos=None, return_tensors="pt", truncation=True, max_length=self.max_seq_len)
            else:
                encoding = self.processor(conv_str, sub_images, return_tensors="pt", truncation=True, max_length=self.max_seq_len)

        if "image_patches" in encoding:
            encoding.pop("attention_mask")
            encoding['image_patches'] = encoding['image_patches'][0] # todo
        encoding["labels"] = torch.full_like(encoding["input_ids"], IGNORE_INDEX, dtype=encoding["input_ids"].dtype)
        input_ids = encoding["input_ids"][0]
        target = encoding["labels"][0]
        if self.conv.sep_style == SeparatorStyle.MFUYU:
            sep_id = self.processor.tokenizer.convert_tokens_to_ids(self.conv.sep)
            sep2_id = self.processor.tokenizer.convert_tokens_to_ids(self.conv.sep2)
            
            sep_idxs = torch.nonzero((input_ids == sep_id), as_tuple=True)[0].tolist() 
            sep2_idxs = torch.nonzero((input_ids == sep2_id), as_tuple=True)[0].tolist() 
            if not (len(sep_idxs) == len(sep2_idxs) or len(sep_idxs) == len(sep2_idxs) + 1):
                torch.set_printoptions(profile="full")
                raise ValueError(f"len({sep_idxs}) != len({sep2_idxs})")
            assert len(sep_idxs) == len(sep2_idxs) or len(sep_idxs) == len(sep2_idxs) + 1, f"len({sep_idxs}) != len({sep2_idxs})"
            if len(sep_idxs) == len(sep2_idxs) + 1:
                sep2_idxs.append(len(input_ids) - 1)
            for j in range(len(sep_idxs)):
                target[sep_idxs[j]+1:sep2_idxs[j] + 1] = input_ids[sep_idxs[j]+1:sep2_idxs[j] + 1]
        elif self.conv.sep_style == SeparatorStyle.SINGLE or \
            self.conv.sep_style == SeparatorStyle.LLAMA_3:
            sep_id = self.processor.tokenizer.convert_tokens_to_ids(self.conv.sep)
            sep_idxs = torch.nonzero((input_ids == sep_id), as_tuple=True)[0].tolist()
            for i in range(len(sep_idxs)):
                if i % 2 == 0:
                    continue
                if i == len(sep_idxs) - 1:
                    target[sep_idxs[i]+1:] = input_ids[sep_idxs[i]+1:]
                else:
                    target[sep_idxs[i]+1:sep_idxs[i+1] + 1] = input_ids[sep_idxs[i]+1:sep_idxs[i+1] + 1]
        elif self.conv.sep_style in [SeparatorStyle.IDEFICS_2, SeparatorStyle.IDEFICS_3, SeparatorStyle.QWEN2VL]:
            if self.conv.system:
                skip_offset = 0
            else:
                skip_offset = 1
            sep_id = self.processor.tokenizer.convert_tokens_to_ids(self.conv.sep)
            sep_idxs = torch.nonzero((input_ids == sep_id), as_tuple=True)[0].tolist()
            for i in range(len(sep_idxs)):
                if i % 2 == skip_offset:
                    continue
                if i == len(sep_idxs) - 1:
                    target[sep_idxs[i]+1:] = input_ids[sep_idxs[i]+1:]
                else:
                    target[sep_idxs[i]+1:sep_idxs[i+1] + 1] = input_ids[sep_idxs[i]+1:sep_idxs[i+1] + 1]
        elif self.conv.sep_style == SeparatorStyle.PLAIN:
            assert DEFAULT_IMAGE_TOKEN_ID is not None, "Please set the default image token id by calling set_default_image_token_id, this is required to masking the image tokens for pretraining."
            # mask the image tokens in the text
            target[input_ids != DEFAULT_IMAGE_TOKEN_ID] = input_ids[input_ids != DEFAULT_IMAGE_TOKEN_ID]
            # source = conv_str
            # tokenized_len = len(self.processor(source[0][1], sub_images, return_tensors="pt")["input_ids"][0])
            # target[tokenized_len:] = input_ids[tokenized_len:]
        else:
            raise ValueError(f"Unknown separator style {self.conv.sep_style}")
        # replace IGNORE_INDEX in target_ids with 0 and decode it, then print for debug
        if torch.all(target == IGNORE_INDEX):
            print("no labels for a sample in ", self.data_path, self.name, self.split, idx)
        
        # print(self.data_path, self.name, len(sub_images), input_ids.shape, [x.size for x in sub_images], conv_messages[0])
        # print(self.data_path, self.name, self.split)
        
        # for debug, print the targets to make sure the right tokens are learned
        # need to print to make sure that the masked tokens are correct.
        # _target = target.clone().detach()
        # _target[_target == IGNORE_INDEX] = 0
        # print(self.processor.tokenizer.decode(input_ids, skip_special_tokens=False))
        # print(self.processor.tokenizer.decode(_target, skip_special_tokens=False))
        

        return encoding

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

class ChatVideoDataset(torch.utils.data.Dataset):
    """
    conv format:
    <s> {system}\n USER: {}<0x04>ASSISTANT: {}</s> ...
    """
    def __init__(
        self, processor, data_path, dataset_type, name, 
        video_dir, split, max_seq_len, conv_format,
        is_master_worker=True, 
        max_size=None, 
        shuffle=False, 
        max_num_frames=None, 
        sample_ratio=1.0,
        fps=1,
        use_video_encoder=False,
    ):
        self.processor = processor
        self.data_path = Path(data_path)
        self.dataset_type = dataset_type
        self.name = name
        self.split = split
        self.is_master_worker = is_master_worker
        self.max_size = max_size
        self.max_num_frames = max_num_frames
        self.print(f"Loading dataset '{name}' from {data_path}")
        self.data = load_json_data(data_path)
        if not video_dir:
            self.video_dir = self.data_path.parent
        else:
            self.video_dir = Path(video_dir)
        assert self.video_dir.exists(), f"{video_dir} does not exist"
        if shuffle:
            random.seed(42)
            random.shuffle(self.data)
        if self.max_size:
            print(f"Truncating dataset to from {len(self.data)} to {self.max_size}")
            self.data = self.data[:self.max_size]
        
        self.conv = conv_format.copy()
        self.conversations, self.all_selected_idxs = self.preprocess()

        self.max_seq_len = max_seq_len
        self.use_video_encoder = use_video_encoder
        self.fps = fps
    
    def print(self, *args, **kwargs):
        if self.is_master_worker:
            print(*args, **kwargs)

    def preprocess(self):
        
        # process formats
        conv = self.conv
        video_dir = self.video_dir
        roles = {"human": conv.roles[0], "gpt": conv.roles[1], "user": conv.roles[0], "assistant": conv.roles[1]}
        conversations = []
        all_selected_idxs = []
        for i, item in tqdm(
            enumerate(self.data), desc="Format conversations and load images", 
            total=len(self.data), disable=not self.is_master_worker
        ):
            # phd
            source_key = "conversation" if "conversation" in item else "conversations"
            source = item[source_key]
            if roles[source[0].get("from", source[0].get("role"))] != conv.roles[0]:
                # Skip the first one if it is not from human
                source = source[1:]

            conv.messages = []
            has_video_token = False
            for j, sentence in enumerate(source):
                role = roles[sentence.get("from", sentence.get("role"))]
                assert role == conv.roles[j % 2], f"{i}"
                content = sentence.get("content", sentence.get("text", sentence.get("value", "")))
                content = content.replace(DEFAULT_IMAGE_TOKEN, "").strip('\n ')
                if DEFAULT_VIDEO_TOKEN in content:
                    has_video_token = True
                conv.append_message(role, content)
            if not has_video_token:
                if random.random() < 0.5:
                    conv.messages[0][1] = DEFAULT_VIDEO_TOKEN + " " + conv.messages[0][1]
                else:
                    conv.messages[0][1] = conv.messages[0][1] + " " + DEFAULT_VIDEO_TOKEN

            conv_messages = conv.messages.copy()
            
            try:
                if "video" in item:
                    video_file = video_dir / item['video']
                    assert video_file.exists(), f"{video_file} does not exist"
                elif "images" in item and item['images'] and len(item['images']) > 0:
                    if isinstance(item['images'][0], str):
                        video_frames = [video_dir / image for image in item['images']]
                        assert all([image.exists() for image in video_frames]), f"{video_frames} does not exist"
                    elif isinstance(item['images'][0], dict):
                        video_frames = [video_dir / image['path'] for image in item['images']]
                        assert all([image.exists() for image in video_frames]), f"{video_frames} does not exist"
                    elif isinstance(item['images'][0], PIL.Image.Image):
                        video_frames = item['images']
                conversations.append(conv_messages)
                all_selected_idxs.append(i)
            except Exception as e:
                print(f"Error at {i}")
                print(video_file)
                print(e)
        
        return conversations, all_selected_idxs
        
    def __len__(self):
        return len(self.conversations)
    
    def __getitem__(self, idx):
        conv_messages = self.conversations[idx]
        selected_idx = self.all_selected_idxs[idx]
        item = self.data[selected_idx]
        video_dir = self.video_dir
        
        if "video" in item:
            video_file = video_dir / item['video']
            container = av.open(video_file)

            # sample uniformly 8 frames from the video
            total_frames = container.streams.video[0].frames
            # print(f"Total frames: {total_frames}")
            # print(f"FPS: {container.streams.video[0].average_rate}")
            # print(f"Duration: {container.streams.video[0].duration}")
            
            if self.max_num_frames and total_frames > self.max_num_frames:
                if self.fps:
                    interval = math.ceil(container.streams.video[0].average_rate / self.fps)
                    indices = np.arange(0, total_frames, interval).astype(int)
                    if len(indices) > self.max_num_frames:
                        indices = indices[:self.max_num_frames]
                else:
                    indices = np.arange(0, total_frames, total_frames / self.max_num_frames).astype(int)
                # print(f"Sample {len(indices)} frames from {total_frames} frames")
            else:
                indices = np.arange(total_frames)
            if self.use_video_encoder:
                video_frames = read_video_pyav(container, indices)
            else:
                # use frames as images instead
                video_frames = [frame.to_image() for frame in container.decode(video=0)]
                video_frames = [video_frames[i] for i in indices]
        elif "images" in item and item['images'] and len(item['images']) > 0:
            if isinstance(item['images'][0], str):
                video_frames = [video_dir / image for image in item['images']]
                video_frames = load_images(video_frames)
            elif isinstance(item['images'][0], dict):
                video_frames = [video_dir / image['path'] for image in item['images']]
                video_frames = load_images(video_frames)
            elif isinstance(item['images'][0], PIL.Image.Image):
                video_frames = item['images']
                
            if self.max_num_frames and len(video_frames) > self.max_num_frames:
                indices = np.arange(0, len(video_frames), len(video_frames) / self.max_num_frames).astype(int)
                video_frames = [video_frames[i] for i in indices]
            # change video frames from PIL.Image to ndarray
            if self.use_video_encoder:
                video_frames = np.stack([np.array(x.convert('RGB')) for x in video_frames])
            else:
                video_frames = [x.convert('RGB') for x in video_frames]
        else:
            video_frames = None
            
        # check the number of images
        self.conv.messages = conv_messages
        
        if self.conv.sep_style == SeparatorStyle.PLAIN:
            assert video_frames is not None, "video frames should not be None"
            # NOTE: this is for the pretraining, where we only use the pure text or interleaved text and images
            source = conv_messages
            assert len(source) >= 2, "we only use the text in the second message for pretraining."
            text = source[1][1]
            if self.use_video_encoder:
                video_token_count = text.count(DEFAULT_VIDEO_TOKEN)
                if video_token_count < 1:
                    text = f"{DEFAULT_VIDEO_TOKEN} " + text
                conv_str = text + self.conv.sep
                encoding = self.processor(text=conv_str, videos=video_frames, return_tensors="pt", truncation=True, max_length=self.max_seq_len)
            else:
                image_token_count = source[1][1].count(DEFAULT_IMAGE_TOKEN)
                if image_token_count < len(video_frames):
                    text = f"{DEFAULT_IMAGE_TOKEN} " * (len(video_frames) - image_token_count) + text
                conv_str = text + self.conv.sep
                if image_token_count > len(video_frames):
                    # replace image token from back to front for the extra image tokens
                    conv_str = conv_str[::-1].replace(DEFAULT_IMAGE_TOKEN[::-1], "", len(video_frames) - image_token_count)[::-1]
                encoding = self.processor(text=conv_str, images=video_frames, return_tensors="pt", truncation=True, max_length=self.max_seq_len)
        else:
            if self.use_video_encoder:
                self.conv.messages = conv_messages
                conv_str = self.conv.get_prompt()
                encoding = self.processor(text=conv_str, videos=video_frames, return_tensors="pt", truncation=True, max_length=self.max_seq_len)
            else:
                # add <image> tokens according to the number of images
                image_token_count = sum([message[1].count(DEFAULT_IMAGE_TOKEN) for message in conv_messages])
                if image_token_count < len(video_frames):
                    conv_messages[0][1] = DEFAULT_IMAGE_TOKEN * (len(video_frames) - image_token_count) + conv_messages[0][1]
                self.conv.messages = conv_messages
                conv_str = self.conv.get_prompt()
                if image_token_count > len(video_frames):
                    # replace image token from back to front for the extra image tokens
                    conv_str = conv_str[::-1].replace(DEFAULT_IMAGE_TOKEN[::-1], "", len(video_frames) - image_token_count)[::-1]
                encoding = self.processor(conv_str, images=video_frames, return_tensors="pt", truncation=True, max_length=self.max_seq_len)
        
        encoding["labels"] = torch.full_like(encoding["input_ids"], IGNORE_INDEX, dtype=encoding["input_ids"].dtype)
        target = encoding["labels"][0]
        input_ids = encoding["input_ids"][0]
        if self.conv.sep_style == SeparatorStyle.PLAIN:
            if self.use_video_encoder:
                target[input_ids != DEFAULT_VIDEO_TOKEN_ID] = input_ids[input_ids != DEFAULT_VIDEO_TOKEN_ID]
            else:
                target[input_ids != DEFAULT_IMAGE_TOKEN_ID] = input_ids[input_ids != DEFAULT_IMAGE_TOKEN_ID]
        else:
            new_conv = self.conv.copy()
            new_conv.messages = []
            for i in range(0, len(conv_messages), 2):
                new_conv.append_message(conv_messages[i][0], conv_messages[i][1])
                prompt = new_conv.get_prompt()
                user_len = len(self.processor(prompt, return_tensors="pt", truncation=True, max_length=self.max_seq_len)["input_ids"][0])

                new_conv.append_message(conv_messages[i+1][0], conv_messages[i+1][1])
                prompt = new_conv.get_prompt()
                user_and_assistant_len = len(self.processor(prompt, return_tensors="pt", truncation=True, max_length=self.max_seq_len)["input_ids"][0])
                target[user_len:user_and_assistant_len] = input_ids[user_len:user_and_assistant_len]
            
        if torch.all(target == IGNORE_INDEX):
            print("no labels for a sample in ", self.data_path, self.name, self.split, selected_idx)
        
        # print(self.data_path, self.name, self.split)
        
        # for debug, print the targets to make sure the right tokens are learned
        # need to print to make sure that the masked tokens are correct.
        # _target = target.clone().detach()
        # _target[_target == IGNORE_INDEX] = 0
        # print(self.processor.tokenizer.decode(input_ids, skip_special_tokens=False))
        # print(self.processor.tokenizer.decode(_target, skip_special_tokens=False))
        # print(input_ids.shape)
        

        return encoding

import threading
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

class SiglipVideoDataset(torch.utils.data.Dataset):
    """
    conv format:
    <s> {system}\n USER: {}<0x04>ASSISTANT: {}</s> ...
    """
    def __init__(
        self, processor, data_path, dataset_type, name, 
        video_dir, split, max_seq_len,
        is_master_worker=True, 
        max_size=None, 
        shuffle=False, 
        max_num_frames=None, 
        fps=None,
        video_reader_engine="pyav"
    ):
        self.processor = processor
        self.data_path = Path(data_path)
        self.dataset_type = dataset_type
        self.name = name
        self.split = split
        self.is_master_worker = is_master_worker
        self.max_size = max_size
        self.max_num_frames = max_num_frames
        self.print(f"Loading dataset '{name}' from {data_path}")
        self.data = load_json_data(data_path)
        if not video_dir:
            self.video_dir = self.data_path.parent
        else:
            self.video_dir = Path(video_dir)
        assert self.video_dir.exists(), f"{video_dir} does not exist"
        if shuffle:
            random.seed(42)
            random.shuffle(self.data)
        if self.max_size:
            print(f"Truncating dataset to from {len(self.data)} to {self.max_size}")
            self.data = self.data[:self.max_size]
        self.video_reader_engine = video_reader_engine

        self.max_seq_len = max_seq_len
        self.fps = fps
        self.num_proc = 8
        
        self.texts, self.all_selected_idxs = self.preprocess()
        
    
    def print(self, *args, **kwargs):
        if self.is_master_worker:
            print(*args, **kwargs)

    def preprocess(self):
        
        # process formats
        video_dir = self.video_dir
        texts = []
        all_selected_idxs = []
        all_video_frames = []
        
        
        # def _preprocess(item, i):
        #     source_key = "text"
        #     source = item[source_key]
            
        #     try:
        #         if "video" in item:
        #             video_file = video_dir / item['video']
        #             assert video_file.exists(), f"{video_file} does not exist"
        #             # container = av.open(video_file)
        #             # if self.max_num_frames:
        #             #     if self.fps:
        #             #         interval = math.ceil(container.streams.video[0].average_rate / self.fps)
        #             #         indices = np.arange(0, container.streams.video[0].frames, interval).astype(int)
        #             #         if len(indices) > self.max_num_frames:
        #             #             indices = indices[:self.max_num_frames]
        #             #     else:
        #             #         indices = np.arange(0, container.streams.video[0].frames, container.streams.video[0].frames / self.max_num_frames).astype(int)
        #             #     # print(f"Sample {len(indices)} frames from {container.streams.video[0].frames} frames")
        #             # else:
        #             #     indices = np.arange(container.streams.video[0].frames)
        #             # video_frames = read_video_pyav(container, indices)
        #             # all_video_frames.append(video_frames)
        #             # print(f"Video frames: {len(video_frames)}")
        #         elif "images" in item and item['images'] and len(item['images']) > 0:
        #             if isinstance(item['images'][0], str):
        #                 video_frames = [video_dir / image for image in item['images']]
        #                 assert all([image.exists() for image in video_frames]), f"{video_frames} does not exist"
        #             elif isinstance(item['images'][0], dict):
        #                 video_frames = [video_dir / image['path'] for image in item['images']]
        #                 assert all([image.exists() for image in video_frames]), f"{video_frames} does not exist"
        #             elif isinstance(item['images'][0], PIL.Image.Image):
        #                 video_frames = item['images']
        #         texts.append(source)
        #         all_selected_idxs.append(i)
        #         item['preprocessed'] = {
        #             "text": source,
        #             # "video_frames": video_frames,
        #             "selected_idx": i
        #         }
        #     except Exception as e:
        #         print(f"Error at {i}")
        #         print(e)
        #         item['preprocessed'] = None
        #     return item
        
        # def _filter_none(item):
        #     return "preprocessed" in item and item["preprocessed"] is not None
        
        # new_dataset = datasets.Dataset.from_list(self.data)
        # new_dataset = new_dataset.map(_preprocess, with_indices=True, desc="Format conversations and load images", num_proc=1)
        # new_dataset = new_dataset.filter(_filter_none, num_proc=self.num_proc)
        # texts = new_dataset["preprocessed"]["text"]
        # all_selected_idxs = new_dataset["preprocessed"]["selected_idx"]
        
        for i, item in tqdm(
            enumerate(self.data), desc="Format conversations and load images", 
            total=len(self.data), disable=not self.is_master_worker
        ):
            source_key = "text"
            source = item[source_key]
            
            try:
                if "video" in item:
                    video_file = video_dir / item['video']
                    assert video_file.exists(), f"{video_file} does not exist"
                elif "images" in item and item['images'] and len(item['images']) > 0:
                    if isinstance(item['images'][0], str):
                        video_frames = [video_dir / image for image in item['images']]
                        assert all([image.exists() for image in video_frames]), f"{video_frames} does not exist"
                    elif isinstance(item['images'][0], dict):
                        video_frames = [video_dir / image['path'] for image in item['images']]
                        assert all([image.exists() for image in video_frames]), f"{video_frames} does not exist"
                    elif isinstance(item['images'][0], PIL.Image.Image):
                        video_frames = item['images']
                texts.append(source)
                all_selected_idxs.append(i)
            except Exception as e:
                print(f"Error at {i}")
                print(e)
        
        return texts, all_selected_idxs
        
    def __len__(self):
        return len(self.texts)
    
    # @with_timeout(30)
    def __mygetitem__(self, idx):
        text = self.texts[idx]
        selected_idx = self.all_selected_idxs[idx]
        item = self.data[selected_idx]
        video_dir = self.video_dir
        
        if "video" in item:
            video_file = video_dir / item['video']
            
            # start = time.time()
            if self.video_reader_engine == "decord":
                video_reader = decord.VideoReader(str(video_file))
                total_frames = len(video_reader)
                video_fps = video_reader.get_avg_fps()
            elif self.video_reader_engine == "pyav":
                container = av.open(video_file)
                total_frames = container.streams.video[0].frames
                video_fps = container.streams.video[0].average_rate
            
            # # sample uniformly 8 frames from the video
            # print(f"Total frames: {total_frames}")
            # print(f"FPS: {video_fps}")
            # # print(f"Duration: {container.streams.video[0].duration}")
            
            # print(f"self.max_num_frames: {self.max_num_frames}")
            # print(f"self.fps: {self.fps}")
            if self.max_num_frames and total_frames > self.max_num_frames:
                if self.fps:
                    interval = math.ceil(video_fps / self.fps)
                    indices = np.arange(0, total_frames, interval).astype(int)
                    if len(indices) > self.max_num_frames:
                        indices = indices[:self.max_num_frames]
                else:
                    indices = np.arange(0, total_frames, total_frames / self.max_num_frames).astype(int)
                # print(f"Sample {len(indices)} frames from {total_frames} frames")
            else:
                indices = np.arange(total_frames)
            # print(f"Decoding video {video_file} with indices {indices}")
            if self.video_reader_engine == "decord":
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
            elif self.video_reader_engine == "pyav":
                video_frames = read_video_pyav(container, indices)
            # end = time.time()
            # print(f"Decoding video {video_file} takes {end - start:.2f} seconds ({len(indices)} frames)")
        elif "images" in item and item['images'] and len(item['images']) > 0:
            # start = time.time()
            if isinstance(item['images'][0], str):
                video_frames = [video_dir / image for image in item['images']]
                video_frames = load_images(video_frames)
            elif isinstance(item['images'][0], dict):
                video_frames = [video_dir / image['path'] for image in item['images']]
                video_frames = load_images(video_frames)
            elif isinstance(item['images'][0], PIL.Image.Image):
                video_frames = item['images']
                
            #     video_frames = [video_frames[i] for i in indices]
            # change video frames from PIL.Image to ndarray
            video_frames = np.stack([np.array(x.convert('RGB')) for x in video_frames])
            # end = time.time()
            # print(f"Loaded {len(video_frames)} frames from images")
        else:
            video_frames = None
        
        encoding = self.processor(text=text, images=video_frames, return_tensors="pt", padding="max_length")

        return encoding
    
    def __getitem__(self, idx):
        try:
            return self.__mygetitem__(idx)
        except Exception as e:
            return self.__getitem__((idx + 1) % len(self))
    
class ClassificationDataset(torch.utils.data.Dataset):
    """
    conv format:
    <s> {system}\n USER: {}<0x04>ASSISTANT: {}</s> ...
    """
    def __init__(
        self, processor, data_path, dataset_type, name, split, max_seq_len,
        is_master_worker=True, 
        max_size=None, 
        shuffle=False, 
        max_num_images=None, 
        vl_only=False,
        offline_sha=None,
        revision="script"
    ):
        self.processor = processor
        self.data_path = Path(data_path)
        self.dataset_type = dataset_type
        self.name = name
        self.split = split
        self.is_master_worker = is_master_worker
        self.max_size = max_size
        self.max_num_images = max_num_images
        self.max_seq_len = max_seq_len
        print("Sleeping for", int(os.environ.get("LOCAL_RANK", 0)) * 5, "seconds")
        time.sleep(int(os.environ.get("LOCAL_RANK", 0)) * 5) # avoid error when multiple processes try to access the same file
        if self.data_path.exists() and self.dataset_type != "huggingface":
            self.print(f"Loading dataset '{name}' from {data_path}")
            self.data = load_json_data(data_path)
            self.image_dir = self.data_path.parent
            if shuffle:
                random.seed(42)
                random.shuffle(self.data)
            if self.max_size:
                print(f"Truncating dataset to from {len(self.data)} to {self.max_size}")
                self.data = self.data[:self.max_size]
        else:
            # load from huggingface datasets
            if HF_DATASETS_OFFLINE:
                # when export HF_DATASETS_OFFLINE=1
                print(f"Loading dataset '{name}' {split} from offline cached huggingface datasets")
                self.data = read_local_cached_dataset(data_path, name, split, offline_sha)
            else:
                self.print(f"Loading dataset '{data_path}' {name} {split} from online huggingface datasets")
                self.data = datasets.load_dataset(data_path, name, split=split, trust_remote_code=True, revision=revision)
            _max_num_images = max([len(x) for x in self.data['images'] if x])
            print(f"Max number of images per sample: {_max_num_images}, limit: {max_num_images}")
            if max_num_images and _max_num_images > max_num_images:
                print(f"Filtering dataset to images <= {max_num_images}")
                self.filtered_data = self.data.filter(lambda x: len(x['images']) <= max_num_images if ('images' in x and x['images']) else True) # max 5 images
                print(f"Filtered dataset size changed from {len(self.data)} to {len(self.filtered_data)}")
                self.data = self.filtered_data
            if vl_only:
                print("Filtering dataset with images only")
                self.data = self.data.filter(lambda x: "images" in x and x['images']) # debug
                print("filter out images, now {}".format(len(self.data)))
            self.image_dir = Path("/")
            if shuffle:
                self.data = self.data.shuffle(seed=42)
            if self.max_size:
                print(f"Truncating dataset to from {len(self.data)} to {self.max_size}")
                self.data = self.data.select(range(self.max_size))

        # for debugging    
        # new_data = []
        # for i, x in enumerate(self.data):
        #     new_data.append(x)
        #     if i > 100:
        #         break
        # self.data = new_data
        
        self.prompts, self.all_images, self.labels = self.preprocess()
        
    def print(self, *args, **kwargs):
        if self.is_master_worker:
            print(*args, **kwargs)

    def preprocess(self):
        
        # process formats
        image_dir = self.image_dir
        prompts = []
        all_images = []
        label_names = list(self.data[0]['labels'].keys())
        self.label_names = label_names
        all_labels = []
        for i, item in tqdm(
            enumerate(self.data), desc="Format prompts and load images", 
            total=len(self.data), disable=not self.is_master_worker
        ):
            # phd
            
            if "image" in item and item['image']:
                if isinstance(item['image'], str):
                    image_file = image_dir / item['image']
                elif isinstance(item['image'], PIL.Image.Image):
                    image_file = item['image']
                elif isinstance(item['image'], dict):
                    image_file = image_dir / item['image']['path']
                else:
                    raise ValueError(f"Unknown image format {item['image']}")
            elif "images" in item and item['images'] and len(item['images']) > 0:
                if isinstance(item['images'][0], str):
                    image_file = [image_dir / image for image in item['images']]
                elif isinstance(item['images'][0], dict):
                    image_file = [image_dir / image['path'] for image in item['images']]
                elif isinstance(item['images'][0], PIL.Image.Image):
                    image_file = item['images']
            else:
                image_file = None
            try:
                if image_file:
                    if isinstance(image_file, list) and all([isinstance(image, Path) for image in image_file]):
                        assert all([image.exists() for image in image_file]), f"{image_file} does not exist"
                    elif isinstance(image_file, Path):
                        assert image_file.exists(), f"{image_file} does not exist"
                else:
                    image_file = None
                prompts.append(item['prompt'])
                all_images.append(image_file)
                labels = [float(item['labels'][label_name]) for label_name in label_names]
                all_labels.append(labels)
                
            except Exception as e:
                print(f"Error at {i}")
                print(e)
        
        return prompts, all_images, all_labels
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        sub_images = self.all_images[idx]
        sub_images = load_images(sub_images)
        labels = self.labels[idx]
        # resize sub_images to be at least 16 * 16 if image is too small, to avoid errors in clip image processor
        if sub_images:
            for i, image in enumerate(sub_images):
                if image.size[0] < 16 or image.size[1] < 16:
                    scale_factor = max(16 / image.size[0], 16 / image.size[1])
                    sub_images[i] = image.resize((int(image.size[0] * scale_factor), int(image.size[1] * scale_factor))).convert("RGB")
        
        image_token_count = prompt.count(DEFAULT_IMAGE_TOKEN)
        if image_token_count < len(sub_images):
            prompt += f"{DEFAULT_IMAGE_TOKEN} " * (len(sub_images) - image_token_count)
        encoding = self.processor(prompt, sub_images, return_tensors="pt", truncation=True, max_length=self.max_seq_len)

        if "image_patches" in encoding:
            encoding.pop("attention_mask")
            encoding['image_patches'] = encoding['image_patches'][0] # todo

        encoding["labels"] = torch.tensor(labels, dtype=torch.float32).unsqueeze(0)

        return encoding
    
    
class Qwen2VideoClassificationDataset(ClassificationDataset):
    def __init__(
        self, *args, fps=1, score_type=None, **kwargs, 
    ):
        super().__init__(*args, **kwargs)
        self.fps = fps
        if hasattr(self.processor.tokenizer, "score_type"):
            self.score_type = self.processor.tokenizer.score_type
            if self.score_type == "special_token":
                self.label_special_tokens = self.processor.tokenizer.label_special_tokens
                self.processor.tokenizer.label_names = self.label_names
        
    
    def __getitem__(self, idx):
        from qwen_vl_utils import process_vision_info
        prompt = self.prompts[idx]
        sub_images = self.all_images[idx]
        sub_images = load_images(sub_images)
        labels = self.labels[idx]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "video",
                        "video": sub_images,
                        "fps": self.fps,
                    },
                ],
            }
        ]
        if self.score_type == "special_token":
            response = ""
            for i in range(len(labels)):
                response += f"The score for {self.label_names[i]} is {self.label_special_tokens[i]}. "
            messages.append({"role": "assistant", "content": [{"type": "text", "text": response}]})
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        image_inputs, video_inputs = process_vision_info(messages)
        encoding = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_seq_len,
        )

        encoding["labels"] = torch.tensor(labels, dtype=torch.float32).unsqueeze(0)
        return encoding
    
class DatasetCollection(torch.utils.data.Dataset):
    def __init__(self, datasets: List[torch.utils.data.Dataset], balancing=False):
        self.datasets = datasets
        self.lengths = [len(dataset) for dataset in datasets]
        self.total_len = sum(self.lengths)
        if balancing:
            sqrt_lengths = [math.sqrt(length) for length in self.lengths]
            sum_sqrt_lengths = sum(sqrt_lengths)
            sampling_probs = [sqrt_length / sum_sqrt_lengths for sqrt_length in sqrt_lengths]
            self._lengths = [int(self.total_len * min(prob * 1.1, 1)) for prob in sampling_probs]
            self.total_len = sum(self._lengths)
            self.cum_lengths = [0] + list(np.cumsum(self._lengths))
        else:
            self.cum_lengths = [0] + list(np.cumsum(self.lengths))
    
    def __len__(self):
        return self.total_len
    
    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cum_lengths, idx) - 1
        sub_idx = (idx - self.cum_lengths[dataset_idx]) % self.lengths[dataset_idx]
        return self.datasets[dataset_idx][sub_idx]
    

class Collator():
    def __init__(self, processor, max_length=None):
        self.processor = processor
        self.max_length = max_length
    
    def _right_pad_inputs_with_attention_mask(self, model_inputs: List[Dict]):
        results = {}
        assert len(model_inputs) == 1, "This method only supports a single input, but get {} inputs".format(len(model_inputs))
        for k in model_inputs[0].keys():
            if "pixel_values" in k and isinstance(model_inputs[0][k], list):
                results[k] = [inputs[k] if inputs[k] is not None else None for inputs in model_inputs]
            elif model_inputs[0][k] is not None:
                results[k] = torch.cat([inputs[k] for inputs in model_inputs], dim=0)
            else:
                results[k] = None
        return results
    
    def __call__(self, batch):
        if not hasattr(self.processor, "_right_pad_inputs_with_attention_mask"):
            batch_encoding = self._right_pad_inputs_with_attention_mask(model_inputs=batch)
        else:
            batch_encoding = self.processor._right_pad_inputs_with_attention_mask(model_inputs=batch)
            
        # # print shapes in the batch
        # for k, v in batch_encoding.items():
        #     if isinstance(v, torch.Tensor):
        #         print(k, v.shape)
        #     elif isinstance(v, list):
        #         print(k, len(v), [x.shape for x in v])
        #     else:
        #         print(k, v)
        
        return batch_encoding

class SiglipVideoCollator():
    def __init__(self, processor, max_length=None):
        self.processor = processor
        self.max_length = max_length
    
    def __call__(self, batch):
        all_input_ids = []
        max_input_ids_len = max([x["input_ids"].shape[1] for x in batch])
        
        for x in batch:
            all_input_ids.append(torch.cat([
                x["input_ids"], self.processor.tokenizer.pad_token_id * torch.ones((1, max_input_ids_len - x["input_ids"].shape[1]), dtype=torch.long)
            ], dim=1))
        all_input_ids = torch.cat(all_input_ids, dim=0)

        pixel_values = [x["pixel_values"] for x in batch]
        return {
            "input_ids": all_input_ids,
            "pixel_values": pixel_values,
            "return_loss": True
        }

def load_data_from_config(data_args, processor):
    """
    Returns:
        all_datasets: Dict[str, List[Dataset]], mapping from split to list of datasets
        collator_fn: Callable
    """
    with open(data_args.data_config_file, "r") as f:
        data_config = yaml.load(f, Loader=yaml.FullLoader)
    data_kwargs = {}
    data_kwargs["max_seq_len"] = data_args.max_seq_len
    print("Max Context Length:", data_args.max_seq_len)
    all_datasets = {}
    collator_class = Collator
    for sub_dataset_config in data_config['data']:
        num_proc = sub_dataset_config.get('max_seq_len', 8)
        max_seq_len = sub_dataset_config.get('max_seq_len', data_args.max_seq_len)
        data_path = sub_dataset_config['path']
        name = sub_dataset_config['name']
        split = sub_dataset_config['split']
        max_size = sub_dataset_config.get('max_size', None)
        shuffle = sub_dataset_config.get('shuffle', False)
        max_num_images = sub_dataset_config.get('max_num_images', None)
        max_num_frames = sub_dataset_config.get('max_num_frames', None)
        dataset_type = sub_dataset_config.get('type', 'huggingface')
        offline_sha = sub_dataset_config.get('offline_sha', None)
        vl_only = sub_dataset_config.get('vl_only', False)
        revision = sub_dataset_config.get('revision', "script")
        video_dir = sub_dataset_config.get('video_dir', None)
        max_image_size = sub_dataset_config.get('max_image_size', None)
        fps = sub_dataset_config.get('fps', None)
        assert split in ['train', 'val', 'test'], f"Unknown split {split}"
        if sub_dataset_config['format'] == 'chat':
            sub_dataset = ChatDataset(processor, data_path, dataset_type, name, split, max_seq_len, data_args.conv_format,
                data_args.is_master_worker, max_size, shuffle, max_num_images, vl_only, 
                offline_sha=offline_sha, revision=revision, max_image_size=max_image_size, num_proc=num_proc)
        elif sub_dataset_config['format'] == 'chat_video':
            sub_dataset = ChatVideoDataset(processor, data_path, dataset_type, name, video_dir, split, max_seq_len, data_args.conv_format,
                data_args.is_master_worker, max_size, shuffle, max_num_frames, fps=fps, use_video_encoder=data_args.use_video_encoder if hasattr(data_args, "use_video_encoder") else False)
        elif sub_dataset_config['format'] == 'classification':
            sub_dataset = ClassificationDataset(processor, data_path, dataset_type, name, split, max_seq_len,
                data_args.is_master_worker, max_size, shuffle, max_num_images, vl_only, offline_sha=offline_sha, revision=revision)
        elif sub_dataset_config['format'] == 'qwen2_video_classification':
            sub_dataset = Qwen2VideoClassificationDataset(processor, data_path, dataset_type, name, split, max_seq_len,
                data_args.is_master_worker, max_size, shuffle, max_num_images, vl_only, offline_sha=offline_sha, revision=revision, fps=fps)
        elif sub_dataset_config['format'] == 'siglip_video':
            sub_dataset = SiglipVideoDataset(processor, data_path, dataset_type, name, video_dir, split, max_seq_len,
                data_args.is_master_worker, max_size, shuffle, max_num_frames, fps=fps)
            collator_class = SiglipVideoCollator
        else:
            raise ValueError(f"Unknown data format {sub_dataset_config['format']}")
        if split not in all_datasets:
            all_datasets[split] = []
        all_datasets[split].append(sub_dataset)
    collator_fn = collator_class(processor, max_length=data_args.max_seq_len)
    
    train_dataset = DatasetCollection(all_datasets['train'], data_args.dataset_balancing) if 'train' in all_datasets else None
    val_dataset = DatasetCollection(all_datasets['val'], data_args.dataset_balancing) if 'val' in all_datasets else None
    test_dataset = DatasetCollection(all_datasets['test'], data_args.dataset_balancing) if 'test' in all_datasets else None
    return train_dataset, val_dataset, test_dataset, collator_fn
