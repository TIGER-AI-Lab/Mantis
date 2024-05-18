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

def set_ignore_index(new_ignore_index=-100):
    global IGNORE_INDEX
    IGNORE_INDEX = new_ignore_index

def set_default_image_token(new_default_image_token="<image>"):
    global DEFAULT_IMAGE_TOKEN
    DEFAULT_IMAGE_TOKEN = new_default_image_token

def set_default_image_token_id(new_default_image_token_id=None):
    global DEFAULT_IMAGE_TOKEN_ID
    DEFAULT_IMAGE_TOKEN_ID = new_default_image_token_id

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
                conversations.append(conv_messages)
                all_images.append(image_file)
            except Exception as e:
                print(f"Error at {i}")
                print(e)
        
        return conversations, all_images
        
    def __len__(self):
        return len(self.conversations)
    
    def __getitem__(self, idx):
        conv_messages = self.conversations[idx]
        sub_images = self.all_images[idx]
        sub_images = load_images(sub_images)
        # resize sub_images to be at least 16 * 16 if image is too small, to avoid errors in clip image processor
        if sub_images:
            for i, image in enumerate(sub_images):
                if image.size[0] < 16 or image.size[1] < 16:
                    scale_factor = max(16 / image.size[0], 16 / image.size[1])
                    sub_images[i] = image.resize((int(image.size[0] * scale_factor), int(image.size[1] * scale_factor))).convert("RGB")
                    
        if self.conv.sep_style == SeparatorStyle.PLAIN:
            # NOTE: this is for the pretraining, where we only use the pure text or interleaved text and images
            source = conv_messages
            assert len(source) == 2, "we only use the text in the second message for pretraining."
            # assert DEFAULT_IMAGE_TOKEN in source[0][1]
            # assert len(sub_images) == 1 if isinstance(sub_images, list) else isinstance(sub_images, PIL.Image.Image)
            if isinstance(sub_images, PIL.Image.Image):
                sub_images = [sub_images]
            text = source[1][1]
            image_token_count = source[1][1].count(DEFAULT_IMAGE_TOKEN)
            if image_token_count < len(sub_images):
                text = f"{DEFAULT_IMAGE_TOKEN} " * (len(sub_images) - image_token_count) + text
            conv_str = text + self.conv.sep
            encoding = self.processor(conv_str, sub_images, return_tensors="pt", truncation=True, max_length=self.max_seq_len)
        else:
            # NOTE: this is for the conversation style finetuning
            # check the number of images
            image_token_count = sum([message[1].count(DEFAULT_IMAGE_TOKEN) for message in conv_messages])
            if isinstance(sub_images, list):
                if image_token_count < len(sub_images):
                    conv_messages[0][1] = DEFAULT_IMAGE_TOKEN * (len(sub_images) - image_token_count) + conv_messages[0][1]
            self.conv.messages = conv_messages
            conv_str = self.conv.get_prompt()
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
        elif self.conv.sep_style == SeparatorStyle.IDEFICS_2:
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
            print("no labels for a sample in ", self.data_path, self.name, self.split)
        
        # print(self.data_path, self.name, self.split)
        
        # for debug, print the targets to make sure the right tokens are learned
        # need to print to make sure that the masked tokens are correct.
        # _target = target.clone().detach()
        # _target[_target == IGNORE_INDEX] = 0
        # print(self.processor.tokenizer.decode(input_ids, skip_special_tokens=False))
        # print(self.processor.tokenizer.decode(_target, skip_special_tokens=False))
        

        return encoding

    @staticmethod
    def get_collator_fn(processor, max_length=None):
        def collator_fn(batch):
            batch_encoding = processor._right_pad_inputs_with_attention_mask(model_inputs=batch)
            return batch_encoding

        return collator_fn

class VQADataset(torch.utils.data.Dataset):
    def __init__(self, processor, data_path, name, split, max_seq_len=1024):
        self.processor = processor
        self.data_path = Path(data_path)
        self.name = name
        self.split = split
        # self.data = load_json_data(data_file)
    
    def preprocess(self):
        pass
        
    def __len__(self):
        pass
    def __getitem__(self, idx):
        pass
    
    @staticmethod
    def get_collator_fn(processor):
        pass

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
            if k == "pixel_values" and isinstance(model_inputs[0][k], list):
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
    for sub_dataset_config in data_config['data']:
        max_seq_len = sub_dataset_config.get('max_seq_len', data_args.max_seq_len)
        data_path = sub_dataset_config['path']
        name = sub_dataset_config['name']
        split = sub_dataset_config['split']
        max_size = sub_dataset_config.get('max_size', None)
        shuffle = sub_dataset_config.get('shuffle', False)
        max_num_images = sub_dataset_config.get('max_num_images', None)
        dataset_type = sub_dataset_config.get('type', 'huggingface')
        offline_sha = sub_dataset_config.get('offline_sha', None)
        vl_only = sub_dataset_config.get('vl_only', False)
        revision = sub_dataset_config.get('revision', "script")
        assert split in ['train', 'val', 'test'], f"Unknown split {split}"
        if sub_dataset_config['format'] == 'chat':
            sub_dataset = ChatDataset(processor, data_path, dataset_type, name, split, max_seq_len, data_args.conv_format,
                data_args.is_master_worker, max_size, shuffle, max_num_images, vl_only, offline_sha=offline_sha, revision=revision)
        elif sub_dataset_config['format'] == 'vqa':
            sub_dataset = VQADataset(processor, data_path, name, split, max_seq_len)
        else:
            raise ValueError(f"Unknown data format {sub_dataset_config['format']}")
        if split not in all_datasets:
            all_datasets[split] = []
        all_datasets[split].append(sub_dataset)
    collator_fn = Collator(processor, max_length=data_args.max_seq_len)
    
    train_dataset = DatasetCollection(all_datasets['train'], data_args.dataset_balancing) if 'train' in all_datasets else None
    val_dataset = DatasetCollection(all_datasets['val'], data_args.dataset_balancing) if 'val' in all_datasets else None
    test_dataset = DatasetCollection(all_datasets['test'], data_args.dataset_balancing) if 'test' in all_datasets else None
    return train_dataset, val_dataset, test_dataset, collator_fn

def load_data(data_args, processor):
    """
    Args:
        data_args: DataArguments
        processor: FuyuProcessor
    Returns:
        train_dataset: Dataset
        val_dataset: Dataset
        test_dataset: Dataset
        collator_fn: Callable
    """
    print("Loading data...")

    data_kwargs = {}
    data_kwargs["max_seq_len"] = data_args.max_seq_len
    print("Max Context Length:", data_args.max_seq_len)
    if data_args.data_format == "chat":
        train_dataset = ChatDataset(processor, data_args.train_data_file, **data_kwargs) if data_args.train_data_file else None
        val_dataset = ChatDataset(processor, data_args.val_data_file, **data_kwargs) if data_args.val_data_file else None
        test_dataset = ChatDataset(processor, data_args.test_data_file, **data_kwargs) if data_args.test_data_file else None
        # collator_fn = ChatDataset.get_collator_fn(processor)
        print("Successfully loaded llava-instruct data.")
    elif data_args.data_format == "vqa":
        train_dataset = VQADataset(processor, data_args.train_data_file, **data_kwargs) if data_args.train_data_file else None
        val_dataset = VQADataset(processor, data_args.val_data_file, **data_kwargs) if data_args.val_data_file else None
        test_dataset = VQADataset(processor, data_args.test_data_file, **data_kwargs) if data_args.test_data_file else None
        # collator_fn = VQADataset.get_collator_fn(processor)
        print("Successfully loaded vqa data.")
    else:
        raise ValueError(f"Unknown data format {data_args.data_format}")
    collator_fn = Collator(processor)
    # collator_fn = collator_fn or default_data_collator
    return train_dataset, val_dataset, test_dataset, collator_fn
