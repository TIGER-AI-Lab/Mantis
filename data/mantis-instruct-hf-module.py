# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# TODO: Address all TODOs and remove all explanatory comments
"""TODO: Add a description here."""


import os
import datasets
import glob
import shutil
import concurrent
from datasets import Dataset, DownloadManager, DatasetDict
from datasets.packaged_modules.parquet.parquet import Parquet
from typing import Optional, Dict, Mapping
from datasets.load import HubDatasetModuleFactoryWithoutScript
from functools import partial
from datasets import DatasetBuilder
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path

# TODO: Name of the dataset usually matches the script name with CamelCase instead of snake_case

def map_image_path_to_absolute_path(image_dir, item):
    for sub_images in item["images"]:
        if sub_images:
            for image in sub_images:
                image["path"] = os.path.abspath(os.path.join(image_dir, image["path"]))
                assert os.path.exists(image["path"]), f"Image file {image['path']} does not exist."
    return item
    
def move_folder(src, dst):
    src = Path(src) if isinstance(src, str) else src
    dst = Path(dst) if isinstance(dst, str) else dst
    if not dst.exists():
        shutil.move(src, dst)
    else:
        for file in src.iterdir():
            if file.is_file():
                shutil.move(file, dst)
            # directory
            elif file.is_dir():
                move_folder(file, dst/file.name)
            else:
                raise ValueError(f"Unknown file type: {file}")
        shutil.rmtree(src)

class MIQA(Parquet):
    """TODO: Short description of my dataset."""

    VERSION = datasets.Version("1.0.0")
    
    data_module = HubDatasetModuleFactoryWithoutScript("TIGER-Lab/Mantis-Instruct").get_module() # change this to the correct dataset name
    BUILDER_CONFIGS = data_module.builder_configs_parameters.builder_configs
    DEFAULT_CONFIG_NAME = data_module.builder_configs_parameters.default_config_name
    
    cache_files = defaultdict(list)
    
    def _post_process(self, dataset: Dataset, resources_paths: Mapping[str, str]) -> Optional[Dataset]:
        """Run dataset transforms or add indexes"""
        
        if isinstance(dataset, DatasetDict):
            for split, ds in dataset.items():
                dataset[split] = self._post_process(ds, resources_paths)
            return dataset
        assert isinstance(dataset, Dataset)
        resource_name = f"{dataset.split}_images"
        image_dir = resources_paths[resource_name].replace(".done", "")
        
        if resource_name in self.cache_files:
            to_move_cache_files = list(set(self.cache_files[resource_name]))
            for file in to_move_cache_files:
                # move the image folder to the target position
                # print(f"Moving {file} to {image_dir}...")
                move_folder(file, image_dir)
            self.cache_files.pop(resource_name)
        ds = dataset.map(partial(map_image_path_to_absolute_path, image_dir), batched=True)
        return ds

    def _post_processing_resources(self, split: str) -> Dict[str, str]:
        """Mapping resource_name -> resource_file_name"""
        return {
            f"{split}_images": f"{split}_images.done",
        }
    
    def _download_post_processing_resources(
        self, split: str, resource_name: str, dl_manager: DownloadManager
    ) -> Optional[str]:
        """Download the resource using the download manager and return the downloaded path."""
        
        if resource_name == f"{split}_images":
            image_zip_list_file = os.path.join(self.config.name, f"{split}_images_zips.txt")
            try:
                local_image_zip_list_file = dl_manager.download(image_zip_list_file)
                split_dir = local_image_zip_list_file[:local_image_zip_list_file.rfind("/")]
                with open(local_image_zip_list_file, "r") as f:
                    image_zips = f.readlines()
            except Exception as e:
                image_zips = [resource_name+".zip"]
                
            # multi-threading download and extracting
            num_proc = dl_manager.download_config.num_proc if dl_manager.download_config.num_proc is not None else 1
            print(f"Downloading and extracting {len(image_zips)} image zip files... (num_proc={num_proc})")
            futures = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_proc) as executor:
                for image_zip in image_zips:
                    resource_in_repo = os.path.join(self.config.name, image_zip.strip())

                    # submit download and extract task
                    futures.append(executor.submit(dl_manager.download_and_extract, resource_in_repo))
                    
                    # resource_dir = dl_manager.download_and_extract(resource_in_repo)
                    
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                resource_dir = future.result()
                self.cache_files[resource_name].append(resource_dir)
                
            # create a .done file to indicate the resource is already downloaded and extracted
            if os.path.exists(split_dir):
                resource_file = os.path.join(split_dir, f"{split}_images.done")
                with open(resource_file, "w") as f:
                    for image_file in glob.glob(os.path.join(resource_dir, "*",)):
                        f.write(image_file+"\n")
                return resource_file
            else:
                return None
        return None