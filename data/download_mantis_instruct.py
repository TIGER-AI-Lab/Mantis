"""
This script download mantis subsets with multiple threads. 
After downloading, the images are all prepared you don't need to download them again loading next time
"""
import fire
import concurrent
from datasets import get_dataset_config_names, load_dataset

def download_subset(dataset_repo_id, config_name):
    
    dataset = load_dataset(dataset_repo_id, config_name)
    print(f"Finish downloading {dataset_repo_id}:{config_name}")
    print(dataset)

def main(
    dataset_repo_id="TIGER-Lab/Mantis-Instruct",
    max_workers=8,
):
    config_names = get_dataset_config_names(dataset_repo_id)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for config_name in config_names:
            futures.append(executor.submit(download_subset, dataset_repo_id, config_name))

        for future in concurrent.futures.as_completed(futures):
            future.result()


if __name__ == "__main__":
    fire.Fire(main)
