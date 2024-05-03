import json
import fire
import os
import pandas as pd
from tqdm import tqdm


class NextQA():
    def __init__(self, split='train'):
        self.split=split
        self.data = pd.read_csv(f'data/nextqa/{split}.csv')
        self.answer_mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}
        self.num_options = 5
        
    def _get_text(self, idx) -> dict:
        question = self.data["question"].values[idx].capitalize().strip()
        if question[-1] != "?":
            question = str(question) + "?"

        options = [f"{self.answer_mapping[i]}: {self.data[f'a{i}'].values[idx]}" \
                   for i in range(self.num_options)]
        answer_idx = self.data['answer'].values[idx]    

        human_text = f"{question}\n"
        for i in range(self.num_options):
            human_text+=f"{options[i]}\n"
        gpt_text = f"{options[answer_idx]}"

        return human_text, gpt_text
    
    def _check_images(self,idx):
        num_frames=8
        vid=str(self.data["video"].values[idx])
        not_exist_frame=[]
        images_folder=f"data/nextqa/images"
        for i in range(num_frames):
            frame_filename = f"{images_folder}/{vid}_{i}.jpg"
            if not os.path.exists(frame_filename):
                print(f"{images_folder}/{vid}_{i}.jpg not exist!\n")
                not_exist_frame.append(f"{vid}_{i}.jpg")
        
        return not_exist_frame
    
    def _get_items(self):
        train_example_dict={}
        num_frames=8
        for idx in tqdm(range(len(self.data))):
            human_text,gpt_text=self._get_text(idx)
            conversation=[{"from":"human", "value":human_text},
                        {"from":"gpt", "value":gpt_text},]
            vid=str(self.data["video"].values[idx])
            self._check_images(idx=idx)
            images=[f"data/nextqa/images/{vid}_{i}.jpg" for i in range(num_frames)]
            
            if vid not in train_example_dict:
                conversation[0]["value"]=f"{human_text}"
                train_example_dict[vid] = \
                    {"id": vid, "images": images, "conversation": conversation}
            else:
                train_example_dict[vid]["conversation"].extend(conversation)
            

        train_example_list = list(train_example_dict.values())        
        return train_example_list


def main(
        output_file,
):  

    QAs=[]
    curr_dataset=NextQA(split="train")
    QAs.extend(curr_dataset._get_items())

    with open(output_file, 'w') as f:
        json.dump(QAs, f, indent=4)

    '''
    curr_dataset=dlist[x](split="val")
    QAs.extend(curr_dataset._get_items())
    output_file=f"val_example_{dataset_mapping[dlist[x]]}.json"
    with open(output_file, 'w') as f:
        json.dump(QAs, f, indent=4) 
    '''
       
if __name__ == '__main__':
    fire.Fire(main)
