from huggingface_hub import hf_hub_download
# data_insf.json
# data_lab.json
# data_real.json
# data_static.json
# data_worsen_gen.json
# frames_insf.zip
# frames_lab.zip
# frames_real.zip
# frames_static.zip
# frames_worsen_gen.zip

hf_hub_download(repo_id="TIGER-Lab/Video_Eval", filename="subset20k_for_mantis/data_insf.json", repo_type="dataset", local_dir="./data")
hf_hub_download(repo_id="TIGER-Lab/Video_Eval", filename="subset20k_for_mantis/data_lab.json", repo_type="dataset", local_dir="./data")
hf_hub_download(repo_id="TIGER-Lab/Video_Eval", filename="subset20k_for_mantis/data_real.json", repo_type="dataset", local_dir="./data")
hf_hub_download(repo_id="TIGER-Lab/Video_Eval", filename="subset20k_for_mantis/data_static.json", repo_type="dataset", local_dir="./data")
hf_hub_download(repo_id="TIGER-Lab/Video_Eval", filename="subset20k_for_mantis/data_worsen_gen.json", repo_type="dataset", local_dir="./data")
hf_hub_download(repo_id="TIGER-Lab/Video_Eval", filename="subset20k_for_mantis/frames_insf.zip", repo_type="dataset", local_dir="./data")
hf_hub_download(repo_id="TIGER-Lab/Video_Eval", filename="subset20k_for_mantis/frames_lab.zip", repo_type="dataset", local_dir="./data")
hf_hub_download(repo_id="TIGER-Lab/Video_Eval", filename="subset20k_for_mantis/frames_real.zip", repo_type="dataset", local_dir="./data")   
hf_hub_download(repo_id="TIGER-Lab/Video_Eval", filename="subset20k_for_mantis/frames_static.zip", repo_type="dataset", local_dir="./data")
hf_hub_download(repo_id="TIGER-Lab/Video_Eval", filename="subset20k_for_mantis/frames_worsen_gen.zip", repo_type="dataset", local_dir="./data")

"""
frames_insf.zip
frames_lab.zip
frames_real.zip
frames_static.zip
frames_worsen_gen.zip
unzip frames_insf.zip
unzip frames_lab.zip
unzip frames_real.zip
unzip frames_static.zip
unzip frames_worsen_gen.zip
"""