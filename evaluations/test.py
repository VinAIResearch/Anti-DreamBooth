import os
from ism_fdfr import matching_score_genimage_id
from tqdm import tqdm
import torch


data_path = "your_path"
fake_path = "your_path"

prompts = ["a_dslr_portrait_of_sks_person", "a_photo_of_a_sks_person"]
result = {
    "a_dslr_portrait_of_sks_person": {
            "ism": [],
            "fdfr": []
        },
    "a_photo_of_a_sks_person": {
            "ism": [],
            "fdfr": []
    }
}

for idx in tqdm(os.listdir(data_path)):
    idx_data_dir = os.path.join(data_path, idx, "set_A"), os.path.join(data_path, idx, "set_B")
    mid_dir = "{}_DREAMBOOTH/checkpoint-1000/dreambooth".format(idx)
    for prompt in prompts:
        idx_fake_dir = os.path.join(fake_path, mid_dir, prompt)
        ism, fdfr = matching_score_genimage_id(idx_fake_dir, idx_data_dir)
        result[prompt]["fdfr"].append(fdfr)
        if ism is None:
            continue
        result[prompt]["ism"].append(ism)
    
for prompt in prompts:
    print("{} ism: {}".format(prompt, torch.mean(torch.stack(result[prompt]["ism"]))))
    print("{} fdfr: {}".format(prompt, torch.mean(torch.tensor(result[prompt]["fdfr"]))))
    
    
