

import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import os
import glob
import numpy as np
import json 
import torchio as tio
class Singleres_dataset(Dataset):
    def __init__(self, root_dir=None, resolution= [32,32,32], generate_latents= False):
        self.all_files = []
        self.resolution = resolution
        self.generate_latents = generate_latents
        if root_dir.endswith('json'):
            with open(root_dir) as json_file:
                dataroots = json.load(json_file)

            for key,value in dataroots.items():
                if not generate_latents:
                    value = value+'_latents'
                file_paths =  glob.glob(value+'/*.nii.gz', recursive=True)
                if len(file_paths[0]) == 0:
                    raise FileNotFoundError(f"No .nii.gz files found in directory: {value}")
                for file_path in file_paths:
                    self.all_files.append({key:file_path})

            self.file_num = len(self.all_files)
            print(f"Total files : {self.file_num}")
        

    def __len__(self):
        return self.file_num


    def __getitem__(self, index):
        if self.generate_latents:
            file_path = list(self.all_files[index].items())[0][1]
            img = tio.ScalarImage(file_path)
            img_data = img.data.to(torch.float32)
            imageout = img_data * 2 - 1
            imageout = imageout.transpose(1,3).transpose(2,3)
            return imageout, file_path
        else:
            cls_idx , file_path = list(self.all_files[index].items())[0][0] , list(self.all_files[index].items())[0][1]
            latent = tio.ScalarImage(file_path)
            latent = latent.data.to(torch.float32)
            return latent, torch.tensor(int(cls_idx)), torch.tensor(self.resolution)/64.0