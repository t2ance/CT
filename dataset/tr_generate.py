
import torch
from torch.utils.data.dataset import Dataset
import os
import glob
import torchio as tio
import json 


class GenerateTrData_dataset(Dataset):
    def __init__(self, root_dir=None,no_norm=False):
        self.no_norm = no_norm
        self.root_dir = root_dir
        self.all_files = glob.glob(os.path.join(root_dir, './*.nii.gz'))
        self.file_num = len(self.all_files)
        print(f"Total files : {self.file_num}")
        print('no_norm:',self.no_norm)

    def __len__(self):
        return self.file_num


    def __getitem__(self, index):
        path = self.all_files[index]
        img = tio.ScalarImage(path)
        imageout = img.data 
        if not self.no_norm:
            imageout = imageout * 2 - 1
        imageout = imageout.transpose(1,3).transpose(2,3)
        imageout = imageout.type(torch.float32)
        return imageout, path