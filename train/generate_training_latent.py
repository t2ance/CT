

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)
from torch.utils.data import DataLoader
from AutoEncoder.model.PatchVolume import patchvolumeAE 
from dataset.Singleres_dataset import Singleres_dataset
import torch
from os.path import join
import argparse
import torchio as tio
os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"


def generate(args):

    tr_dataset = Singleres_dataset(root_dir=args.data_path,generate_latents = True)

    tr_dataloader = DataLoader(tr_dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    AE_ckpt = args.AE_ckpt
    AE = patchvolumeAE.load_from_checkpoint(AE_ckpt)
    AE = AE.to(device)
    AE.eval()
    

    for sample,paths in tr_dataloader:
        sample = sample.cuda()
        with torch.no_grad():
            z =  AE.patch_encode(sample,patch_size = 64)
            output = ((z - AE.codebook.embeddings.min()) /
            (AE.codebook.embeddings.max() -
            AE.codebook.embeddings.min())) * 2.0 - 1.0
        output = output.cpu()
        for idx, path in enumerate(paths):
            output_ = output[idx]
            dir_name = os.path.basename(os.path.dirname(path))
            latent_dir_name = dir_name + '_latents'
            path = path.replace(dir_name, latent_dir_name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            img = tio.ScalarImage(tensor = output_ )
            img.save(path)   

if __name__ == "__main__":
 
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--AE-ckpt", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=2) 
    parser.add_argument("--num-workers", type=int, default=8) 
    args = parser.parse_args()
    generate(args)



