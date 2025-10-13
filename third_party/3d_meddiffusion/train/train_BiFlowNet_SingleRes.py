
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from collections import OrderedDict
from glob import glob
from time import time
import argparse
import logging
import os
from ddpm.BiFlowNet import  GaussianDiffusion
from ddpm.BiFlowNet import BiFlowNet
from AutoEncoder.model.PatchVolume import patchvolumeAE
import torchio as tio
import copy
from torch.cuda.amp import autocast, GradScaler
import random
from torch.optim.lr_scheduler import StepLR
from dataset.Singleres_dataset import Singleres_dataset
from torch.utils.data import DataLoader
#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

def _ddp_dict(_dict):
    new_dict = {}
    for k in _dict:
        new_dict['module.' + k] = _dict[k]
    return new_dict


#################################################################################
#                                  Training Loop                                #
#################################################################################
def get_optimizer_size_in_bytes(optimizer):
    """Calculates the size of the optimizer state in bytes."""
    size_in_bytes = 0
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                size_in_bytes += v.numel() * v.element_size()
    return size_in_bytes

def format_size(bytes_size):
    """Formats the size in bytes into a human-readable string."""
    if bytes_size < 1024:
        return f"{bytes_size} B"
    elif bytes_size < 1024 ** 2:
        return f"{bytes_size / 1024:.2f} KB"
    elif bytes_size < 1024 ** 3:
        return f"{bytes_size / 1024 ** 2:.2f} MB"
    else:
        return f"{bytes_size / 1024 ** 3:.2f} GB"
class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def main(args):
    """
    Trains a new DiT model.
    """

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    start_epoch = 0
    train_steps = 0
    log_steps = 0
    running_loss = 0

    # Setup DDP:
    dist.init_process_group("nccl")
    print(dist.get_world_size())

    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        if args.ckpt == None:
            experiment_index = len(glob(f"{args.results_dir}/*"))
            model_string_name = args.model 
            experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        else:
            experiment_dir = os.path.dirname(os.path.dirname(args.ckpt))
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        samples_dir = f"{experiment_dir}/samples" 
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(samples_dir, exist_ok= True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # Create model:



    model = BiFlowNet(
            dim=args.model_dim,
            dim_mults=args.dim_mults,
            channels=args.volume_channels,
            init_kernel_size=3,
            cond_classes=args.num_classes,
            learn_sigma=False,
            use_sparse_linear_attn=args.use_attn,
            vq_size=args.vq_size,
            num_mid_DiT = args.num_dit,
            patch_size = args.patch_size
        ).cuda()
    diffusion= GaussianDiffusion(
        channels=args.volume_channels,
        timesteps=args.timesteps,
        loss_type=args.loss_type,
    ).cuda()
    ema = EMA(0.995)
    ema_model = copy.deepcopy(model)
    updata_ema_every = 10
    step_start_ema = 2000
    model = DDP(model.to(device), device_ids=[rank])
    amp = args.enable_amp
    scaler = GradScaler(enabled=amp)
    if args.AE_ckpt:
        AE = patchvolumeAE.load_from_checkpoint(args.AE_ckpt).cuda()
        AE.eval()
    else:
        raise NotImplementedError()

    logger.info(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    print(args.epochs)
    

    if args.ckpt:
        checkpoint = torch.load(args.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(_ddp_dict(checkpoint['model']), strict= True)
        ema_model.load_state_dict(checkpoint['ema'], strict=True)
        scaler.load_state_dict(checkpoint['scaler'])
        opt.load_state_dict(checkpoint['opt'])
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch']
        del checkpoint 
        logger.info(f'Using checkpoint: {args.ckpt}')
    # Setup data:

    dataset = Singleres_dataset(args.data_path, resolution=args.resolution)
    loader = DataLoader(
        dataset=dataset,
        batch_size = args.batch_size, 
        num_workers=args.num_workers,
        shuffle=True,
    )



    if args.ckpt:
        train_steps = int(os.path.basename(args.ckpt).split('.')[0])
        logger.info(f'Inital state: step = {train_steps}, epoch = {start_epoch}')

    model.train()  
    ema_model.eval() 


    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(start_epoch,args.epochs):
        logger.info(f"Beginning epoch {epoch}...")
        for z,y,res in loader:
            b = z.shape[0]
            z = z.to(device)
            y = y.to(device)
            res = res.to(device)
            with autocast(enabled=amp):
                t = torch.randint(0, diffusion.num_timesteps, (b,), device=device)
                loss = diffusion.p_losses(model, z,t,y=y,res=res)
                scaler.scale(loss).backward()

            scaler.step(opt)
            scaler.update()
            opt.zero_grad()          



            running_loss += loss.item()
            log_steps += 1
            train_steps += 1

            if train_steps % updata_ema_every == 0:
                if train_steps < step_start_ema:
                    ema_model.load_state_dict(model.module.state_dict(),strict= True)
                else:
                    ema.update_model_average(ema_model,model)

            if train_steps % args.log_every == 0:

                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}, LR : {opt.state_dict()['param_groups'][0]['lr']:.6f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()
            # Save Diffusion checkpoint:

            # train_steps = args.ckpt_every #
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema_model.state_dict(),
                        "scaler": scaler.state_dict(),
                        "opt":opt.state_dict(),
                        "args": args,
                        "epoch":epoch
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                    if len(os.listdir(checkpoint_dir))>6:
                        os.remove(f"{checkpoint_dir}/{train_steps-6*args.ckpt_every:07d}.pt")
                    with torch.no_grad():
                        milestone = train_steps // args.ckpt_every
                        
                        cls_num = np.random.choice(list(range(0,args.num_classes)))
                        # cls_num = 1 #
                        volume_size = args.resolution
                        z = torch.randn(1, args.volume_channels, volume_size[0], volume_size[1],volume_size[2], device=device)
                        y = torch.tensor([cls_num], device=device)
                        res = torch.tensor(volume_size,device=device)/64.0
                        samples = diffusion.p_sample_loop(
                            ema_model, z, y = y,res=res
                        )
                        samples = (((samples + 1.0) / 2.0) * (AE.codebook.embeddings.max() -
                                                            AE.codebook.embeddings.min())) + AE.codebook.embeddings.min()
                        torch.cuda.empty_cache()


                        volume = AE.decode(samples, quantize=True)


                        volume_path = os.path.join(samples_dir,str(f'{milestone}_{str(cls_num)}.nii.gz')) 
                        volume = volume.detach().squeeze(0).cpu()
                        volume = volume.transpose(1,3).transpose(1,2)
                        tio.ScalarImage(tensor = volume).save(volume_path)
                dist.barrier()
                torch.cuda.empty_cache()
            # scheduler.step()
    model.eval()  # important! This disables randomized embedding dropout
   

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--loss-type", type=str, default='l1')
    parser.add_argument("--volume-channels", type=int, default=8)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--model-dim", type=int, default=72)
    parser.add_argument("--dim-mults", nargs='+', type=int, default=[1,1,2,4,8])
    parser.add_argument("--use-attn", nargs='+', type=int, default=[0,0,0,1,1])
    parser.add_argument("--patch-size", type=int, default=1)
    parser.add_argument("--num-dit", type=int, default=1)
    parser.add_argument("--enable_amp", type=bool, default=True)
    parser.add_argument("--model", type=str,default="BiFlowNet")
    parser.add_argument("--AE-ckpt", type=str, required=True)
    parser.add_argument("--num-classes", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=8) 
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument('--resolution', nargs='+', type=int, default=[32, 32, 32])
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--ckpt-every", type=int, default=500)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--vq-size", type=int, default=64)
    args = parser.parse_args()
    main(args)
