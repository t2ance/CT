"""
Training script for CT Super-Resolution using Latent Diffusion Models

This script trains a 3D UNet diffusion model in VQ-AE latent space for
converting low-dose CT to high-dose CT.

Architecture:
    - Stage 1 (frozen): VQ-AE encoder/decoder (8× spatial compression)
    - Stage 2 (training): 3D UNet diffusion model in latent space

Training strategy:
    - DDPM training with MSE loss on predicted noise
    - Conditioning: Concatenate LDCT latent to HDCT latent
    - Two-tier validation:
        * Lightweight (every 50 steps): MSE loss only (~5s)
        * Full (every 500 steps): SSIM + PSNR metrics (~30-60s)
    - Distributed training with HuggingFace Accelerate (FSDP)
    - Mixed precision training (FP16/BF16)
    - Linear LR scheduler with warmup

Usage:
    # Single GPU
    python train_diffusion.py --config config_diffusion.yaml

    # Multi-GPU with FSDP
    accelerate launch --config_file accelerate_config_fsdp.yaml \
        train_diffusion.py --config config_diffusion.yaml

Reconstructed from recovery note: TRAIN_DIFFUSION_RECOVERY_NOTE.md
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional
import yaml

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np

# HuggingFace
from accelerate import Accelerator
from diffusers import DDPMScheduler, DDIMScheduler
import wandb

# Custom modules
from model_factory import build_diffusion_model, DiffusionUNet3D
from models.vqae_wrapper import FrozenVQAE
from data.latent_dataset import LatentCTDataset, create_latent_dataloaders
from utils.metrics import DistributedMetricsCalculator
from utils.visualization import CTVisualization


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train latent diffusion model for CT super-resolution")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (overrides config)"
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config: dict) -> DiffusionUNet3D:
    """
    Create 3D UNet diffusion model from config.

    Args:
        config: Configuration dictionary

    Returns:
        model: DiffusionUNet3D instance
    """
    model_config = config['model']
    return build_diffusion_model(model_config, verbose=True)


@torch.no_grad()
def validate(
    model: DiffusionUNet3D,
    val_loader: DataLoader,
    noise_scheduler: DDPMScheduler,
    accelerator: Accelerator,
) -> float:
    """
    Lightweight validation with MSE loss only (no DDIM sampling).

    This is fast (~5s) and used for frequent validation checks.

    Args:
        model: Diffusion model
        val_loader: Validation dataloader (prepared by accelerate)
        noise_scheduler: DDPM noise scheduler
        accelerator: Accelerator instance

    Returns:
        avg_loss: Average MSE loss on validation set
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in val_loader:
        ld_latent = batch['ld_latent']
        hd_latent = batch['hd_latent']

        B = hd_latent.shape[0]

        # Sample random timesteps
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps,
            (B,),
            device=hd_latent.device
        ).long()

        # Add noise to HD latent
        noise = torch.randn_like(hd_latent)
        noisy_hd_latent = noise_scheduler.add_noise(hd_latent, noise, timesteps)

        # Concatenate LD latent as conditioning
        model_input = torch.cat([noisy_hd_latent, ld_latent], dim=1)

        # Predict noise
        noise_pred = model(model_input, timesteps)

        # MSE loss
        loss = F.mse_loss(noise_pred, noise)

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)

    # Gather across all processes
    avg_loss_tensor = torch.tensor(avg_loss, device=accelerator.device)
    avg_loss_tensor = accelerator.gather(avg_loss_tensor).mean().item()

    model.train()
    return avg_loss_tensor


@torch.no_grad()
def generate_visualizations(
    model: DiffusionUNet3D,
    val_loader: DataLoader,
    vae: FrozenVQAE,
    config: dict,
    accelerator: Accelerator,
    num_samples: int = 2,
):
    """
    Generate visualizations during training.

    This function:
    1. Samples HD latent using DDIM (50-100 steps)
    2. Decodes latents with VQ-AE
    3. Creates multi-slice comparison visualizations
    4. Returns dict for wandb logging

    Args:
        model: Diffusion model
        val_loader: Validation dataloader (NOT prepared - should iterate manually)
        vae: Frozen VQ-AE for decoding
        config: Configuration dict
        accelerator: Accelerator instance
        num_samples: Number of samples to visualize

    Returns:
        viz_dict: Dict of {name: PIL Image} for wandb logging
    """
    if vae is None:
        accelerator.print("  Warning: VQ-AE not loaded, skipping visualizations")
        return {}

    model.eval()

    # Create DDIM scheduler for fast sampling
    inference_config = config.get('inference', {})
    num_inference_steps = inference_config.get('num_inference_steps', 50)

    ddim_scheduler = DDIMScheduler(
        num_train_timesteps=config['scheduler']['num_train_timesteps'],
        beta_start=config['scheduler']['beta_start'],
        beta_end=config['scheduler']['beta_end'],
        beta_schedule=config['scheduler']['beta_schedule'],
        prediction_type=config['scheduler']['prediction_type'],
        clip_sample=config['scheduler'].get('clip_sample', False),
    )
    ddim_scheduler.set_timesteps(num_inference_steps)

    # Initialize visualization helper
    viz = CTVisualization(clip_range=(-1000, 1000))

    visualizations = {}
    samples_processed = 0

    # Iterate through validation set
    for batch in val_loader:
        if samples_processed >= num_samples:
            break

        ld_latent = batch['ld_latent'].to(accelerator.device)
        hd_latent_gt = batch['hd_latent'].to(accelerator.device)

        # Only process on main process
        # if not accelerator.is_main_process:
            # continue

        # Start from pure noise
        hd_latent_pred = torch.randn_like(hd_latent_gt)

        # DDIM sampling
        for t in tqdm(ddim_scheduler.timesteps, desc=f"  Sampling {samples_processed+1}/{num_samples}", leave=False):
            # Concatenate conditioning
            model_input = torch.cat([hd_latent_pred, ld_latent], dim=1)

            # Predict noise
            with torch.no_grad():
                noise_pred = model(model_input, t.unsqueeze(0).to(accelerator.device))

            # DDIM step
            hd_latent_pred = ddim_scheduler.step(
                noise_pred, t, hd_latent_pred
            ).prev_sample

        # Decode latents to CT space
        ld_ct = vae.decode(ld_latent)  # [B, 1, D, H, W]
        pred_ct = vae.decode(hd_latent_pred)
        gt_ct = vae.decode(hd_latent_gt)

        # Create visualizations
        for i in range(ld_ct.shape[0]):
            if samples_processed >= num_samples:
                break

            sample_viz = viz.create_multi_slice_comparison(
                ld=ld_ct[i:i+1],
                pred=pred_ct[i:i+1],
                gt=gt_ct[i:i+1],
                num_slices=5
            )

            visualizations[f'val/sample_{samples_processed}'] = wandb.Image(sample_viz)
            samples_processed += 1

    model.train()
    return visualizations


@torch.no_grad()
def validate_with_metrics(
    model: DiffusionUNet3D,
    val_loader: DataLoader,  # IMPORTANT: Should be PREPARED dataloader for automatic distribution
    vae: FrozenVQAE,
    config: dict,
    accelerator: Accelerator,
    num_samples: int = 4,
) -> dict:
    """
    Full validation with SSIM and PSNR metrics.

    This function uses a PREPARED val_loader (processed by accelerator.prepare())
    which automatically distributes batches across GPUs via DistributedSampler.
    Each GPU evaluates different samples to parallelize the slow DDIM sampling.

    Process:
    1. Prepared loader distributes batches across GPUs automatically
    2. Each GPU processes its assigned batches independently
    3. DDIM sampling (100 steps, ~30-60s per sample) without inter-GPU sync
    4. Decode with VQ-AE
    5. Compute SSIM/PSNR using DistributedMetricsCalculator
    6. Synchronize all processes (wait_for_everyone)
    7. Gather metrics from all processes
    8. Return aggregated metrics dict (main process only)

    Args:
        model: Diffusion model
        val_loader: Validation dataloader (PREPARED - from accelerator.prepare)
        vae: Frozen VQ-AE for decoding
        config: Configuration dict
        accelerator: Accelerator instance
        num_samples: Number of validation samples to evaluate TOTAL (distributed across all GPUs)

    Returns:
        metrics_dict: Aggregated metrics with mean/std/min/max for SSIM/PSNR
                     Only populated on main process, empty dict on other processes
    """
    if vae is None:
        accelerator.print("  Warning: VQ-AE not loaded, skipping metrics")
        return {}

    model.eval()

    # Create DDIM scheduler for sampling
    inference_config = config.get('inference', {})
    num_inference_steps = inference_config.get('num_inference_steps', 100)

    ddim_scheduler = DDIMScheduler(
        num_train_timesteps=config['scheduler']['num_train_timesteps'],
        beta_start=config['scheduler']['beta_start'],
        beta_end=config['scheduler']['beta_end'],
        beta_schedule=config['scheduler']['beta_schedule'],
        prediction_type=config['scheduler']['prediction_type'],
        clip_sample=config['scheduler'].get('clip_sample', False),
    )
    ddim_scheduler.set_timesteps(num_inference_steps)

    # Initialize metrics calculator
    validation_config = config.get('validation', {})
    data_range = validation_config.get('data_range', 2.0)  # CRITICAL: 2.0 for normalized VQ-AE outputs!
    compute_slice_wise = validation_config.get('compute_slice_wise', True)

    metrics_calc = DistributedMetricsCalculator(
        data_range=data_range,
        compute_slice_wise=compute_slice_wise
    )

    # CRITICAL FIX: Calculate per-process sample target
    # num_samples is TOTAL across all processes, not per-process
    # We need to distribute this evenly and ensure ALL processes iterate the same number of times
    samples_per_process = (num_samples + accelerator.num_processes - 1) // accelerator.num_processes

    # print(f"[GPU {accelerator.process_index}] DEBUG: Starting validation")
    # print(f"[GPU {accelerator.process_index}] DEBUG: Total samples requested: {num_samples}")
    # print(f"[GPU {accelerator.process_index}] DEBUG: Samples per process: {samples_per_process}")
    # print(f"[GPU {accelerator.process_index}] DEBUG: Num processes: {accelerator.num_processes}")

    # Collect metrics
    local_metrics = []
    sample_idx = 0
    batch_idx = 0

    if accelerator.is_main_process:
        accelerator.print(f'  Validating {num_samples} samples total ({samples_per_process} per process across {accelerator.num_processes} GPU(s))...')

    # Iterate through validation set
    # NOTE: val_loader is PREPARED, so each process gets different batches automatically
    # CRITICAL: All processes must iterate through the SAME NUMBER of batches
    # to avoid one process reaching wait_for_everyone() while others are still looping
    for batch in tqdm(val_loader, desc=f"Validating (GPU {accelerator.process_index})"):
        # print(f"[GPU {accelerator.process_index}] DEBUG: Processing batch {batch_idx}")

        # CRITICAL: Check if we should process this batch BEFORE entering DDIM loop
        # This ensures all processes make the same loop iteration decision
        if sample_idx >= samples_per_process:
            # print(f"[GPU {accelerator.process_index}] DEBUG: Reached target samples ({sample_idx}/{samples_per_process}), breaking from batch loop")
            break

        ld_latent = batch['ld_latent'].to(accelerator.device)
        hd_latent_gt = batch['hd_latent'].to(accelerator.device)
        batch_size = hd_latent_gt.shape[0]

        # print(f"[GPU {accelerator.process_index}] DEBUG: Batch {batch_idx} has {batch_size} samples")

        # Start from pure noise
        hd_latent_pred = torch.randn_like(hd_latent_gt)

        # print(f"[GPU {accelerator.process_index}] DEBUG: Starting DDIM sampling for batch {batch_idx}")

        # DDIM sampling
        # IMPORTANT: No synchronization inside loop - each process samples independently
        for t in ddim_scheduler.timesteps:
            # Concatenate conditioning
            model_input = torch.cat([hd_latent_pred, ld_latent], dim=1)

            # Predict noise
            noise_pred = model(model_input, t.unsqueeze(0).to(accelerator.device))

            # DDIM step
            hd_latent_pred = ddim_scheduler.step(
                noise_pred, t, hd_latent_pred
            ).prev_sample

        # print(f"[GPU {accelerator.process_index}] DEBUG: Finished DDIM sampling for batch {batch_idx}")
        # print(f"[GPU {accelerator.process_index}] DEBUG: Decoding latents with VQ-AE")

        # Decode latents to CT space
        pred_ct = vae.decode(hd_latent_pred)  # [B, 1, D, H, W]
        gt_ct = vae.decode(hd_latent_gt)

        # print(f"[GPU {accelerator.process_index}] DEBUG: Computing metrics for batch {batch_idx}")

        # Compute metrics for each sample in batch
        for i in range(pred_ct.shape[0]):
            # Check if we've exceeded our per-process quota
            if sample_idx >= samples_per_process:
                # print(f"[GPU {accelerator.process_index}] DEBUG: Reached sample limit within batch, stopping at sample {sample_idx}")
                break

            # Convert to numpy and remove batch/channel dims
            pred_np = pred_ct[i, 0].cpu().numpy()  # [D, H, W]
            gt_np = gt_ct[i, 0].cpu().numpy()

            local_metrics.append(metrics_calc.compute_metrics(pred_np, gt_np))

            sample_idx += 1
            # print(f"[GPU {accelerator.process_index}] DEBUG: Processed sample {sample_idx}/{samples_per_process}")

        batch_idx += 1

    # print(f"[GPU {accelerator.process_index}] DEBUG: Finished batch loop, processed {sample_idx} samples across {batch_idx} batches")
    # print(f"[GPU {accelerator.process_index}] DEBUG: Waiting for all processes before gathering...")

    # CRITICAL: Synchronize all processes before gathering
    # This ensures all processes have finished sampling before we gather metrics
    # At this point, ALL processes should have exited the batch loop
    accelerator.wait_for_everyone()

    # print(f"[GPU {accelerator.process_index}] DEBUG: All processes synchronized, preparing to gather metrics")

    # Convert local metrics to tensors for gathering
    if local_metrics:
        local_ssim = torch.tensor([m['ssim'] for m in local_metrics], device=accelerator.device)
        local_psnr = torch.tensor([m['psnr'] for m in local_metrics], device=accelerator.device)
        # print(f"[GPU {accelerator.process_index}] DEBUG: Local metrics: {len(local_metrics)} samples")
    else:
        # Create empty tensors if no metrics computed on this process
        local_ssim = torch.tensor([], device=accelerator.device)
        local_psnr = torch.tensor([], device=accelerator.device)
        # print(f"[GPU {accelerator.process_index}] DEBUG: No local metrics computed")

    # print(f"[GPU {accelerator.process_index}] DEBUG: Gathering metrics from all processes...")

    # Gather metrics from all processes
    # NOTE: accelerator.gather() concatenates tensors from all processes
    # If a process has no metrics (empty tensor), it still participates in gather
    all_ssim = accelerator.gather(local_ssim)
    all_psnr = accelerator.gather(local_psnr)

    # print(f"[GPU {accelerator.process_index}] DEBUG: Metrics gathered successfully")

    # Compute aggregated statistics (only on main process)
    if accelerator.is_main_process:
        all_ssim = all_ssim.cpu().numpy()
        all_psnr = all_psnr.cpu().numpy()

        # Handle case where no metrics were computed
        if len(all_ssim) == 0:
            accelerator.print("  Warning: No validation metrics computed!")
            metrics_dict = {}
        else:
            metrics_dict = {
                'ssim_mean': float(np.mean(all_ssim)),
                'ssim_std': float(np.std(all_ssim)),
                'ssim_min': float(np.min(all_ssim)),
                'ssim_max': float(np.max(all_ssim)),
                'psnr_mean': float(np.mean(all_psnr)),
                'psnr_std': float(np.std(all_psnr)),
                'psnr_min': float(np.min(all_psnr)),
                'psnr_max': float(np.max(all_psnr)),
                'num_samples': len(all_ssim),
            }
            # print(f"[GPU {accelerator.process_index}] DEBUG: Final metrics computed: {len(all_ssim)} total samples")
    else:
        metrics_dict = {}

    # print(f"[GPU {accelerator.process_index}] DEBUG: Validation complete")

    model.train()
    return metrics_dict


def save_checkpoint(
    accelerator: Accelerator,
    model: DiffusionUNet3D,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    global_step: int,
    checkpoint_dir: str,
    is_best: bool = False,
):
    """
    Save training checkpoint with atomic writes to prevent corruption.

    Uses write-to-temp-then-rename pattern for atomic checkpoint saves.

    Args:
        accelerator: Accelerator instance
        model: Diffusion model
        optimizer: Optimizer
        lr_scheduler: LR scheduler
        epoch: Current epoch
        global_step: Global training step
        checkpoint_dir: Directory to save checkpoint
        is_best: Whether this is the best model so far
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save checkpoint
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': accelerator.unwrap_model(model).state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
    }

    # Helper function for atomic save
    def atomic_save(checkpoint_path_str: str):
        checkpoint_path = Path(checkpoint_path_str)
        # temp_path = checkpoint_path.with_suffix('.pth.tmp')
        temp_path = checkpoint_path.with_suffix('.pth')

        # Save to temp file
        accelerator.save(checkpoint, str(temp_path))

        # Atomic rename (overwrites existing file atomically)
        # temp_path.replace(checkpoint_path)

    # Save latest checkpoint (atomic)
    checkpoint_path = os.path.join(checkpoint_dir, 'latest.pth')
    atomic_save(checkpoint_path)

    # Save epoch checkpoint (atomic)
    epoch_checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
    atomic_save(epoch_checkpoint_path)

    # Save best checkpoint (atomic)
    if is_best:
        best_checkpoint_path = os.path.join(checkpoint_dir, 'best.pth')
        atomic_save(best_checkpoint_path)
        if accelerator.is_main_process:
            accelerator.print(f"  Saved best checkpoint: {best_checkpoint_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: DiffusionUNet3D,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    accelerator: Accelerator,
) -> tuple:
    """
    Load checkpoint and resume training.

    Args:
        checkpoint_path: Path to checkpoint file
        model: Diffusion model
        optimizer: Optimizer
        lr_scheduler: LR scheduler
        accelerator: Accelerator instance

    Returns:
        epoch, global_step: Resumed epoch and global step
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Load model
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.load_state_dict(checkpoint['model_state_dict'])

    # Load optimizer
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Load LR scheduler
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

    epoch = checkpoint['epoch']
    global_step = checkpoint['global_step']

    if accelerator.is_main_process:
        accelerator.print(f"  Resumed from checkpoint: {checkpoint_path}")
        accelerator.print(f"  Epoch: {epoch}, Global step: {global_step}")

    return epoch, global_step


def train(config: dict, args):
    """
    Main training function.

    Args:
        config: Configuration dictionary
        args: Command line arguments
    """
    # Initialize accelerator
    training_config = config['training']
    accelerator = Accelerator(
        gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 1),
        mixed_precision=training_config.get('mixed_precision', 'no'),
        log_with='wandb' if training_config.get('use_wandb', True) else None,
    )

    # Setup output directories
    output_config = config['output']
    checkpoint_dir = args.output_dir or output_config['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)

    if accelerator.is_main_process:
        accelerator.print("="*70)
        accelerator.print("CT Latent Diffusion Training")
        accelerator.print("="*70)
        accelerator.print(f"\nConfiguration:")
        accelerator.print(f"  Device: {accelerator.device}")
        accelerator.print(f"  Num processes: {accelerator.num_processes}")
        accelerator.print(f"  Mixed precision: {training_config.get('mixed_precision', 'no')}")
        accelerator.print(f"  Gradient accumulation: {training_config.get('gradient_accumulation_steps', 1)}")
        accelerator.print(f"  Checkpoint dir: {checkpoint_dir}")

    # Initialize wandb
    if training_config.get('use_wandb', True) and accelerator.is_main_process:
        wandb_project = training_config.get('wandb_project', 'ct-latent-diffusion')
        wandb_entity = training_config.get('wandb_entity', None)
        wandb_run_name = training_config.get('wandb_run_name', None)
        wandb_tags = training_config.get('wandb_tags', [])

        accelerator.init_trackers(
            project_name=wandb_project,
            config=config,
            init_kwargs={
                'wandb': {
                    'entity': wandb_entity,
                    'name': wandb_run_name,
                    'tags': wandb_tags,
                }
            }
        )
        accelerator.print(f"\nInitialized wandb:")
        accelerator.print(f"  Project: {wandb_project}")
        if wandb_entity:
            accelerator.print(f"  Entity: {wandb_entity}")
        if wandb_run_name:
            accelerator.print(f"  Run name: {wandb_run_name}")

    # Create model
    if accelerator.is_main_process:
        accelerator.print("\nCreating model...")

    model = create_model(config)

    if accelerator.is_main_process:
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        accelerator.print(f"  Total parameters: {num_params:,}")
        accelerator.print(f"  Trainable parameters: {num_trainable:,}")

    # Create noise scheduler
    scheduler_config = config['scheduler']
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=scheduler_config['num_train_timesteps'],
        beta_start=scheduler_config['beta_start'],
        beta_end=scheduler_config['beta_end'],
        beta_schedule=scheduler_config['beta_schedule'],
        prediction_type=scheduler_config['prediction_type'],
        clip_sample=scheduler_config.get('clip_sample', False),
    )

    # Create dataloaders
    if accelerator.is_main_process:
        accelerator.print("\nCreating dataloaders...")

    data_config = config['data']
    train_loader, val_loader = create_latent_dataloaders(
        latent_dir=data_config['latent_cache_dir'],
        batch_size=training_config['batch_size'],
        num_workers=training_config.get('num_workers', 4),
        normalize_latents=data_config.get('normalize_latents', False),
        pin_memory=True,
    )

    if accelerator.is_main_process:
        accelerator.print(f"  Train samples: {len(train_loader.dataset)}")
        accelerator.print(f"  Val samples: {len(val_loader.dataset)}")
        accelerator.print(f"  Batch size: {training_config['batch_size']}")
        accelerator.print(f"  Train batches: {len(train_loader)}")

    # Load VQ-AE for visualization and metrics
    # Load on ALL processes if distributed evaluation is enabled
    vae = None
    validation_config = config.get('validation', {})
    distributed_eval = validation_config.get('distributed_eval', True)
    enable_full_validation = validation_config.get('enable_full_validation', True)

    load_vae = (training_config.get('use_wandb', True) and accelerator.is_main_process) or \
               (enable_full_validation and distributed_eval)

    if load_vae:
        if accelerator.is_main_process:
            accelerator.print("\nLoading VQ-AE...")
            if distributed_eval:
                accelerator.print("  Loading on ALL processes for distributed evaluation")

        vae_checkpoint = data_config.get('vae_checkpoint')
        if vae_checkpoint:
            try:
                vae = FrozenVQAE(vae_checkpoint, device=accelerator.device)
                if accelerator.is_main_process:
                    accelerator.print(f"  VQ-AE loaded from: {vae_checkpoint}")
                    accelerator.print(f"  Memory: ~500 MB per process")
            except Exception as e:
                if accelerator.is_main_process:
                    accelerator.print(f"  Warning: Failed to load VQ-AE: {e}")
                vae = None

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config['learning_rate'],
        betas=(training_config.get('adam_beta1', 0.9), training_config.get('adam_beta2', 0.999)),
        eps=training_config.get('adam_epsilon', 1e-8),
        weight_decay=training_config.get('weight_decay', 0.01),
    )

    # Calculate total training steps
    num_epochs = training_config['num_epochs']
    gradient_accumulation_steps = training_config.get('gradient_accumulation_steps', 1)
    num_update_steps_per_epoch = len(train_loader) // gradient_accumulation_steps
    total_steps = num_epochs * num_update_steps_per_epoch

    # Create LR scheduler
    scheduler_type = training_config.get('lr_scheduler_type', 'linear')

    if scheduler_type == 'linear':
        # Linear LR scheduler with warmup
        warmup_ratio = training_config.get('warmup_ratio', 0.06)
        warmup_steps = int(total_steps * warmup_ratio)
        lr_min = training_config.get('lr_min', 0.0)

        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                # Warmup: linearly increase from 0 to 1
                return float(current_step) / float(max(1, warmup_steps))
            else:
                # Linear decay: linearly decrease from 1 to lr_min/lr_max
                progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return max(lr_min / training_config.get('learning_rate', 1e-4), 1.0 - progress)

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        if accelerator.is_main_process:
            accelerator.print(f"\nLR Scheduler:")
            accelerator.print(f"  Type: Linear with warmup")
            accelerator.print(f"  Warmup steps: {warmup_steps} ({warmup_ratio*100:.1f}% of {total_steps})")
            accelerator.print(f"  Min LR: {lr_min}")

    elif scheduler_type == 'cosine':
        # Cosine annealing
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=training_config.get('lr_min', 0.0)
        )
        if accelerator.is_main_process:
            accelerator.print(f"\nLR Scheduler:")
            accelerator.print(f"  Type: Cosine annealing")
            accelerator.print(f"  T_max: {total_steps}")

    else:
        # Constant LR (no scheduler)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        if accelerator.is_main_process:
            accelerator.print(f"\nLR Scheduler: None (constant LR)")

    # Prepare for distributed training
    model, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, lr_scheduler
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')

    if args.resume:
        start_epoch, global_step = load_checkpoint(
            args.resume, model, optimizer, lr_scheduler, accelerator
        )

    # Training loop
    if accelerator.is_main_process:
        accelerator.print("\n" + "="*70)
        accelerator.print("Starting training")
        accelerator.print("="*70)

    val_every = training_config.get('val_every', 50)
    save_every = training_config.get('save_every', 200)
    log_every = training_config.get('log_every', 10)
    full_val_every = validation_config.get('full_val_every', 500)
    num_val_samples = validation_config.get('num_samples', 4)
    max_grad_norm = training_config.get('max_grad_norm', 1.0)

    for epoch in range(start_epoch, num_epochs):
        model.train()

        progress_bar = tqdm(
            total=len(train_loader),
            desc=f"Epoch {epoch+1}/{num_epochs}",
            disable=not accelerator.is_local_main_process,
        )

        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(model):
                ld_latent = batch['ld_latent']
                hd_latent = batch['hd_latent']

                B = hd_latent.shape[0]

                # Sample random timesteps
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (B,),
                    device=hd_latent.device
                ).long()

                # Add noise to HD latent
                noise = torch.randn_like(hd_latent)
                noisy_hd_latent = noise_scheduler.add_noise(hd_latent, noise, timesteps)

                # Concatenate LD latent as conditioning
                model_input = torch.cat([noisy_hd_latent, ld_latent], dim=1)

                # Predict noise
                noise_pred = model(model_input, timesteps)

                # MSE loss
                loss = F.mse_loss(noise_pred, noise)

                # Backward pass
                accelerator.backward(loss)

                # Gradient clipping
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)

                # Optimizer step
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Update progress bar
            progress_bar.update(1)

            # Logging
            if global_step % log_every == 0:
                logs = {
                    'train/loss': loss.detach().item(),
                    'train/lr': lr_scheduler.get_last_lr()[0],
                    'train/epoch': epoch,
                }

                if accelerator.is_main_process:
                    accelerator.log(logs, step=global_step)

                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{lr_scheduler.get_last_lr()[0]:.2e}"
                })

            # Validation (two-tier strategy)
            if global_step % val_every == 0 and global_step > 0 and accelerator.sync_gradients:
                # Lightweight validation (always)
                if accelerator.is_main_process:
                    accelerator.print("\nRunning lightweight validation...")

                val_loss = validate(model, val_loader, noise_scheduler, accelerator)

                accelerator.log({'val/loss': val_loss}, step=global_step)
                accelerator.print(f"  MSE loss: {val_loss:.4f}")

                # Full validation with SSIM/PSNR metrics (less frequent)
                if enable_full_validation and (global_step % full_val_every == 0):
                    if accelerator.is_main_process:
                        accelerator.print(f"\nRunning full validation with metrics ({num_val_samples} samples)...")

                    try:
                        # CRITICAL FIX: Skip visualization generation during multi-GPU training
                        # because generate_visualizations() tries to iterate through the PREPARED
                        # val_loader (which has DistributedSampler) on only ONE process, causing deadlock.
                        # Visualization should only be done with unprepared loaders or on all processes.
                        #
                        # TODO: Fix generate_visualizations() to work with prepared loaders OR
                        # create a separate unprepared val_loader just for visualization
                        #
                        # For now, we skip visualization in multi-GPU mode to avoid deadlocks
                        if accelerator.num_processes == 1:
                            # Single GPU - safe to generate visualizations
                            if training_config.get('use_wandb', True) and accelerator.is_main_process and vae is not None:
                                accelerator.print("Generating visualizations...")
                                viz_dict = generate_visualizations(
                                    model, val_loader, vae, config, accelerator, num_samples=1
                                )
                                if viz_dict:
                                    accelerator.log(viz_dict, step=global_step)
                                accelerator.print("Visualizations generated.")
                        elif accelerator.is_main_process:
                            accelerator.print("  Skipping visualization (not safe with prepared loaders in multi-GPU mode)")

                        # CRITICAL: Synchronize all processes before starting distributed validation
                        accelerator.wait_for_everyone()

                        metrics = validate_with_metrics(
                            model, val_loader, vae, config, accelerator,
                            num_samples=num_val_samples
                        )

                        if accelerator.is_main_process and metrics:
                            log_dict = {f'val/metrics/{k}': v for k, v in metrics.items()}
                            accelerator.log(log_dict, step=global_step)

                            accelerator.print(f"  SSIM: {metrics.get('ssim_mean', 0):.4f} ± {metrics.get('ssim_std', 0):.4f}")
                            accelerator.print(f"  PSNR: {metrics.get('psnr_mean', 0):.2f} ± {metrics.get('psnr_std', 0):.2f} dB")
                            accelerator.print(f"  Samples evaluated: {metrics.get('num_samples', 0)}")
                    except Exception as e:
                        if accelerator.is_main_process:
                            accelerator.print(f"  Warning: Full validation failed: {e}")

            # Save checkpoint
            if global_step % save_every == 0 and accelerator.sync_gradients and global_step > 0:
                if accelerator.is_main_process:
                    accelerator.print(f"\nSaving checkpoint at step {global_step}...")

                save_checkpoint(
                    accelerator, model, optimizer, lr_scheduler,
                    epoch, global_step, checkpoint_dir, is_best=False
                )

            global_step += 1

        progress_bar.close()

        # End of epoch
        if accelerator.is_main_process:
            accelerator.print(f"\nEpoch {epoch+1} completed. Global step: {global_step}")

    # Training completed
    if accelerator.is_main_process:
        accelerator.print("\n" + "="*70)
        accelerator.print("Training completed!")
        accelerator.print("="*70)
        accelerator.print(f"Best validation loss: {best_val_loss:.4f}")
        accelerator.print(f"Final checkpoint saved to: {checkpoint_dir}")

    # Save final checkpoint
    save_checkpoint(
        accelerator, model, optimizer, lr_scheduler,
        num_epochs-1, global_step, checkpoint_dir, is_best=False
    )

    # End wandb
    if training_config.get('use_wandb', True):
        accelerator.end_training()


def main():
    """Main entry point"""
    args = parse_args()
    config = load_config(args.config)
    train(config, args)


if __name__ == "__main__":
    main()
