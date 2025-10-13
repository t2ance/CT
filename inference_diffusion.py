"""
Inference script for CT Super-Resolution using Latent Diffusion Models

This script runs inference with a trained diffusion model to convert
low-dose CT scans to high-dose CT scans.

Features:
- Automatic checkpoint validation and fallback
- DDIM sampling for fast inference (50-100 steps)
- Batch processing of CT volumes
- Visualization output with comparisons
- Wandb logging support

Usage:
    python inference_diffusion.py \
        --config config_diffusion.yaml \
        --checkpoint outputs/checkpoints/best.pth \
        --input_dir data/ct_scans/low_dose \
        --output_dir outputs/inference

Requirements:
- Pre-trained diffusion model checkpoint
- Pre-trained VQ-AE checkpoint (for encoding/decoding)
- Low-dose CT scans in NIfTI format (.nii.gz)
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Dict
import yaml

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import numpy as np
import nibabel as nib

# Diffusion components
from diffusers import DDIMScheduler

# Custom modules
from model_factory import build_diffusion_model, DiffusionUNet3D
from models.vqae_wrapper import FrozenVQAE
from utils.visualization import CTVisualization
from utils.metrics import DistributedMetricsCalculator

# Optional wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Visualizations will not be logged to wandb.")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run inference with trained diffusion model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pth file)"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Directory containing low-dose CT scans (.nii.gz files)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save inference results"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to process (default: all)"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=None,
        help="Number of DDIM sampling steps (default: from config)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Log visualizations to wandb"
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def validate_checkpoint(checkpoint_path: Path) -> bool:
    """
    Validate checkpoint file is not corrupted.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        valid: True if checkpoint is valid, False otherwise
    """
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return False

    # Check file size (should be >100 MB for diffusion model)
    size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
    if size_mb < 100:
        print(f"Checkpoint file too small ({size_mb:.1f} MB), likely corrupted")
        return False

    try:
        # Try loading checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Check required keys
        required_keys = ['model_state_dict', 'epoch', 'global_step']
        if not all(key in checkpoint for key in required_keys):
            print(f"Checkpoint missing required keys: {required_keys}")
            return False

        return True

    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return False


def find_best_checkpoint(checkpoint_dir: Path, requested: str) -> Optional[Path]:
    """
    Find best valid checkpoint with automatic fallback.

    Searches for valid checkpoints in this order:
    1. Requested checkpoint
    2. best.pth
    3. latest.pth
    4. Most recent checkpoint_epoch_*.pth

    Args:
        checkpoint_dir: Directory containing checkpoints
        requested: Requested checkpoint filename

    Returns:
        checkpoint_path: Path to valid checkpoint, or None if not found
    """
    requested_path = Path(requested)

    # Try requested checkpoint first
    if validate_checkpoint(requested_path):
        return requested_path

    print(f"Requested checkpoint invalid: {requested}")
    print("Searching for alternative checkpoints...")

    # Try best.pth
    best_path = checkpoint_dir / "best.pth"
    if validate_checkpoint(best_path):
        print(f"Using best checkpoint: {best_path}")
        return best_path

    # Try latest.pth
    latest_path = checkpoint_dir / "latest.pth"
    if validate_checkpoint(latest_path):
        print(f"Using latest checkpoint: {latest_path}")
        return latest_path

    # Try epoch checkpoints (most recent first)
    epoch_checkpoints = sorted(
        checkpoint_dir.glob("checkpoint_epoch_*.pth"),
        key=lambda p: int(p.stem.split('_')[-1]),
        reverse=True
    )

    for ckpt_path in epoch_checkpoints:
        if validate_checkpoint(ckpt_path):
            print(f"Using epoch checkpoint: {ckpt_path}")
            return ckpt_path

    print("No valid checkpoints found!")
    return None


def load_model(
    config: dict,
    checkpoint_path: Path,
    device: torch.device
) -> DiffusionUNet3D:
    """
    Load diffusion model from checkpoint.

    Args:
        config: Configuration dictionary
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        model: Loaded diffusion model
    """
    print("\nLoading diffusion model...")

    model_config = config['model']
    model = build_diffusion_model(model_config, verbose=True)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    # Move to device and set to eval mode
    model = model.to(device)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded: {num_params:,} parameters")
    print(f"  Checkpoint epoch: {checkpoint['epoch']}")
    print(f"  Checkpoint step: {checkpoint['global_step']}")

    return model


@torch.no_grad()
def run_inference(
    model: DiffusionUNet3D,
    ld_latent: torch.Tensor,
    scheduler: DDIMScheduler,
    device: torch.device,
) -> torch.Tensor:
    """
    Run diffusion inference to generate HD latent from LD latent.

    Args:
        model: Trained diffusion model
        ld_latent: Low-dose CT latent [B, C, D, H, W]
        scheduler: DDIM scheduler for sampling
        device: Device to run inference on

    Returns:
        hd_latent: Predicted high-dose CT latent [B, C, D, H, W]
    """
    # Start from pure noise
    hd_latent = torch.randn_like(ld_latent).to(device)
    ld_latent = ld_latent.to(device)

    # DDIM sampling loop
    for t in tqdm(scheduler.timesteps, desc="DDIM sampling", leave=False):
        # Concatenate conditioning
        model_input = torch.cat([hd_latent, ld_latent], dim=1)

        # Predict noise
        timestep = t.unsqueeze(0).to(device) if t.dim() == 0 else t.to(device)
        noise_pred = model(model_input, timestep)

        # DDIM step
        hd_latent = scheduler.step(noise_pred, t, hd_latent).prev_sample

    return hd_latent


def main():
    """Main inference function"""
    args = parse_args()
    config = load_config(args.config)

    print("="*70)
    print("CT Latent Diffusion Inference")
    print("="*70)

    # Validate and find checkpoint
    checkpoint_path = Path(args.checkpoint)
    checkpoint_dir = checkpoint_path.parent

    checkpoint_path = find_best_checkpoint(checkpoint_dir, args.checkpoint)
    if checkpoint_path is None:
        print("\nERROR: No valid checkpoint found!")
        sys.exit(1)

    print(f"\nUsing checkpoint: {checkpoint_path}")

    # Setup device
    device = torch.device(args.device)
    print(f"Device: {device}")

    # Load models
    model = load_model(config, checkpoint_path, device)

    print("\nLoading VQ-AE...")
    vae_checkpoint = config['data']['vae_checkpoint']
    vae = FrozenVQAE(vae_checkpoint, device=device)
    print(f"  VQ-AE loaded from: {vae_checkpoint}")

    # Create DDIM scheduler
    print("\nCreating DDIM scheduler...")
    scheduler_config = config['scheduler']
    inference_config = config.get('inference', {})

    num_inference_steps = args.num_inference_steps or inference_config.get('num_inference_steps', 100)

    ddim_scheduler = DDIMScheduler(
        num_train_timesteps=scheduler_config['num_train_timesteps'],
        beta_start=scheduler_config['beta_start'],
        beta_end=scheduler_config['beta_end'],
        beta_schedule=scheduler_config['beta_schedule'],
        prediction_type=scheduler_config['prediction_type'],
        clip_sample=scheduler_config.get('clip_sample', False),
    )
    ddim_scheduler.set_timesteps(num_inference_steps)
    print(f"  Inference steps: {num_inference_steps}")

    # Setup output directory
    output_dir = Path(args.output_dir or config['output']['visualization_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Initialize wandb if requested
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=config['training'].get('wandb_project', 'ct-latent-diffusion'),
            name=f"inference_{checkpoint_path.stem}",
            tags=['inference', 'ct', 'diffusion'],
            config=config,
        )
        print("\nWandB initialized")

    # Initialize visualization and metrics
    viz = CTVisualization()
    metrics_calc = DistributedMetricsCalculator(data_range=2.0, compute_slice_wise=True)

    # Load input data (placeholder - adapt to your data loading)
    print("\n" + "="*70)
    print("Running inference on validation set...")
    print("="*70)

    # TODO: Load your validation data here
    # This is a placeholder - replace with actual data loading
    from data.latent_dataset import LatentCTDataset
    dataset = LatentCTDataset(config['data']['latent_cache_dir'], split='val')

    num_samples = args.num_samples or len(dataset)
    num_samples = min(num_samples, len(dataset))

    print(f"Processing {num_samples} samples...")

    for idx in range(num_samples):
        print(f"\nSample {idx+1}/{num_samples}")

        # Load latent
        sample = dataset[idx]
        ld_latent = sample['ld_latent'].unsqueeze(0)  # [1, C, D, H, W]
        hd_latent_gt = sample['hd_latent'].unsqueeze(0)

        # Run inference
        hd_latent_pred = run_inference(model, ld_latent, ddim_scheduler, device)

        # Decode to CT space
        print("  Decoding latents...")
        ld_ct = vae.decode(ld_latent)
        pred_ct = vae.decode(hd_latent_pred)
        gt_ct = vae.decode(hd_latent_gt)

        # Compute metrics
        pred_np = pred_ct[0, 0].cpu().numpy()
        gt_np = gt_ct[0, 0].cpu().numpy()

        metrics = metrics_calc.compute_metrics(pred_np, gt_np)
        ssim = metrics['ssim']
        psnr = metrics['psnr']

        print(f"  SSIM: {ssim:.4f}")
        print(f"  PSNR: {psnr:.2f} dB")

        # Create visualization
        print("  Creating visualization...")
        sample_viz = viz.create_multi_slice_comparison(
            ld=ld_ct,
            pred=pred_ct,
            gt=gt_ct,
            num_slices=5
        )

        # Save visualization
        viz_path = output_dir / f"sample_{idx:03d}.png"
        sample_viz.save(viz_path)
        print(f"  Saved: {viz_path}")

        # Log to wandb
        if args.use_wandb and WANDB_AVAILABLE:
            wandb.log({
                f'inference/sample_{idx}': wandb.Image(sample_viz),
                f'inference/ssim_{idx}': ssim,
                f'inference/psnr_{idx}': psnr,
            })

        # Save predicted CT volume
        pred_nifti_path = output_dir / f"sample_{idx:03d}_pred.nii.gz"
        pred_nifti = nib.Nifti1Image(pred_np, affine=np.eye(4))
        nib.save(pred_nifti, pred_nifti_path)

    print("\n" + "="*70)
    print("Inference completed!")
    print(f"Results saved to: {output_dir}")
    print("="*70)

    if args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()


if __name__ == "__main__":
    main()
