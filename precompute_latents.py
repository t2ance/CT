"""
Precompute VQ-AE latent representations for CT scans.

This script encodes raw CT scans (LDCT and HDCT pairs) into latent space
using a pre-trained VQ-AE model. The latents are cached to disk for fast
training of the diffusion model.

Directory structure expected:
    INPUT (data_dir):
        low_dose/
            sample_000.nii.gz
            sample_001.nii.gz
            ...
        high_dose/
            sample_000.nii.gz
            sample_001.nii.gz
            ...

    OUTPUT (latent_cache_dir):
        train/
            sample_000.pt  (contains {'ld_latent': ..., 'hd_latent': ...})
            sample_001.pt
            ...
        val/
            sample_000.pt
            ...

Usage:
    python precompute_latents.py \
        --data_dir data/ct_scans \
        --latent_cache_dir latents_cache \
        --vae_checkpoint path/to/vqae.ckpt \
        --train_split 0.8 \
        --device cuda
"""

import argparse
from pathlib import Path
from typing import Tuple, List
import yaml

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import nibabel as nib

from models.vqae_wrapper import FrozenVQAE


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Precompute VQ-AE latents for CT scans")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing low_dose/ and high_dose/ subdirectories"
    )
    parser.add_argument(
        "--latent_cache_dir",
        type=str,
        default="./latents_cache",
        help="Output directory for cached latents"
    )
    parser.add_argument(
        "--vae_checkpoint",
        type=str,
        default=None,
        help="Path to VQ-AE checkpoint (default: from config)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config_diffusion.yaml",
        help="Path to config file (for VQ-AE checkpoint path)"
    )
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.8,
        help="Fraction of data to use for training (rest is validation)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for encoding (usually 1 for 3D CT volumes)"
    )
    parser.add_argument(
        "--target_shape",
        type=int,
        nargs=3,
        default=[200, 128, 128],
        help="Target CT shape (D H W) for resizing"
    )
    return parser.parse_args()


def load_ct_scan(ct_path: Path, target_shape: Tuple[int, int, int] = None) -> torch.Tensor:
    """
    Load and preprocess CT scan from NIfTI file.

    Args:
        ct_path: Path to .nii.gz file
        target_shape: Target shape (D, H, W) for resizing

    Returns:
        ct_tensor: Preprocessed CT scan [1, 1, D, H, W]
    """
    # Load NIfTI file
    nifti = nib.load(str(ct_path))
    ct_array = nifti.get_fdata()

    # Convert to tensor
    ct_tensor = torch.from_numpy(ct_array).float()

    # Add batch and channel dimensions [B, C, D, H, W]
    ct_tensor = ct_tensor.unsqueeze(0).unsqueeze(0)

    # Resize if needed
    if target_shape is not None:
        current_shape = ct_tensor.shape[2:]
        if current_shape != tuple(target_shape):
            ct_tensor = F.interpolate(
                ct_tensor,
                size=target_shape,
                mode='trilinear',
                align_corners=False
            )

    return ct_tensor


def find_ct_pairs(data_dir: Path) -> List[Tuple[Path, Path]]:
    """
    Find matched pairs of low-dose and high-dose CT scans.

    Args:
        data_dir: Directory containing low_dose/ and high_dose/ subdirectories

    Returns:
        pairs: List of (ld_path, hd_path) tuples
    """
    ld_dir = data_dir / "low_dose"
    hd_dir = data_dir / "high_dose"

    if not ld_dir.exists():
        raise ValueError(f"Low-dose directory not found: {ld_dir}")
    if not hd_dir.exists():
        raise ValueError(f"High-dose directory not found: {hd_dir}")

    # Find all files
    ld_files = sorted(list(ld_dir.glob("*.nii.gz")))
    hd_files = sorted(list(hd_dir.glob("*.nii.gz")))

    if len(ld_files) == 0:
        raise ValueError(f"No .nii.gz files found in {ld_dir}")
    if len(hd_files) == 0:
        raise ValueError(f"No .nii.gz files found in {hd_dir}")

    # Match by filename
    ld_dict = {f.stem.replace('.nii', ''): f for f in ld_files}
    hd_dict = {f.stem.replace('.nii', ''): f for f in hd_files}

    common_names = set(ld_dict.keys()) & set(hd_dict.keys())

    if len(common_names) == 0:
        raise ValueError("No matching LD/HD pairs found!")

    pairs = [(ld_dict[name], hd_dict[name]) for name in sorted(common_names)]

    return pairs


def split_train_val(pairs: List[Tuple[Path, Path]], train_split: float) -> Tuple[List, List]:
    """
    Split pairs into train and validation sets.

    Args:
        pairs: List of (ld_path, hd_path) tuples
        train_split: Fraction for training

    Returns:
        train_pairs, val_pairs
    """
    num_train = int(len(pairs) * train_split)
    train_pairs = pairs[:num_train]
    val_pairs = pairs[num_train:]

    return train_pairs, val_pairs


@torch.no_grad()
def encode_and_save(
    pairs: List[Tuple[Path, Path]],
    vae: FrozenVQAE,
    output_dir: Path,
    target_shape: Tuple[int, int, int],
    device: torch.device,
):
    """
    Encode CT pairs to latent space and save to disk.

    Args:
        pairs: List of (ld_path, hd_path) tuples
        vae: Frozen VQ-AE model
        output_dir: Output directory for latents
        target_shape: Target CT shape (D, H, W)
        device: Device to use
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, (ld_path, hd_path) in enumerate(tqdm(pairs, desc=f"Encoding to {output_dir.name}")):
        try:
            # Load CT scans
            ld_ct = load_ct_scan(ld_path, target_shape).to(device)
            hd_ct = load_ct_scan(hd_path, target_shape).to(device)

            # Encode to latent space
            ld_latent = vae.encode(ld_ct)  # [1, C, D', H', W']
            hd_latent = vae.encode(hd_ct)

            # Remove batch dimension
            ld_latent = ld_latent.squeeze(0).cpu()  # [C, D', H', W']
            hd_latent = hd_latent.squeeze(0).cpu()

            # Save latents
            output_path = output_dir / f"sample_{idx:03d}.pt"
            torch.save({
                'ld_latent': ld_latent,
                'hd_latent': hd_latent,
                'ld_path': str(ld_path),
                'hd_path': str(hd_path),
            }, output_path)

        except Exception as e:
            print(f"\nError processing {ld_path.name}: {e}")
            continue


def main():
    """Main precomputation function"""
    args = parse_args()

    print("="*70)
    print("VQ-AE Latent Precomputation")
    print("="*70)

    # Load config if VQ-AE checkpoint not specified
    vae_checkpoint = args.vae_checkpoint
    if vae_checkpoint is None:
        print(f"\nLoading VQ-AE checkpoint path from config: {args.config}")
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        vae_checkpoint = config['data']['vae_checkpoint']

    print(f"\nConfiguration:")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Latent cache directory: {args.latent_cache_dir}")
    print(f"  VQ-AE checkpoint: {vae_checkpoint}")
    print(f"  Train split: {args.train_split:.1%}")
    print(f"  Device: {args.device}")
    print(f"  Target shape: {args.target_shape}")

    # Setup device
    device = torch.device(args.device)

    # Find CT pairs
    print(f"\nFinding CT pairs in {args.data_dir}...")
    data_dir = Path(args.data_dir)
    pairs = find_ct_pairs(data_dir)
    print(f"  Found {len(pairs)} CT pairs")

    # Split train/val
    train_pairs, val_pairs = split_train_val(pairs, args.train_split)
    print(f"  Train: {len(train_pairs)} pairs")
    print(f"  Val: {len(val_pairs)} pairs")

    # Load VQ-AE
    print(f"\nLoading VQ-AE from {vae_checkpoint}...")
    vae = FrozenVQAE(vae_checkpoint, device=device)
    print("  VQ-AE loaded successfully")

    # Test encoding one sample to get latent shape
    print("\nTesting encoding...")
    test_ct = load_ct_scan(train_pairs[0][0], tuple(args.target_shape)).to(device)
    test_latent = vae.encode(test_ct)
    print(f"  CT shape: {test_ct.shape}")
    print(f"  Latent shape: {test_latent.shape}")
    print(f"  Compression ratio: {test_ct.numel() / test_latent.numel():.1f}×")
    del test_ct, test_latent
    torch.cuda.empty_cache()

    # Create output directories
    latent_cache_dir = Path(args.latent_cache_dir)
    train_dir = latent_cache_dir / "train"
    val_dir = latent_cache_dir / "val"

    # Encode and save training set
    print(f"\n{'='*70}")
    print("Encoding training set...")
    print(f"{'='*70}")
    encode_and_save(train_pairs, vae, train_dir, tuple(args.target_shape), device)

    # Encode and save validation set
    print(f"\n{'='*70}")
    print("Encoding validation set...")
    print(f"{'='*70}")
    encode_and_save(val_pairs, vae, val_dir, tuple(args.target_shape), device)

    # Summary
    print("\n" + "="*70)
    print("Precomputation completed!")
    print("="*70)
    print(f"Output directory: {latent_cache_dir}")
    print(f"  Train samples: {len(list(train_dir.glob('*.pt')))}")
    print(f"  Val samples: {len(list(val_dir.glob('*.pt')))}")

    # Calculate total size
    train_size = sum(f.stat().st_size for f in train_dir.glob('*.pt')) / (1024**3)
    val_size = sum(f.stat().st_size for f in val_dir.glob('*.pt')) / (1024**3)
    print(f"\nTotal size:")
    print(f"  Train: {train_size:.2f} GB")
    print(f"  Val: {val_size:.2f} GB")
    print(f"  Total: {train_size + val_size:.2f} GB")

    print("\nYou can now train the diffusion model with:")
    print(f"  python train_diffusion.py --config {args.config}")


if __name__ == "__main__":
    main()
