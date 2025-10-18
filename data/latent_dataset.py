"""
Dataset for loading pre-computed VQ-AE latent representations.

This dataset loads latent embeddings that have been pre-encoded using
the VQ-AE model, enabling efficient training of the diffusion model.
"""

import os
import warnings
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset


class LatentCTDataset(Dataset):
    """
    Dataset for loading pre-computed VQ-AE latents.

    The dataset expects the following directory structure:
        latent_cache_dir/
            train/
                sample_000.pt
                sample_001.pt
                ...
            val/
                sample_000.pt
                sample_001.pt
                ...

    Each .pt file should contain a dict with:
        {
            'ld_latent': torch.Tensor,  # [C, D, H, W] Low-dose latent
            'hd_latent': torch.Tensor,  # [C, D, H, W] High-dose latent
        }

    Args:
        latent_dir: Directory containing train/ and val/ subdirectories
        split: 'train' or 'val'
        normalize_latents: If True, normalize latents to [-1, 1]
                          (Usually False, as latents are ~[-30, 30])
    """

    def __init__(
        self,
        latent_dir: str,
        split: str = 'train',
        normalize_latents: bool = False
    ):
        self.latent_dir = Path(latent_dir)
        self.split = split
        self.normalize_latents = normalize_latents

        # Find all latent files
        split_dir = self.latent_dir / split
        if not split_dir.exists():
            raise ValueError(f"Split directory not found: {split_dir}")

        self.latent_files = sorted(list(split_dir.glob('*.pt')))
        if len(self.latent_files) == 0:
            raise ValueError(f"No .pt files found in {split_dir}")

        print(f"Loaded {len(self.latent_files)} {split} samples from {latent_dir}")

    def __len__(self) -> int:
        return len(self.latent_files)

    def __getitem__(self, idx: int) -> dict:
        """
        Load a single latent sample.

        Returns:
            dict with keys:
                'ld_latent': torch.Tensor [C, D, H, W]
                'hd_latent': torch.Tensor [C, D, H, W]
        """
        latent_path = self.latent_files[idx]

        # Load latents
        try:
            data = torch.load(latent_path, map_location='cpu')
        except Exception as e:
            print(f"Error loading latent file {latent_path}: {e}")
            raise e

        ld_latent = data['ld_latent']
        hd_latent = data['hd_latent']

        # Normalize if requested
        if self.normalize_latents:
            ld_latent = self._normalize(ld_latent)
            hd_latent = self._normalize(hd_latent)

        return {
            'ld_latent': ld_latent,
            'hd_latent': hd_latent,
        }

    def _normalize(self, latent: torch.Tensor) -> torch.Tensor:
        """Normalize latent to [-1, 1] range"""
        latent_min = latent.min()
        latent_max = latent.max()
        latent_norm = (latent - latent_min) / (latent_max - latent_min + 1e-8)
        latent_norm = latent_norm * 2 - 1  # [0, 1] → [-1, 1]
        return latent_norm


def create_latent_dataloaders(
    latent_dir: str,
    batch_size: Optional[int] = None,
    train_batch_size: Optional[int] = None,
    eval_batch_size: Optional[int] = None,
    num_workers: int = 4,
    normalize_latents: bool = False,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders for latent representations.

    Args:
        latent_dir: Directory containing train/ and val/ subdirectories
        batch_size: (Deprecated) Batch size for both loaders
        train_batch_size: Batch size for training loader
        eval_batch_size: Batch size for validation loader
        num_workers: Number of dataloader workers
        normalize_latents: Whether to normalize latents to [-1, 1]
        pin_memory: Whether to pin memory for faster GPU transfer

    Returns:
        train_loader, val_loader
    """
    if train_batch_size is None and eval_batch_size is None:
        if batch_size is not None:
            warnings.warn(
                "create_latent_dataloaders(batch_size=...) is deprecated. "
                "Please use train_batch_size / eval_batch_size.",
                DeprecationWarning,
            )
            train_batch_size = batch_size
            eval_batch_size = batch_size
        else:
            train_batch_size = eval_batch_size = 4
    else:
        if train_batch_size is None and eval_batch_size is not None:
            train_batch_size = eval_batch_size
        if eval_batch_size is None and train_batch_size is not None:
            eval_batch_size = train_batch_size
        if batch_size is not None and (train_batch_size != batch_size or eval_batch_size != batch_size):
            warnings.warn(
                "Both batch_size and train/eval batch sizes provided; batch_size is ignored.",
                UserWarning,
            )

    # Create datasets
    train_dataset = LatentCTDataset(
        latent_dir=latent_dir,
        split='train',
        normalize_latents=normalize_latents
    )

    val_dataset = LatentCTDataset(
        latent_dir=latent_dir,
        split='val',
        normalize_latents=normalize_latents
    )

    print(f"Train dataset length: {len(train_dataset)}")
    print(f"Val dataset length: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False  # Drop last incomplete batch
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    return train_loader, val_loader


def create_train_subset_dataloader(
    latent_dir: str,
    num_samples: int = 4,
    batch_size: int = 1,
    num_workers: int = 4,
    normalize_latents: bool = False,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Create a dataloader for a subset of the training set (for validation metrics).

    Args:
        latent_dir: Directory containing train/ and val/ subdirectories
        num_samples: Number of training samples to include in subset
        batch_size: Batch size for the dataloader
        num_workers: Number of dataloader workers
        normalize_latents: Whether to normalize latents to [-1, 1]
        pin_memory: Whether to pin memory for faster GPU transfer

    Returns:
        train_subset_loader: DataLoader with a subset of training samples
    """
    # Create full training dataset
    train_dataset = LatentCTDataset(
        latent_dir=latent_dir,
        split='train',
        normalize_latents=normalize_latents
    )

    # Create subset indices
    num_samples = min(num_samples, len(train_dataset))
    indices = list(range(num_samples))
    train_subset = Subset(train_dataset, indices)

    # Create dataloader
    train_subset_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    print(f"Train subset dataloader created with {num_samples} samples")
    return train_subset_loader


if __name__ == "__main__":
    # Quick test
    print("="*70)
    print("Testing LatentCTDataset")
    print("="*70)

    # Create dataset
    latent_dir = './latents_cache'
    if not os.path.exists(latent_dir):
        print(f"Error: {latent_dir} not found")
        print("Please run VQ-AE encoding first to generate latents")
        exit(1)

    # Create dataloaders
    train_loader, val_loader = create_latent_dataloaders(
        latent_dir=latent_dir,
        batch_size=2,
        num_workers=0,
        normalize_latents=False,
    )

    # Test loading a batch
    print("\n1. Testing train loader:")
    for batch in train_loader:
        ld_latent = batch['ld_latent']
        hd_latent = batch['hd_latent']

        print(f"   LD latent shape: {ld_latent.shape}")
        print(f"   HD latent shape: {hd_latent.shape}")
        print(f"   LD latent range: [{ld_latent.min():.2f}, {ld_latent.max():.2f}]")
        print(f"   HD latent range: [{hd_latent.min():.2f}, {hd_latent.max():.2f}]")
        break

    print("\n2. Testing val loader:")
    for batch in val_loader:
        ld_latent = batch['ld_latent']
        hd_latent = batch['hd_latent']

        print(f"   LD latent shape: {ld_latent.shape}")
        print(f"   HD latent shape: {hd_latent.shape}")
        break

    print("\n" + "="*70)
    print("✓ Tests passed!")
    print("="*70)
