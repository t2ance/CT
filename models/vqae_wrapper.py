"""
Frozen VQ-AE wrapper for latent diffusion

Wraps the pretrained 3D-MedDiffusion VQ-AE model for encoding/decoding
CT volumes to/from latent space.
"""

import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn

# Add vendored 3D-MedDiffusion subtree to module path
vendored_repo = Path(__file__).resolve().parents[1] / 'third_party' / '3d_meddiffusion'
vendored_repo_str = str(vendored_repo)
if vendored_repo_str not in sys.path:
    sys.path.insert(0, vendored_repo_str)

from AutoEncoder.model.PatchVolume import patchvolumeAE


class FrozenVQAE(nn.Module):
    """
    Frozen VQ-AE for encoding/decoding CT volumes.

    This wrapper provides a clean interface to the pretrained VQ-AE from
    3D-MedDiffusion. All parameters are frozen and the model is used only
    for encoding/decoding during latent diffusion training.

    Architecture:
        - Encoder: 3D CNN → latent space
        - VQ codebook: 8192 vectors × 8 dimensions
        - Decoder: latent space → 3D CNN
        - Compression: 8× spatial (per dimension)
        - Latent channels: 8

    Input/Output shapes (for D=200, H=128, W=128):
        - encode: [B, 1, 200, 128, 128] → [B, 8, 25, 16, 16]
        - decode: [B, 8, 25, 16, 16] → [B, 1, 200, 128, 128]

    Args:
        checkpoint_path: Path to pretrained VQ-AE checkpoint (.ckpt)
        device: Device to load model on ('cuda', 'cpu', or torch.device)

    Example:
        >>> vae = FrozenVQAE('PatchVolume_8x_s2.ckpt', device='cuda')
        >>> # Encode CT volume
        >>> ct = torch.randn(1, 1, 200, 128, 128).cuda()
        >>> z = vae.encode(ct)  # [1, 8, 25, 16, 16]
        >>> # Decode back
        >>> ct_recon = vae.decode(z)  # [1, 1, 200, 128, 128]
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[torch.device] = None
    ):
        super().__init__()

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        # Load pretrained model
        print(f"Loading VQ-AE from {checkpoint_path}")
        self.model = patchvolumeAE.load_from_checkpoint(
            checkpoint_path,
            map_location=device
        )
        self.model.eval()
        self.model.to(device)

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Store model info
        self.latent_channels = 8  # Cℓ
        self.compression_factor = 8  # Spatial compression per dimension
        self.codebook_size = 8192

        print(f"✓ VQ-AE loaded successfully")
        print(f"  Device: {device}")
        print(f"  Latent channels: {self.latent_channels}")
        print(f"  Compression: {self.compression_factor}× spatial")

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode CT volume to latent embeddings.

        Args:
            x: Input CT volume [B, 1, D, H, W] normalized to [-1, 1]

        Returns:
            z_q: Quantized latent embeddings [B, Cℓ, D', H', W']
                 where D'=D/8, H'=H/8, W'=W/8

        Note:
            Input dimensions should be divisible by 8 for perfect reconstruction.
            Non-divisible dimensions will be slightly truncated during decode.
        """
        if x.dim() != 5:
            raise ValueError(f"Expected 5D input [B,C,D,H,W], got {x.shape}")
        if x.shape[1] != 1:
            raise ValueError(f"Expected 1 channel, got {x.shape[1]}")

        # Ensure on correct device
        x = x.to(self.device)

        # Encode to latent (get embeddings, not indices)
        z_q, _ = self.model.encode(x, include_embeddings=True, quantize=True)

        return z_q

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent embeddings to CT volume.

        Args:
            z: Latent embeddings [B, Cℓ, D', H', W']

        Returns:
            x_recon: Reconstructed CT volume [B, 1, D, H, W]
                     where D=D'*8, H=H'*8, W=W'*8 (approximately)

        Note:
            Output may differ by 1 slice/pixel from original input due to
            compression artifacts.
        """
        if z.dim() != 5:
            raise ValueError(f"Expected 5D latent [B,C,D,H,W], got {z.shape}")
        if z.shape[1] != self.latent_channels:
            raise ValueError(
                f"Expected {self.latent_channels} channels, got {z.shape[1]}"
            )

        # Ensure on correct device
        z = z.to(self.device)

        # Decode to image space
        x_recon = self.model.decode(z)

        return x_recon

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode then decode (full reconstruction).

        Args:
            x: Input CT volume [B, 1, D, H, W]

        Returns:
            x_recon: Reconstructed CT volume [B, 1, D, H, W]
        """
        z = self.encode(x)
        return self.decode(z)

    def get_latent_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Calculate latent shape from input shape.

        Args:
            input_shape: Input shape (B, C, D, H, W)

        Returns:
            latent_shape: Latent shape (B, Cℓ, D', H', W')

        Example:
            >>> vae.get_latent_shape((1, 1, 200, 128, 128))
            (1, 8, 25, 16, 16)
        """
        if len(input_shape) != 5:
            raise ValueError("Expected shape (B, C, D, H, W)")

        B, C, D, H, W = input_shape

        # Calculate compressed dimensions
        D_latent = D // self.compression_factor
        H_latent = H // self.compression_factor
        W_latent = W // self.compression_factor

        return (B, self.latent_channels, D_latent, H_latent, W_latent)

    def normalize_latent(
        self,
        z: torch.Tensor,
        target_range: Tuple[float, float] = (-1.0, 1.0)
    ) -> torch.Tensor:
        """
        Normalize latent values to target range.

        VQ-AE latents have approximate range [-35, +37]. This function
        normalizes them to a standard range for diffusion training.

        Args:
            z: Latent embeddings [B, Cℓ, D', H', W']
            target_range: Target (min, max) range, default (-1, 1)

        Returns:
            z_norm: Normalized latents

        Note:
            Store normalization parameters if you need to denormalize later.
        """
        z_min = z.min()
        z_max = z.max()

        # Normalize to [0, 1]
        z_norm = (z - z_min) / (z_max - z_min + 1e-8)

        # Scale to target range
        target_min, target_max = target_range
        z_norm = z_norm * (target_max - target_min) + target_min

        return z_norm

    def __repr__(self) -> str:
        return (
            f"FrozenVQAE(\n"
            f"  device={self.device},\n"
            f"  latent_channels={self.latent_channels},\n"
            f"  compression={self.compression_factor}x,\n"
            f"  codebook_size={self.codebook_size}\n"
            f")"
        )


if __name__ == "__main__":
    # Quick test
    checkpoint_path = "~/projects/BioAgent/3D-MedDiffusion/checkpoints/3DMedDiffusion_checkpoints/PatchVolume_8x_s2.ckpt"

    print("="*70)
    print("Testing FrozenVQAE")
    print("="*70)

    # Initialize
    vae = FrozenVQAE(checkpoint_path, device='cuda')
    print(f"\n{vae}\n")

    # Test shapes
    test_cases = [
        (1, 1, 200, 128, 128),
        (2, 1, 200, 128, 128),
        (1, 1, 40, 128, 128),
    ]

    for input_shape in test_cases:
        print(f"Testing shape: {input_shape}")

        # Create dummy input
        x = torch.randn(*input_shape).cuda() * 0.5

        # Encode
        z = vae.encode(x)
        print(f"  Encoded: {input_shape} → {tuple(z.shape)}")

        # Expected shape
        expected = vae.get_latent_shape(input_shape)
        assert tuple(z.shape) == expected, f"Shape mismatch: {z.shape} vs {expected}"

        # Decode
        x_recon = vae.decode(z)
        print(f"  Decoded: {tuple(z.shape)} → {tuple(x_recon.shape)}")

        # Check reconstruction
        print(f"  Reconstruction error: {torch.abs(x_recon - x).mean().item():.4f}\n")

    print("="*70)
    print("✓ All tests passed!")
    print("="*70)
