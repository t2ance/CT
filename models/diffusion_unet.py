"""
3D UNet for Diffusion Models in Latent Space

This module implements a 3D UNet architecture for denoising diffusion models.
The network takes noisy latent representations and timesteps as input, and
predicts the noise component.

Architecture:
- Multi-scale encoder-decoder with skip connections
- Time conditioning via sinusoidal embeddings
- Self-attention at specified resolution levels
- Conditioning via concatenation (LDCT latent + noisy HDCT latent)

Reference implementations:
- diffusers.models.UNet2DConditionModel
- https://github.com/lucidrains/denoising-diffusion-pytorch
- https://github.com/CompVis/latent-diffusion
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Time Embedding Module
# ============================================================================

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal positional embeddings for timesteps.

    Converts scalar timesteps to dense embeddings using sine/cosine functions
    at different frequencies (similar to Transformer positional encodings).
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time: Timesteps [B] or [B, 1]

        Returns:
            embeddings: [B, dim]
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings


class TimeEmbedding(nn.Module):
    """
    Time embedding module that converts timesteps to feature vectors.

    Process: timestep -> sinusoidal encoding -> MLP -> time features
    """

    def __init__(self, time_embed_dim: int):
        super().__init__()
        self.time_embed_dim = time_embed_dim

        # Sinusoidal encoding
        self.time_proj = SinusoidalPositionEmbeddings(time_embed_dim)

        # MLP to process time embeddings
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 4, time_embed_dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: [B] or [B, 1]

        Returns:
            time_emb: [B, time_embed_dim]
        """
        if timesteps.dim() == 2:
            timesteps = timesteps.squeeze(-1)

        time_emb = self.time_proj(timesteps)
        time_emb = self.time_mlp(time_emb)
        return time_emb


# ============================================================================
# Basic Building Blocks
# ============================================================================

class GroupNorm3D(nn.GroupNorm):
    """
    3D Group Normalization with automatic group count.
    Uses 32 groups by default (following standard diffusion models).
    """

    def __init__(self, num_channels: int, num_groups: int = 32, eps: float = 1e-6):
        # Adjust num_groups if num_channels is too small
        num_groups = min(num_groups, num_channels)
        super().__init__(num_groups, num_channels, eps=eps)


class ResBlock3D(nn.Module):
    """
    3D Residual block with time conditioning.

    Architecture:
        x -> [GroupNorm -> SiLU -> Conv3D] -> + time_emb ->
             [GroupNorm -> SiLU -> Dropout -> Conv3D] -> + skip -> out

    Args:
        in_channels: Input channels
        out_channels: Output channels (if None, same as in_channels)
        time_embed_dim: Dimension of time embeddings
        dropout: Dropout probability
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        time_embed_dim: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        out_channels = out_channels or in_channels

        # First conv block
        self.norm1 = GroupNorm3D(in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)

        # Time embedding projection
        self.time_emb_proj = nn.Linear(time_embed_dim, out_channels)

        # Second conv block
        self.norm2 = GroupNorm3D(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)

        # Skip connection
        if in_channels != out_channels:
            self.skip_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip_conv = nn.Identity()

        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, D, H, W]
            time_emb: [B, time_embed_dim]

        Returns:
            out: [B, C_out, D, H, W]
        """
        h = x

        # First conv block
        h = self.norm1(h)
        h = self.act(h)
        h = self.conv1(h)

        # Add time embedding (broadcast to spatial dimensions)
        time_emb = self.time_emb_proj(time_emb)
        time_emb = time_emb[:, :, None, None, None]  # [B, C, 1, 1, 1]
        h = h + time_emb

        # Second conv block
        h = self.norm2(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)

        # Skip connection
        skip = self.skip_conv(x)

        return h + skip


class AttentionBlock3D(nn.Module):
    """
    3D Self-attention block for diffusion models.

    Computes attention over spatial dimensions (D*H*W flattened to sequence).
    Uses multi-head attention for better feature learning.

    Args:
        channels: Number of channels
        num_heads: Number of attention heads (must divide channels evenly)
    """

    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

        assert channels % num_heads == 0, f"channels {channels} must be divisible by num_heads {num_heads}"

        self.norm = GroupNorm3D(channels)

        # QKV projection
        self.qkv = nn.Conv3d(channels, channels * 3, kernel_size=1)

        # Output projection
        self.proj_out = nn.Conv3d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, D, H, W]

        Returns:
            out: [B, C, D, H, W]
        """
        B, C, D, H, W = x.shape

        # Normalize
        h = self.norm(x)

        # QKV projection
        qkv = self.qkv(h)  # [B, C*3, D, H, W]

        # Reshape to sequence: [B, C*3, D*H*W] -> [B, D*H*W, C*3]
        qkv = qkv.reshape(B, C * 3, D * H * W).permute(0, 2, 1)

        # Split into Q, K, V
        q, k, v = qkv.chunk(3, dim=-1)  # Each: [B, D*H*W, C]

        # Reshape for multi-head attention: [B, num_heads, D*H*W, C//num_heads]
        head_dim = C // self.num_heads
        q = q.reshape(B, D * H * W, self.num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, D * H * W, self.num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, D * H * W, self.num_heads, head_dim).permute(0, 2, 1, 3)

        # Attention: softmax(Q @ K^T / sqrt(d)) @ V
        scale = head_dim ** -0.5
        attn = torch.softmax(q @ k.transpose(-2, -1) * scale, dim=-1)  # [B, num_heads, D*H*W, D*H*W]
        h = attn @ v  # [B, num_heads, D*H*W, head_dim]

        # Reshape back: [B, num_heads, D*H*W, head_dim] -> [B, C, D*H*W] -> [B, C, D, H, W]
        h = h.permute(0, 2, 1, 3).reshape(B, D * H * W, C)
        h = h.permute(0, 2, 1).reshape(B, C, D, H, W)

        # Output projection
        h = self.proj_out(h)

        # Skip connection
        return x + h


class Downsample3D(nn.Module):
    """
    3D downsampling layer (2× spatial downsampling).
    Uses strided convolution for learnable downsampling.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, D, H, W]

        Returns:
            out: [B, C, D/2, H/2, W/2]
        """
        return self.conv(x)


class Upsample3D(nn.Module):
    """
    3D upsampling layer (2× spatial upsampling).
    Uses nearest neighbor interpolation followed by convolution.
    This ensures spatial dimensions match properly with Downsample3D.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, D, H, W]

        Returns:
            out: [B, C, D*2, H*2, W*2]
        """
        # Upsample using nearest neighbor (2x)
        h = F.interpolate(x, scale_factor=2, mode='nearest')
        # Refine with convolution
        h = self.conv(h)
        return h


# ============================================================================
# Main 3D UNet Architecture
# ============================================================================

class DiffusionUNet3D(nn.Module):
    """
    3D UNet for Diffusion Models in Latent Space.

    This is a time-conditioned 3D UNet that predicts noise in diffusion models.
    It uses a standard encoder-decoder architecture with skip connections,
    ResNet blocks, and self-attention at specified resolution levels.

    Architecture:
        - Encoder: Progressive downsampling with ResBlocks and optional attention
        - Bottleneck: ResBlocks with attention at lowest resolution
        - Decoder: Progressive upsampling with ResBlocks, attention, and skip connections

    Args:
        in_channels: Input channels (typically 8*2=16 for LD+HD latent concatenation)
        out_channels: Output channels (typically 8 for predicted noise)
        model_channels: Base channel count (scaled by channel_mult at each level)
        channel_mult: Channel multipliers for each resolution level (e.g., [1, 2, 4, 8])
        num_blocks: Number of ResNet blocks per resolution level
        attention_levels: Which levels get self-attention (e.g., [1, 2, 3])
        time_embed_dim: Dimension of time embeddings
        dropout: Dropout probability
        num_heads: Number of attention heads

    Example:
        >>> model = DiffusionUNet3D(
        ...     in_channels=16,  # LD + HD latent
        ...     out_channels=8,  # Predicted noise
        ...     model_channels=128,
        ...     channel_mult=[1, 2, 4, 8],
        ...     num_blocks=3,
        ...     attention_levels=[1, 2, 3],
        ...     time_embed_dim=512,
        ...     dropout=0.1,
        ...     num_heads=8,
        ... )
        >>> x = torch.randn(2, 16, 25, 16, 16)  # [B, C, D, H, W]
        >>> t = torch.tensor([100, 200])
        >>> noise_pred = model(x, t)  # [2, 8, 25, 16, 16]
    """

    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 8,
        model_channels: int = 128,
        channel_mult: List[int] = [1, 2, 4, 8],
        num_blocks: int = 3,
        attention_levels: List[int] = [1, 2, 3],
        time_embed_dim: int = 512,
        dropout: float = 0.1,
        num_heads: int = 8,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_levels = len(channel_mult)
        self.num_blocks = num_blocks

        # Time embedding
        self.time_embed = TimeEmbedding(time_embed_dim)

        # Initial convolution
        self.conv_in = nn.Conv3d(in_channels, model_channels, kernel_size=3, padding=1)

        # ====================================================================
        # Encoder (Downsampling path)
        # ====================================================================
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()

        ch = model_channels
        for level in range(self.num_levels):
            out_ch = model_channels * channel_mult[level]

            # Add ResBlocks for this level
            for _ in range(num_blocks):
                block_modules = nn.ModuleList([
                    ResBlock3D(ch, out_ch, time_embed_dim, dropout)
                ])

                # Add attention if specified for this level
                if level in attention_levels:
                    block_modules.append(AttentionBlock3D(out_ch, num_heads))

                self.down_blocks.append(block_modules)
                ch = out_ch

            # Downsample (except for last level)
            if level != self.num_levels - 1:
                self.down_samples.append(Downsample3D(ch))
            else:
                self.down_samples.append(nn.Identity())

        # ====================================================================
        # Bottleneck (Middle)
        # ====================================================================
        self.mid_block1 = ResBlock3D(ch, ch, time_embed_dim, dropout)
        self.mid_attn = AttentionBlock3D(ch, num_heads)
        self.mid_block2 = ResBlock3D(ch, ch, time_embed_dim, dropout)

        # ====================================================================
        # Decoder (Upsampling path)
        # ====================================================================
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()

        for level in reversed(range(self.num_levels)):
            out_ch = model_channels * channel_mult[level]

            # Add ResBlocks for this level (num_blocks + 1 for extra skip)
            for i in range(num_blocks + 1):
                # First block at each level gets skip connection
                if i == 0:
                    in_ch = ch + out_ch  # ch from previous + skip from encoder
                else:
                    in_ch = ch

                block_modules = nn.ModuleList([
                    ResBlock3D(in_ch, out_ch, time_embed_dim, dropout)
                ])

                # Add attention if specified for this level
                if level in attention_levels:
                    block_modules.append(AttentionBlock3D(out_ch, num_heads))

                self.up_blocks.append(block_modules)
                ch = out_ch

            # Upsample (except for first level in reversed order)
            if level != 0:
                self.up_samples.append(Upsample3D(ch))
            else:
                self.up_samples.append(nn.Identity())

        # ====================================================================
        # Output
        # ====================================================================
        self.norm_out = GroupNorm3D(ch)
        self.conv_out = nn.Conv3d(ch, out_channels, kernel_size=3, padding=1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights (especially important for output layer)."""
        # Zero-initialize the final conv layer (helps with training stability)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [B, C, D, H, W]
               - For training: concatenated [noisy_HD_latent, LD_latent]
               - C = in_channels (e.g., 16 for 8+8)
            timesteps: Timesteps [B] or [B, 1]
            cond: Optional conditioning (not used, conditioning is via concatenation)

        Returns:
            noise_pred: Predicted noise [B, out_channels, D, H, W]
        """
        # Time embedding
        time_emb = self.time_embed(timesteps)  # [B, time_embed_dim]

        # Initial convolution
        h = self.conv_in(x)  # [B, model_channels, D, H, W]

        # ====================================================================
        # Encoder
        # ====================================================================
        skip_connections = []

        block_idx = 0
        for level in range(self.num_levels):
            for _ in range(self.num_blocks):
                block_modules = self.down_blocks[block_idx]

                for module in block_modules:
                    # Check if module needs time_emb by checking for time_emb_proj attribute
                    # This is more robust than isinstance checks with FSDP wrapping
                    if hasattr(module, 'time_emb_proj'):
                        h = module(h, time_emb)
                    else:  # AttentionBlock3D or FlashAttentionBlock3D
                        h = module(h)

                block_idx += 1

            # Store skip connection BEFORE downsampling
            skip_connections.append(h)

            # Downsample
            h = self.down_samples[level](h)

        # ====================================================================
        # Bottleneck
        # ====================================================================
        h = self.mid_block1(h, time_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, time_emb)

        # ====================================================================
        # Decoder
        # ====================================================================
        block_idx = 0
        for level in reversed(range(self.num_levels)):
            for i in range(self.num_blocks + 1):
                # Get skip connection (only for first block at each level)
                if i == 0 and len(skip_connections) > 0:
                    skip = skip_connections.pop()

                    # Ensure spatial dimensions match (resize if needed)
                    if h.shape[2:] != skip.shape[2:]:
                        h = F.interpolate(h, size=skip.shape[2:], mode='trilinear', align_corners=False)

                    # Concatenate skip connection
                    h = torch.cat([h, skip], dim=1)

                block_modules = self.up_blocks[block_idx]

                for module in block_modules:
                    # Check if module needs time_emb by checking for time_emb_proj attribute
                    # This is more robust than isinstance checks with FSDP wrapping
                    if hasattr(module, 'time_emb_proj'):
                        h = module(h, time_emb)
                    else:  # AttentionBlock3D or FlashAttentionBlock3D
                        h = module(h)

                block_idx += 1

            # Upsample
            h = self.up_samples[self.num_levels - 1 - level](h)

        # ====================================================================
        # Output
        # ====================================================================
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)

        return h


# ============================================================================
# Testing and Utilities
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("Testing DiffusionUNet3D")
    print("="*70)

    # Create model with config from config_diffusion.yaml
    model = DiffusionUNet3D(
        in_channels=16,  # 8 (LD latent) + 8 (HD latent)
        out_channels=8,
        model_channels=128,
        channel_mult=[1, 2, 4, 8],
        num_blocks=3,
        attention_levels=[1, 2, 3],
        time_embed_dim=512,
        dropout=0.1,
        num_heads=8,
    )

    print(f"\nModel architecture:")
    print(f"  Input channels: 16 (8 LD + 8 HD)")
    print(f"  Output channels: 8 (noise prediction)")
    print(f"  Model channels: 128")
    print(f"  Channel multipliers: [1, 2, 4, 8]")
    print(f"  Blocks per level: 3")
    print(f"  Attention levels: [1, 2, 3]")

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nParameters:")
    print(f"  Total: {num_params:,}")
    print(f"  Trainable: {num_trainable:,}")

    # Test forward pass
    print(f"\nTesting forward pass:")

    # Typical latent shape: [B, C, D, H, W]
    # For 200×128×128 CT → 25×16×16 latent (8× compression)
    batch_size = 2
    latent_shape = (batch_size, 8, 25, 16, 16)

    print(f"  Input shape: {latent_shape}")

    # Create dummy inputs
    hd_noisy = torch.randn(batch_size, 8, 25, 16, 16)
    ld_cond = torch.randn(batch_size, 8, 25, 16, 16)

    # Concatenate for model input
    x = torch.cat([hd_noisy, ld_cond], dim=1)  # [B, 16, 25, 16, 16]
    t = torch.randint(0, 1000, (batch_size,))

    print(f"  Concatenated input: {tuple(x.shape)}")
    print(f"  Timesteps: {tuple(t.shape)}")

    # Forward pass
    with torch.no_grad():
        noise_pred = model(x, t)

    print(f"  Output shape: {tuple(noise_pred.shape)}")
    print(f"  Output range: [{noise_pred.min().item():.3f}, {noise_pred.max().item():.3f}]")

    # Test with different input sizes
    print(f"\nTesting different input sizes:")
    test_sizes = [
        (1, 16, 25, 16, 16),  # Standard
        (1, 16, 40, 32, 32),  # Larger
        (1, 16, 16, 8, 8),    # Smaller
    ]

    for size in test_sizes:
        x_test = torch.randn(*size)
        t_test = torch.randint(0, 1000, (size[0],))

        with torch.no_grad():
            out = model(x_test, t_test)

        print(f"  {tuple(size)} → {tuple(out.shape)}")

    print("\n" + "="*70)
    print("✓ All tests passed!")
    print("="*70)
