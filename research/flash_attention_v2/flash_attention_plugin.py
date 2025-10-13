"""
FlashAttention v2 integration hooks for DiffusionUNet3D.

This module provides:
  - `FlashAttentionBlock3D`: drop-in wrapper around the existing AttentionBlock3D
    that dispatches to FlashAttention when the inputs are CUDA tensors in
    fp16/bf16.
  - `apply_flash_attention(model)`: utility to replace every AttentionBlock3D in a
    DiffusionUNet3D instance with the Flash-aware wrapper.

All logic lives in a separate file to keep the core model untouched and the
integration reversible.
"""

from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn

try:
    from flash_attn.flash_attn_interface import flash_attn_qkvpacked_func
except ImportError:  # pragma: no cover - handled in _can_use_flash path
    flash_attn_qkvpacked_func = None  # type: ignore


def _load_attention_types() -> Tuple[type, type]:
    """Load DiffusionUNet3D/AttentionBlock3D without importing models package."""
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "models" / "diffusion_unet.py"
    spec = importlib.util.spec_from_file_location("diffusion_unet_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module.DiffusionUNet3D, module.AttentionBlock3D


DiffusionUNet3D, AttentionBlock3D = _load_attention_types()


class FlashAttentionBlock3D(nn.Module):
    """
    Wrapper that reuses an existing AttentionBlock3D's parameters, dispatching
    the attention matmul to FlashAttention v2 when feasible.
    """

    def __init__(self, attention_block: AttentionBlock3D):
        super().__init__()
        self.channels = attention_block.channels
        self.num_heads = attention_block.num_heads
        self.head_dim = self.channels // self.num_heads
        self.scale = self.head_dim ** -0.5

        # Reuse the original submodules / parameters.
        self.norm = attention_block.norm
        self.qkv = attention_block.qkv
        self.proj_out = attention_block.proj_out

    @staticmethod
    def _can_use_flash(x: torch.Tensor) -> bool:
        """
        FlashAttention requirements (FlashSelfAttention.forward docstring):
          - CUDA tensor
          - dtype fp16 or bf16
          - flash_attn package installed
        """
        if flash_attn_qkvpacked_func is None:
            return False
        if not x.is_cuda:
            return False
        if x.dtype not in (torch.float16, torch.bfloat16):
            return False
        return True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._can_use_flash(x):
            return self._naive_forward(x)
        return self._flash_forward(x)

    # ---------------------------------------------------------------------
    # Original AttentionBlock3D pathway (for fallback / CPU inference)
    # ---------------------------------------------------------------------
    def _naive_forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        qkv = qkv.reshape(B, C * 3, D * H * W).permute(0, 2, 1)
        q, k, v = qkv.chunk(3, dim=-1)
        head_dim = self.head_dim
        num_heads = self.num_heads
        q = q.reshape(B, D * H * W, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, D * H * W, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, D * H * W, num_heads, head_dim).permute(0, 2, 1, 3)
        attn = torch.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)
        h = attn @ v
        h = h.permute(0, 2, 1, 3).reshape(B, C, D, H, W)
        h = self.proj_out(h)
        return x + h

    # ---------------------------------------------------------------------
    # FlashAttention path
    # ---------------------------------------------------------------------
    def _flash_forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, D, H, W)
        qkv = qkv.permute(0, 4, 5, 6, 1, 2, 3).reshape(B, D * H * W, 3, self.num_heads, self.head_dim)

        # FlashAttention expects (B, S, 3, n_heads, head_dim)
        out = flash_attn_qkvpacked_func(
            qkv,
            dropout_p=0.0,
            softmax_scale=None,
            causal=False,
        )

        out = out.reshape(B, D, H, W, self.num_heads, self.head_dim)
        out = out.permute(0, 4, 5, 1, 2, 3).reshape(B, C, D, H, W)
        out = self.proj_out(out)
        return x + out


def apply_flash_attention(model: nn.Module, verbose: bool = True) -> int:
    """
    Replace all AttentionBlock3D modules with FlashAttentionBlock3D wrappers.

    Returns:
        Number of modules swapped.
    """
    replaced = 0

    def _replace(module: nn.Module) -> None:
        nonlocal replaced
        for name, child in list(module.named_children()):
            if isinstance(child, AttentionBlock3D):
                module._modules[name] = FlashAttentionBlock3D(child)
                replaced += 1
            else:
                _replace(child)

    _replace(model)

    if verbose:
        if replaced == 0:
            warnings.warn("No AttentionBlock3D modules were found to replace.")
        elif flash_attn_qkvpacked_func is None:
            warnings.warn(
                "FlashAttention is not installed; swapped modules will fall back to the naive path.",
            )
        else:
            warnings.warn(
                f"Replaced {replaced} AttentionBlock3D modules with FlashAttention wrappers. "
                "Ensure inputs are CUDA tensors in bf16/fp16 for acceleration."
            )

    return replaced


__all__ = ["FlashAttentionBlock3D", "apply_flash_attention"]
