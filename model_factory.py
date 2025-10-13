"""
Factory helpers for constructing diffusion models with optional FlashAttention.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict

from research.flash_attention_v2 import flash_attention_plugin as flash_plugin

DiffusionUNet3D = flash_plugin.DiffusionUNet3D
apply_flash_attention = flash_plugin.apply_flash_attention


def build_diffusion_model(model_config: Dict[str, Any], *, verbose: bool = True) -> DiffusionUNet3D:
    """
    Instantiate DiffusionUNet3D using project configuration and optionally
    wrap attention blocks with FlashAttention.
    """
    in_channels = model_config["in_channels"]
    if model_config.get("concat_conditioning", True):
        in_channels *= 2

    model = DiffusionUNet3D(
        in_channels=in_channels,
        out_channels=model_config["out_channels"],
        model_channels=model_config["model_channels"],
        channel_mult=model_config["channel_mult"],
        num_blocks=model_config["num_blocks"],
        attention_levels=model_config["attention_levels"],
        time_embed_dim=model_config["time_embed_dim"],
        dropout=model_config.get("dropout", 0.0),
        num_heads=model_config.get("num_heads", 8),
    )

    if model_config.get("use_flash_attention", False):
        replaced = apply_flash_attention(model, verbose=False)

        if replaced == 0:
            if getattr(flash_plugin, "flash_attn_qkvpacked_func", None) is None:
                warnings.warn(
                    "FlashAttention requested but the flash-attn package is unavailable. "
                    "Continuing with standard attention."
                )
            else:
                warnings.warn(
                    "FlashAttention requested but no AttentionBlock3D modules were replaced."
                )
        elif verbose:
            print(f"FlashAttention enabled: wrapped {replaced} attention blocks.")

    return model


__all__ = ["DiffusionUNet3D", "build_diffusion_model"]
