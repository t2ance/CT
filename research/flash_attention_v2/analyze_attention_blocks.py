"""
Utility script to inspect attention usage in DiffusionUNet3D.

Counts the number of AttentionBlock3D modules and prints their
approximate sequence lengths for a representative latent tensor.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import List, Tuple

import importlib.util

import torch


def load_diffusion_unet() -> Tuple[type, type]:
    """Load DiffusionUNet3D and AttentionBlock3D without importing models package."""
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "models" / "diffusion_unet.py"
    spec = importlib.util.spec_from_file_location("diffusion_unet_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module.DiffusionUNet3D, module.AttentionBlock3D


def summarize_attention(
    model: torch.nn.Module,
    sample_shape: Tuple[int, int, int, int, int],
    attention_cls: type,
) -> None:
    """Print attention block counts and their spatial shapes."""
    attn_modules: List[torch.nn.Module] = []
    for module in model.modules():
        if isinstance(module, attention_cls):
            attn_modules.append(module)

    print(f"Total AttentionBlock3D modules: {len(attn_modules)}")

    # Track spatial shapes after each attention call using forward hooks
    shapes: Counter = Counter()

    hooks = []

    def hook_fn(_: torch.nn.Module, inputs, outputs):
        x = inputs[0]
        shapes[x.shape[2:]] += 1

    for module in attn_modules:
        hooks.append(module.register_forward_hook(hook_fn))

    with torch.no_grad():
        x = torch.randn(*sample_shape)
        t = torch.randint(0, 1000, (sample_shape[0],), dtype=torch.long)
        model(x, t)

    for h in hooks:
        h.remove()

    print("Attention spatial shapes and counts:")
    for shape, count in shapes.items():
        seq_len = shape[0] * shape[1] * shape[2]
        print(f"  {shape} -> {seq_len:,} tokens × {count} occurrences")


if __name__ == "__main__":
    DiffusionUNet3D, AttentionBlock3D = load_diffusion_unet()
    model = DiffusionUNet3D()
    summarize_attention(
        model,
        sample_shape=(1, 16, 25, 16, 16),
        attention_cls=AttentionBlock3D,
    )
