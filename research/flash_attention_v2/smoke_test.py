"""
CPU smoke test to ensure FlashAttention wrappers preserve baseline behavior.

This does NOT exercise the FlashAttention kernels (package not installed in CI),
but confirms that the naive fallback matches the original AttentionBlock3D.
"""

from __future__ import annotations

import torch

from flash_attention_plugin import DiffusionUNet3D, apply_flash_attention


def main() -> None:
    torch.manual_seed(0)
    model = DiffusionUNet3D()
    model.eval()

    x = torch.randn(1, 16, 25, 16, 16)
    t = torch.randint(0, 1000, (1,))

    with torch.no_grad():
        baseline = model(x, t)

    apply_flash_attention(model, verbose=False)

    with torch.no_grad():
        wrapped = model(x, t)

    max_diff = (baseline - wrapped).abs().max().item()
    print(f"Max deviation after wrapping: {max_diff:.6f}")
    assert torch.allclose(baseline, wrapped, atol=1e-5), "Outputs diverged after wrapping"


if __name__ == "__main__":
    main()
