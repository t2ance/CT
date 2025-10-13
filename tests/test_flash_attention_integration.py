import torch

from model_factory import build_diffusion_model
from research.flash_attention_v2 import flash_attention_plugin as flash_plugin

FlashAttentionBlock3D = flash_plugin.FlashAttentionBlock3D
apply_flash_attention = flash_plugin.apply_flash_attention
DiffusionUNet3D = flash_plugin.DiffusionUNet3D


def base_config(use_flash: bool = False) -> dict:
    return {
        "in_channels": 8,
        "out_channels": 8,
        "model_channels": 64,
        "channel_mult": [1, 2],
        "num_blocks": 2,
        "attention_levels": [1],
        "time_embed_dim": 128,
        "dropout": 0.0,
        "num_heads": 4,
        "concat_conditioning": True,
        "use_flash_attention": use_flash,
    }


def count_flash_blocks(model: torch.nn.Module) -> int:
    return sum(1 for module in model.modules() if isinstance(module, FlashAttentionBlock3D))


def test_apply_flash_attention_replaces_blocks():
    model = DiffusionUNet3D()
    replaced = apply_flash_attention(model, verbose=False)
    assert replaced > 0
    assert count_flash_blocks(model) == replaced


def test_flash_attention_flag_in_factory():
    config = base_config(use_flash=False)
    model = build_diffusion_model(config, verbose=False)
    assert count_flash_blocks(model) == 0

    config = base_config(use_flash=True)
    model = build_diffusion_model(config, verbose=False)
    assert count_flash_blocks(model) > 0


def test_flash_attention_fallback_output_matches():
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

    assert torch.allclose(baseline, wrapped, atol=1e-5)


if __name__ == "__main__":
    test_apply_flash_attention_replaces_blocks()
    test_flash_attention_flag_in_factory()
    test_flash_attention_fallback_output_matches()
    print("All FlashAttention integration tests passed.")
