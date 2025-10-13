"""
Models module for CT super-resolution with latent diffusion.
"""

from .vqae_wrapper import FrozenVQAE

__all__ = ['FrozenVQAE']

# Diffusion model will be imported when available
try:
    from .diffusion_unet import DiffusionUNet3D
    __all__.append('DiffusionUNet3D')
except ImportError:
    pass
