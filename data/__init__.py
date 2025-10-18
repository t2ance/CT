"""
Data module for CT super-resolution.
"""

# Import dataset classes when available
try:
    from .latent_dataset import LatentCTDataset, create_latent_dataloaders
    __all__ = ['LatentCTDataset', 'create_latent_dataloaders']
except ImportError:
    __all__ = []
