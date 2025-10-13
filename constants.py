"""
Global constants used across the CT diffusion project.
"""

# Default HU clipping window applied before normalization and for visualization.
# Narrow window emphasises soft tissue contrast while keeping values stable.
HU_CLIP_RANGE = (-200.0, 200.0)
HU_CLIP_WIDTH = HU_CLIP_RANGE[1] - HU_CLIP_RANGE[0]

