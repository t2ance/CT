"""
Visualization utilities for 3D CT super-resolution
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import gridspec
import io
from PIL import Image


class CTVisualization:
    """
    Comprehensive visualization for 3D CT super-resolution results.

    Methods:
    1. Multi-slice comparison (axial, coronal, sagittal)
    2. Central slice comparison with error map
    3. 3D volume rendering (optional)
    4. Profile plots along lines
    5. Histogram comparison
    """

    def __init__(
        self,
        clip_range=(-1000, 1000),
        display_range=None,
        percentile_window=(1.0, 99.0),
    ):
        self.clip_range = clip_range
        self.display_range = display_range
        self.percentile_window = percentile_window

    def _denormalize(self, tensor):
        """Convert from [-1, 1] to HU values"""
        # Assuming normalized as: (data - min) / (max - min) * 2 - 1
        data = (tensor + 1) / 2  # [-1, 1] → [0, 1]
        data = data * (self.clip_range[1] - self.clip_range[0]) + self.clip_range[0]
        return data

    def _to_numpy(self, tensor):
        """Convert tensor to numpy and handle dimensions"""
        if torch.is_tensor(tensor):
            tensor = tensor.detach().cpu().numpy()

        # Remove batch and channel dims if present
        if tensor.ndim == 5:  # (B, C, D, H, W)
            tensor = tensor[0, 0]
        elif tensor.ndim == 4:  # (C, D, H, W)
            tensor = tensor[0]

        return tensor

    def _resolve_display_range(self, tensor):
        """Determine display window for visualization."""
        if self.display_range is not None:
            return self.display_range

        if self.percentile_window is not None:
            lo_pct, hi_pct = self.percentile_window
            if 0.0 <= lo_pct < hi_pct <= 100.0:
                lo_val = float(np.percentile(tensor, lo_pct))
                hi_val = float(np.percentile(tensor, hi_pct))
                if hi_val - lo_val > 1e-3:
                    return (lo_val, hi_val)

        return self.clip_range

    def create_multi_slice_comparison(self, ld, pred, gt, num_slices=5):
        """
        Create multi-slice comparison showing LD, Prediction, and GT.

        Args:
            ld: Low-dose input (B, C, D, H, W) or (D, H, W)
            pred: Predicted HD (B, C, D, H, W) or (D, H, W)
            gt: Ground truth HD (B, C, D, H, W) or (D, H, W)
            num_slices: Number of slices to show

        Returns:
            PIL Image
        """
        ld = self._to_numpy(ld)
        pred = self._to_numpy(pred)
        gt = self._to_numpy(gt)

        # Denormalize from [-1, 1] to HU values
        ld = self._denormalize(ld)
        pred = self._denormalize(pred)
        gt = self._denormalize(gt)

        # Select evenly spaced slices from GT
        gt_depth = gt.shape[0]
        slice_indices = np.linspace(0, gt_depth-1, num_slices, dtype=int)
        vmin, vmax = self._resolve_display_range(gt)

        # Upsample LD to match GT depth for visualization
        ld_upsampled = self._upsample_ld(ld, gt_depth)

        # Create figure
        fig = plt.figure(figsize=(15, 3 * num_slices))
        gs = gridspec.GridSpec(num_slices, 4, wspace=0.3, hspace=0.3)

        for i, slice_idx in enumerate(slice_indices):
            # LD
            ax1 = fig.add_subplot(gs[i, 0])
            ax1.imshow(ld_upsampled[slice_idx], cmap='gray', vmin=vmin, vmax=vmax)
            ax1.set_title(f'LD (Slice {slice_idx})')
            ax1.axis('off')

            # Prediction
            ax2 = fig.add_subplot(gs[i, 1])
            ax2.imshow(pred[slice_idx], cmap='gray', vmin=vmin, vmax=vmax)
            ax2.set_title(f'Predicted (Slice {slice_idx})')
            ax2.axis('off')

            # Ground Truth
            ax3 = fig.add_subplot(gs[i, 2])
            ax3.imshow(gt[slice_idx], cmap='gray', vmin=vmin, vmax=vmax)
            ax3.set_title(f'Ground Truth (Slice {slice_idx})')
            ax3.axis('off')

            # Error map
            ax4 = fig.add_subplot(gs[i, 3])
            error = np.abs(pred[slice_idx] - gt[slice_idx])
            im = ax4.imshow(error, cmap='hot', vmin=0, vmax=100)
            ax4.set_title(f'Error Map (MAE={error.mean():.2f})')
            ax4.axis('off')
            plt.colorbar(im, ax=ax4, fraction=0.046)

        # Convert to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf)

    def create_central_slice_comparison(self, ld, pred, gt):
        """
        Detailed comparison of central axial slice with multiple views.

        Returns:
            PIL Image
        """
        ld = self._to_numpy(ld)
        pred = self._to_numpy(pred)
        gt = self._to_numpy(gt)

        # Denormalize from [-1, 1] to HU values
        ld = self._denormalize(ld)
        pred = self._denormalize(pred)
        gt = self._denormalize(gt)

        # Get central slices
        center_idx = gt.shape[0] // 2
        ld_center = ld[ld.shape[0] // 2] if ld.shape[0] > 1 else ld[0]
        vmin, vmax = self._resolve_display_range(gt)
        pred_center = pred[center_idx]
        gt_center = gt[center_idx]

        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Row 1: Images
        axes[0, 0].imshow(ld_center, cmap='gray', vmin=vmin, vmax=vmax)
        axes[0, 0].set_title('Low-Dose Input', fontsize=14)
        axes[0, 0].axis('off')

        axes[0, 1].imshow(pred_center, cmap='gray', vmin=vmin, vmax=vmax)
        axes[0, 1].set_title('Prediction', fontsize=14)
        axes[0, 1].axis('off')

        axes[0, 2].imshow(gt_center, cmap='gray', vmin=vmin, vmax=vmax)
        axes[0, 2].set_title('Ground Truth', fontsize=14)
        axes[0, 2].axis('off')

        # Row 2: Analysis
        # Error map
        error = np.abs(pred_center - gt_center)
        im1 = axes[1, 0].imshow(error, cmap='hot', vmin=0, vmax=100)
        axes[1, 0].set_title(f'Absolute Error (MAE: {error.mean():.2f} HU)', fontsize=14)
        axes[1, 0].axis('off')
        plt.colorbar(im1, ax=axes[1, 0], fraction=0.046)

        # Profile plot (horizontal line through center)
        mid_row = pred_center.shape[0] // 2
        axes[1, 1].plot(ld_center[mid_row], label='LD Input', alpha=0.7, linewidth=1)
        axes[1, 1].plot(pred_center[mid_row], label='Prediction', alpha=0.7, linewidth=1.5)
        axes[1, 1].plot(gt_center[mid_row], label='Ground Truth', alpha=0.7, linewidth=1.5, linestyle='--')
        axes[1, 1].set_title('Horizontal Profile (Center Row)', fontsize=14)
        axes[1, 1].set_xlabel('Pixel Position')
        axes[1, 1].set_ylabel('HU Value')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # Histogram comparison
        axes[1, 2].hist(pred_center.flatten(), bins=50, alpha=0.6, label='Prediction', density=True)
        axes[1, 2].hist(gt_center.flatten(), bins=50, alpha=0.6, label='Ground Truth', density=True)
        axes[1, 2].set_title('Intensity Histogram', fontsize=14)
        axes[1, 2].set_xlabel('HU Value')
        axes[1, 2].set_ylabel('Density')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()

        # Convert to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf)

    def create_orthogonal_views(self, ld, pred, gt):
        """
        Show axial, coronal, and sagittal views of central slices.

        Returns:
            PIL Image
        """
        ld = self._to_numpy(ld)
        pred = self._to_numpy(pred)
        gt = self._to_numpy(gt)

        # Denormalize from [-1, 1] to HU values
        ld = self._denormalize(ld)
        pred = self._denormalize(pred)
        gt = self._denormalize(gt)

        # Get central slices for each view
        d, h, w = gt.shape

        # Axial (z-slice)
        axial_ld = ld[ld.shape[0] // 2] if ld.shape[0] > 1 else ld[0]
        axial_pred = pred[d // 2]
        axial_gt = gt[d // 2]

        # Coronal (y-slice)
        coronal_ld = ld[:, ld.shape[1] // 2, :] if ld.shape[0] > 1 else ld[0, ld.shape[1] // 2, :]
        coronal_pred = pred[:, h // 2, :]
        coronal_gt = gt[:, h // 2, :]

        # Sagittal (x-slice)
        sagittal_ld = ld[:, :, ld.shape[2] // 2] if ld.shape[0] > 1 else ld[0, :, ld.shape[2] // 2]
        sagittal_pred = pred[:, :, w // 2]
        sagittal_gt = gt[:, :, w // 2]

        # Create figure
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        vmin, vmax = self._resolve_display_range(gt)

        views = [
            ('Axial', axial_ld, axial_pred, axial_gt),
            ('Coronal', coronal_ld, coronal_pred, coronal_gt),
            ('Sagittal', sagittal_ld, sagittal_pred, sagittal_gt)
        ]

        for i, (view_name, ld_slice, pred_slice, gt_slice) in enumerate(views):
            axes[i, 0].imshow(ld_slice, cmap='gray', vmin=vmin, vmax=vmax)
            axes[i, 0].set_title(f'{view_name} - LD Input')
            axes[i, 0].axis('off')

            axes[i, 1].imshow(pred_slice, cmap='gray', vmin=vmin, vmax=vmax)
            axes[i, 1].set_title(f'{view_name} - Prediction')
            axes[i, 1].axis('off')

            axes[i, 2].imshow(gt_slice, cmap='gray', vmin=vmin, vmax=vmax)
            axes[i, 2].set_title(f'{view_name} - Ground Truth')
            axes[i, 2].axis('off')

        plt.tight_layout()

        # Convert to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf)

    def create_mip_comparison(self, ld, pred, gt):
        """
        Maximum Intensity Projection (MIP) comparison for 3D volumes.

        Returns:
            PIL Image
        """
        ld = self._to_numpy(ld)
        pred = self._to_numpy(pred)
        gt = self._to_numpy(gt)

        # Denormalize from [-1, 1] to HU values
        ld = self._denormalize(ld)
        pred = self._denormalize(pred)
        gt = self._denormalize(gt)

        # Upsample LD to match GT
        ld_upsampled = self._upsample_ld(ld, gt.shape[0])
        vmin, vmax = self._resolve_display_range(gt)

        # Compute MIP along each axis
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))

        projections = [
            ('Z-axis (Axial)', 0),
            ('Y-axis (Coronal)', 1),
            ('X-axis (Sagittal)', 2)
        ]

        for i, (proj_name, axis) in enumerate(projections):
            # LD MIP
            axes[i, 0].imshow(np.max(ld_upsampled, axis=axis), cmap='gray', vmin=vmin, vmax=vmax)
            axes[i, 0].set_title(f'{proj_name} MIP - LD')
            axes[i, 0].axis('off')

            # Prediction MIP
            axes[i, 1].imshow(np.max(pred, axis=axis), cmap='gray', vmin=vmin, vmax=vmax)
            axes[i, 1].set_title(f'{proj_name} MIP - Prediction')
            axes[i, 1].axis('off')

            # GT MIP
            axes[i, 2].imshow(np.max(gt, axis=axis), cmap='gray', vmin=vmin, vmax=vmax)
            axes[i, 2].set_title(f'{proj_name} MIP - Ground Truth')
            axes[i, 2].axis('off')

        plt.tight_layout()

        # Convert to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf)

    def _upsample_ld(self, ld, target_depth):
        """Upsample LD to match HD depth for visualization"""
        if ld.shape[0] == target_depth:
            return ld

        ld_tensor = torch.from_numpy(ld[None, None, ...]).float()
        upsampled = torch.nn.functional.interpolate(
            ld_tensor,
            size=(target_depth, ld.shape[1], ld.shape[2]),
            mode='trilinear',
            align_corners=False
        )
        return upsampled.squeeze().numpy()

    def create_all_visualizations(self, ld, pred, gt, prefix=""):
        """
        Create all visualizations and return as dict for WandB logging.

        Args:
            ld: Low-dose input
            pred: Predicted HD
            gt: Ground truth HD
            prefix: Prefix for keys (e.g., "train/" or "val/")

        Returns:
            Dict of {name: PIL Image}
        """
        visualizations = {}

        try:
            visualizations[f'{prefix}multi_slice'] = self.create_multi_slice_comparison(ld, pred, gt)
        except Exception as e:
            print(f"Warning: Failed to create multi-slice viz: {e}")

        try:
            visualizations[f'{prefix}central_slice'] = self.create_central_slice_comparison(ld, pred, gt)
        except Exception as e:
            print(f"Warning: Failed to create central slice viz: {e}")

        try:
            visualizations[f'{prefix}orthogonal'] = self.create_orthogonal_views(ld, pred, gt)
        except Exception as e:
            print(f"Warning: Failed to create orthogonal viz: {e}")

        try:
            visualizations[f'{prefix}mip'] = self.create_mip_comparison(ld, pred, gt)
        except Exception as e:
            print(f"Warning: Failed to create MIP viz: {e}")

        return visualizations


def create_loss_plot(train_losses, val_losses):
    """Create training/validation loss plot"""
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)

    ax.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Progress', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)
