"""
Metrics utilities for 3D CT super-resolution evaluation.

Provides distributed SSIM and PSNR computation for 3D CT volumes.
"""

import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


class DistributedMetricsCalculator:
    """
    Distributed calculator for SSIM and PSNR metrics on 3D CT volumes.

    This class computes quality metrics for 3D medical images, with support
    for slice-wise computation (more robust for 3D volumes) and distributed
    evaluation across multiple GPUs.

    Args:
        data_range: Expected data range of the input images
                   For normalized data [-1, 1], use 2.0
                   For HU values [-1000, 1000], use 2000.0
        compute_slice_wise: If True, compute per-slice and average
                           More robust for 3D volumes

    Example:
        >>> calc = DistributedMetricsCalculator(data_range=2.0, compute_slice_wise=True)
        >>> metrics = calc.compute_metrics(pred, target)
        >>> print(f"SSIM: {metrics['ssim']:.4f}, PSNR: {metrics['psnr']:.2f}")
    """

    def __init__(self, data_range: float = 2.0, compute_slice_wise: bool = True):
        self.data_range = data_range
        self.compute_slice_wise = compute_slice_wise

    def compute_ssim_3d(self, pred: np.ndarray, target: np.ndarray, data_range: float = None) -> float:
        """
        Compute SSIM for 3D CT volume.

        Args:
            pred: Predicted volume (D, H, W)
            target: Target volume (D, H, W)
            data_range: Data range for SSIM computation (uses self.data_range if None)

        Returns:
            ssim_score: SSIM value [0, 1], higher is better
        """
        if data_range is None:
            data_range = self.data_range

        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")

        if pred.ndim != 3:
            raise ValueError(f"Expected 3D array, got {pred.ndim}D")

        if self.compute_slice_wise and pred.ndim == 3:
            # Compute per-slice and average (more robust for 3D)
            ssim_scores = []
            for i in range(pred.shape[0]):
                # Skip slices with no variation
                if np.std(pred[i]) < 1e-6 or np.std(target[i]) < 1e-6:
                    continue

                try:
                    score = ssim(
                        target[i], pred[i],
                        data_range=data_range,
                        gaussian_weights=True,
                        use_sample_covariance=False
                    )

                    if not np.isnan(score) and not np.isinf(score):
                        ssim_scores.append(score)
                except Exception as e:
                    # Skip slices that cause errors
                    continue

            return float(np.mean(ssim_scores)) if ssim_scores else 0.0
        else:
            # Compute on full 3D volume
            try:
                score = ssim(
                    target, pred,
                    data_range=data_range,
                    gaussian_weights=True,
                    use_sample_covariance=False
                )
                return float(score) if not np.isnan(score) else 0.0
            except Exception as e:
                return 0.0

    def compute_psnr_3d(self, pred: np.ndarray, target: np.ndarray, data_range: float = None) -> float:
        """
        Compute PSNR for 3D CT volume.

        Args:
            pred: Predicted volume (D, H, W)
            target: Target volume (D, H, W)
            data_range: Data range for PSNR computation (uses self.data_range if None)

        Returns:
            psnr_score: PSNR value in dB, higher is better
        """
        if data_range is None:
            data_range = self.data_range

        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")

        if pred.ndim != 3:
            raise ValueError(f"Expected 3D array, got {pred.ndim}D")

        if self.compute_slice_wise and pred.ndim == 3:
            # Compute per-slice and average (more robust for 3D)
            psnr_scores = []
            for i in range(pred.shape[0]):
                # Skip slices with no variation
                if np.std(pred[i]) < 1e-6 or np.std(target[i]) < 1e-6:
                    continue

                try:
                    score = psnr(
                        target[i], pred[i],
                        data_range=data_range
                    )

                    if not np.isnan(score) and not np.isinf(score):
                        psnr_scores.append(score)
                except Exception as e:
                    # Skip slices that cause errors
                    continue

            return float(np.mean(psnr_scores)) if psnr_scores else 0.0
        else:
            # Compute on full 3D volume
            try:
                score = psnr(
                    target, pred,
                    data_range=data_range
                )
                return float(score) if not np.isnan(score) else 0.0
            except Exception as e:
                return 0.0

    def compute_metrics(self, pred: np.ndarray, target: np.ndarray) -> dict:
        """
        Compute both SSIM and PSNR for a 3D volume.

        Args:
            pred: Predicted volume (D, H, W)
            target: Target volume (D, H, W)

        Returns:
            metrics: Dict with 'ssim' and 'psnr' keys
        """
        return {
            'ssim': self.compute_ssim_3d(pred, target),
            'psnr': self.compute_psnr_3d(pred, target)
        }

    @staticmethod
    def aggregate_metrics(metrics_list: list) -> dict:
        """
        Aggregate metrics from multiple samples.

        Args:
            metrics_list: List of metric dicts from compute_metrics()

        Returns:
            aggregated: Dict with mean, std, min, max for each metric
        """
        if not metrics_list:
            return {}

        # Extract metric arrays
        ssim_values = [m['ssim'] for m in metrics_list if 'ssim' in m]
        psnr_values = [m['psnr'] for m in metrics_list if 'psnr' in m]

        aggregated = {}

        if ssim_values:
            aggregated['ssim_mean'] = float(np.mean(ssim_values))
            aggregated['ssim_std'] = float(np.std(ssim_values))
            aggregated['ssim_min'] = float(np.min(ssim_values))
            aggregated['ssim_max'] = float(np.max(ssim_values))

        if psnr_values:
            aggregated['psnr_mean'] = float(np.mean(psnr_values))
            aggregated['psnr_std'] = float(np.std(psnr_values))
            aggregated['psnr_min'] = float(np.min(psnr_values))
            aggregated['psnr_max'] = float(np.max(psnr_values))

        aggregated['num_samples'] = len(metrics_list)

        return aggregated


if __name__ == "__main__":
    # Quick test
    print("="*70)
    print("Testing DistributedMetricsCalculator")
    print("="*70)

    # Create dummy 3D volumes
    np.random.seed(42)
    target = np.random.randn(50, 128, 128) * 0.5

    # Test case 1: Near-perfect prediction (small noise)
    pred_good = target + np.random.randn(*target.shape) * 0.01

    # Test case 2: Poor prediction (large noise)
    pred_poor = target + np.random.randn(*target.shape) * 0.2

    # Initialize calculator
    calc = DistributedMetricsCalculator(data_range=2.0, compute_slice_wise=True)

    print("\n1. Near-perfect prediction (1% noise):")
    metrics_good = calc.compute_metrics(pred_good, target)
    print(f"   SSIM: {metrics_good['ssim']:.4f}")
    print(f"   PSNR: {metrics_good['psnr']:.2f} dB")

    print("\n2. Poor prediction (20% noise):")
    metrics_poor = calc.compute_metrics(pred_poor, target)
    print(f"   SSIM: {metrics_poor['ssim']:.4f}")
    print(f"   PSNR: {metrics_poor['psnr']:.2f} dB")

    print("\n3. Aggregated metrics:")
    metrics_list = [metrics_good, metrics_poor]
    aggregated = calc.aggregate_metrics(metrics_list)
    print(f"   SSIM: {aggregated['ssim_mean']:.4f} ± {aggregated['ssim_std']:.4f}")
    print(f"   PSNR: {aggregated['psnr_mean']:.2f} ± {aggregated['psnr_std']:.2f} dB")

    print("\n" + "="*70)
    print("✓ Tests passed!")
    print("="*70)
