# CT Super-Resolution with Latent Diffusion

**Status:** PARTIALLY RECOVERED (2025-01-12)

## Recovery Summary

This codebase was partially recovered from context after accidental deletion. Most utility files and configurations were successfully restored, but some core training components need reconstruction.

### ✅ Fully Recovered Files

These files are **complete and ready to use**:

1. **`models/vqae_wrapper.py`** - Frozen VQ-AE wrapper
   - Loads pretrained 3D-MedDiffusion VQ-AE
   - Encoding/decoding CT volumes to/from latent space
   - 8× spatial compression, latent channels: 8

2. **`utils/visualization.py`** - CT visualization utilities
   - Multi-slice comparison
   - Central slice analysis with error maps
   - Orthogonal views (axial, coronal, sagittal)
   - MIP (Maximum Intensity Projection)
   - Ready for wandb logging

3. **`utils/metrics.py`** - Distributed metrics calculator
   - SSIM and PSNR computation for 3D CT volumes
   - Slice-wise computation (more robust)
   - Distributed evaluation across GPUs
   - **BUG FIX APPLIED:** Uses correct data_range=2.0

4. **`data/latent_dataset.py`** - Latent dataset loader
   - Loads pre-computed VQ-AE latents from cache
   - Supports train/val splits
   - Returns dicts with 'ld_latent' and 'hd_latent'

5. **`config_diffusion.yaml`** - Complete configuration
   - **Scaled-up model:** 128 channels, deeper hierarchy
   - **Fixed batch size:** 32 (was 512)
   - **Fixed learning rate:** 2e-4 (was 5e-5)
   - **Linear LR scheduler** with warmup (6%)
   - **BUG FIX:** data_range=2.0 (was 2000.0)
   - Two-tier validation strategy

6. **`requirements.txt`** - Python dependencies
7. **`models/__init__.py`, `utils/__init__.py`, `data/__init__.py`** - Module initializers

### ⚠️ Partially Recovered / Needs Reconstruction

1. **`models/diffusion_unet.py`** - 3D UNet for diffusion
   - **Status:** STUB ONLY
   - **Action needed:** Implement or restore from backup
   - See stub file for architecture specifications

2. **`train_diffusion.py`** - Main training script
   - **Status:** PARTIAL (key functions recovered, main loop missing)
   - **Action needed:** Reconstruct or restore from backup
   - See `TRAIN_DIFFUSION_RECOVERY_NOTE.md` for details

### ❌ Not Recovered (Data Files)

These files were intentionally not recovered:
- `/data/ct_scans/` - Original CT scan data (regenerate if needed)
- `/outputs/checkpoints/` - Model checkpoints (lost)
- `/outputs/logs/` - Training logs (lost)

## Recent Bug Fixes (Before Deletion)

### 1. Metrics Data Range Bug (CRITICAL)
**Issue:** SSIM > 0.99, PSNR > 60 dB, but visualizations showed poor quality

**Root cause:** VQ-AE outputs normalized values in [-1, 1], not HU values within the new default window (-200, 200). The config used `data_range=2000.0`, causing inflated metrics.

**Fix applied:**
```yaml
# config_diffusion.yaml
validation:
  data_range: 2.0  # Was 2000.0 (FIXED)
```

**Impact:** PSNR decreased by ~60 dB to realistic values (25-35 dB range)

### 2. Distributed Validation Bug
**Issue:** Both GPU ranks processed the same validation samples

**Root cause:** `val_loader` was prepared by Accelerate (auto-distributed), but `validate_with_metrics()` expected to manually distribute work.

**Fix applied:**
- Created separate `val_loader_unprepared` (NOT prepared by accelerate)
- `validate_with_metrics()` uses unprepared loader
- Each process sees ALL samples, manually skips non-assigned ones

### 3. Model Scaling Issues
**Issue:** Train loss around 0.6, poor performance

**Fixes applied:**
- Scaled up model: 64 → 128 channels
- Fixed batch size: 512 → 32 (critical for diffusion)
- Increased LR: 5e-5 → 2e-4
- Added gradient accumulation: 2 steps
- Added dropout: 0.1

## Project Architecture

### Two-Stage Latent Diffusion

```
Stage 1 (Frozen): VQ-AE
  CT Volume [1, 200, 128, 128]
    ↓ Encode (8× compression)
  Latent [8, 25, 16, 16]
    ↓ Decode
  CT Volume [1, 200, 128, 128]

Stage 2 (Training): Diffusion in Latent Space
  LD Latent [8, 25, 16, 16] (condition)
  + Noise
    ↓ Diffusion Model (3D UNet)
  HD Latent [8, 25, 16, 16] (denoised)
    ↓ VQ-AE Decode
  HD CT Volume [1, 200, 128, 128]
```

### Key Features
- **Latent diffusion:** 8× faster than pixel-space diffusion
- **Conditional generation:** Conditioned on low-dose CT latent
- **Distributed training:** HuggingFace Accelerate with FSDP
- **Two-tier validation:**
  - Lightweight (MSE): Every 50 steps, fast
  - Full (SSIM/PSNR): Every 500 steps, slow but accurate

## Next Steps to Resume Training

### Option 1: Restore from Backup (Recommended)
If you have a backup or git stash:
```bash
# Check git reflog for previous state
git reflog

# Or restore from backup
# cp -r /path/to/backup/* .
```

### Option 2: Reconstruct Missing Components

1. **Implement `models/diffusion_unet.py`:**
   - See stub file for specifications
   - Reference: diffusers UNet2DConditionModel (adapt to 3D)
   - Use config parameters: 128 channels, [1,2,4,8] multipliers

2. **Reconstruct `train_diffusion.py`:**
   - See `TRAIN_DIFFUSION_RECOVERY_NOTE.md` for recovered sections
   - Main loop structure: Standard DDPM training
   - Key: Use unprepared dataloader for metrics

3. **Test incrementally:**
   ```bash
   # Test VQ-AE
   python models/vqae_wrapper.py

   # Test dataset
   python data/latent_dataset.py

   # Test metrics
   python utils/metrics.py

   # Test training (once implemented)
   accelerate launch --multi_gpu --num_processes=2 train_diffusion.py
   ```

## Expected Metrics (After Fixes)

### For normalized data (VQ-AE outputs):
- **SSIM:**
  - Excellent: 0.95 - 1.00
  - Good: 0.85 - 0.95
  - Fair: 0.70 - 0.85
  - Poor: < 0.70

- **PSNR:**
  - Excellent: > 35 dB
  - Good: 30 - 35 dB
  - Fair: 25 - 30 dB
  - Poor: < 25 dB

### Training loss:
- Initial: ~0.5 - 1.0 (MSE on noise prediction)
- Well-trained: ~0.1 - 0.2

## Configuration Highlights

```yaml
# Model (scaled up)
model:
  model_channels: 128        # Was 64
  channel_mult: [1, 2, 4, 8]  # Was [1, 2, 4, 4]
  num_blocks: 3              # Was 2
  time_embed_dim: 512        # Was 256

# Training (fixed)
training:
  batch_size: 32             # Was 512 (CRITICAL FIX)
  learning_rate: 2.0e-4      # Was 5e-5
  lr_scheduler_type: 'linear'
  warmup_ratio: 0.06

# Validation (fixed)
validation:
  data_range: 2.0            # Was 2000.0 (CRITICAL FIX)
  full_val_every: 500
  num_samples: 32
```

## File Structure

```
ct/
├── models/
│   ├── __init__.py              ✅ Complete
│   ├── vqae_wrapper.py          ✅ Complete
│   └── diffusion_unet.py        ⚠️ Stub only (needs implementation)
│
├── utils/
│   ├── __init__.py              ✅ Complete
│   ├── visualization.py         ✅ Complete
│   └── metrics.py               ✅ Complete
│
├── data/
│   ├── __init__.py              ✅ Complete
│   └── latent_dataset.py        ✅ Complete
│
├── config_diffusion.yaml        ✅ Complete (with fixes)
├── requirements.txt             ✅ Complete
├── train_diffusion.py           ⚠️ Missing (see recovery note)
├── README.md                    ✅ This file
└── TRAIN_DIFFUSION_RECOVERY_NOTE.md  ✅ Detailed recovery guide
```

## Important Notes

1. **VQ-AE Checkpoint Required:**
   ```
   /data2/peijia/projects/BioAgent/3D-MedDiffusion/checkpoints/PatchVolume_8x_s2.ckpt
   ```
   Ensure this exists and is accessible.

2. **Latent Cache Required:**
   ```
   ./latents_cache/train/
   ./latents_cache/val/
   ```
   You'll need to re-encode CT volumes to latents if cache is missing.

3. **Distributed Training:**
   ```bash
   # Configure accelerate first
   accelerate config

   # Launch training
   accelerate launch --multi_gpu --num_processes=2 train_diffusion.py
   ```

4. **Wandb Integration:**
   Set your wandb entity in config or disable:
   ```yaml
   training:
     use_wandb: false  # Disable if not using wandb
   ```

## References

- **3D-MedDiffusion:** https://github.com/DiffusionMRIPreprocessing/3D-MedDiffusion
- **HuggingFace Diffusers:** https://github.com/huggingface/diffusers
- **DDPM Paper:** https://arxiv.org/abs/2006.11239
- **Latent Diffusion:** https://arxiv.org/abs/2112.10752

## Troubleshooting

### If metrics are too high (SSIM > 0.99, PSNR > 60):
→ Check `data_range` in config (should be 2.0, not 2000.0)

### If both GPUs process same samples:
→ Use unprepared dataloader in `validate_with_metrics()`

### If training is slow or OOM:
→ Check batch_size (should be 32, not 512)
→ Enable gradient_accumulation_steps

### If model performance is poor:
→ Ensure model is scaled up (128 channels, not 64)
→ Check learning rate (should be 2e-4, not 5e-5)

---

**Recovery Date:** 2025-01-12
**Status:** Utilities complete, training components need reconstruction
**Priority:** Implement/restore `diffusion_unet.py` and `train_diffusion.py`
