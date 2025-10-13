**Goal & context**
- Quantify how many attention layers our `DiffusionUNet3D` uses and whether they are a performance hotspot.  
- Evaluate FlashAttention v2 (Dao et al.) as a drop-in optimization and draft a reversible plugin.  
- Document feasibility, risks, and next steps for optionally enabling FlashAttention during training/inference.

**TL;DR**
- The current config instantiates 12 self-attention blocks, all operating on ~832 tokens (13×8×8) with 512 channels/head dim 64 (`attention_levels=[1]`, `num_blocks=5`). See `research/flash_attention_v2/attention_profile.txt`.  
- FlashAttention v2 supports bf16/fp16 CUDA workloads with head dim ≤256 and provides installation guidance via `pip install flash-attn --no-build-isolation` (`flash-attention/README.md`).  
- The new factory (`model_factory.py`) builds `DiffusionUNet3D` and, when `model.use_flash_attention: true`, swaps 12 attention blocks for the Flash wrapper (`research/flash_attention_v2/flash_attention_plugin.py`) while preserving CPU/fp32 fallback. Recommended as an optional optimization on Ampere/Ada/Hopper GPUs.

**Integration options**
- **Adapter (recommended)**  
  1. Install FlashAttention v2 in the training env (`pip install flash-attn --no-build-isolation`).  
  2. Import and run `apply_flash_attention(model)` after model construction; this swaps 12 attention blocks for Flash-enabled wrappers while keeping parameters intact.  
  3. Guard runtime with `x.is_cuda` and dtype checks (already in plugin) so CPU or fp32 paths remain unchanged.
- **Partial**  
  - Replace only the bottleneck block via the same helper if you want to validate gains on the most memory-bound layer first.  
  - Combine with PyTorch 2.x SDPA (`torch.nn.functional.scaled_dot_product_attention`) for fallback when FlashAttention isn’t available.  
- **Full**  
  - Migrate to Hugging Face Diffusers’ UNet3D implementation, which already integrates FlashAttention for many configs. This would require porting checkpoints and conditioning logic, so keep as a long-term option.

**Minimal examples**
```python
from model_factory import build_diffusion_model

model = build_diffusion_model(cfg["model"], verbose=True)
```
```bash
# Optional smoke test (falls back to naive kernels on CPU)
PYTHONPATH=. python research/flash_attention_v2/smoke_test.py
```
Expected output: `Max deviation after wrapping: 0.000000`.

**Direct migration notes**
- Added files:  
  - `model_factory.py`: centralizes diffusion model construction and FlashAttention enablement.  
  - `research/flash_attention_v2/analyze_attention_blocks.py`: utility used to profile attention shapes/counts.  
  - `research/flash_attention_v2/attention_profile.txt`: captured counts showing 12 attention blocks at 832 tokens each.  
  - `research/flash_attention_v2/flash_attention_plugin.py`: adapter exposing `FlashAttentionBlock3D` and `apply_flash_attention`.  
  - `research/flash_attention_v2/smoke_test.py`: CPU regression check for the wrapper.  
  - `tests/test_flash_attention_integration.py`: lightweight tests validating wrapping and output parity.
- Revert by deleting `model_factory.py` plus the `research/flash_attention_v2/` helpers and skipping the `use_flash_attention` flag.

**Alternatives considered**
- **PyTorch native SDPA kernels**: already available in 2.0+, but v2 FlashAttention still advertises higher throughput on Ampere/Ada GPUs for long sequences.  
- **xFormers memory-efficient attention**: broader dtype/backend support, but adds another heavy dependency and lacks the explicit bf16 focus highlighted in FlashAttention v2 docs.

**Risks & mitigations**
- Requires CUDA 12.x and Ampere/Ada/Hopper GPUs; compilation can take minutes and needs `ninja` (`flash-attention/README.md`). Mitigation: document install flags (`--no-build-isolation`, `MAX_JOBS`).  
- Kernels only accept fp16/bf16 CUDA tensors (`flash_attn/modules/mha.py` assertions). Plugin falls back when conditions aren’t met.  
- Head dim must stay ≤256 (`flash-attention/README.md`); current config uses 64, so safe. Monitor future config changes.  
- Additional dependency increases build complexity; keep optional and wrap in try/except.

**Open questions**
- Do production runs always use bf16/fp16 on supported GPUs? If not, we may need runtime knobs to disable the plugin.  
- Should we expose a CLI/config flag (`use_flash_attention`) to avoid touching training scripts?  
- Is there appetite for benchmarking before/after to quantify wall-clock gains on 12 medium-length attention blocks?

**Citations/links**
- `config_diffusion.yaml` (model.attention_levels, num_blocks, dtype settings).  
- `research/flash_attention_v2/attention_profile.txt` (local profiling output).  
- FlashAttention installation & hardware requirements: `research/flash_attention_v2/flash-attention/README.md`.  
- FlashAttention kernel dtype/CUDA assertions: `research/flash_attention_v2/flash-attention/flash_attn/modules/mha.py`.
