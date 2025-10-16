#!/bin/bash
# Precompute VQ-AE latents for CT scans
#
# This script encodes raw CT scans into latent space using a pre-trained VQ-AE.
# The latents are cached to disk for fast training of the diffusion model.
#
# Data location: /data1/peijia/ct/processed/ct_pairs/ (433 pairs)
#
# Usage:
#   ./scripts/run_precompute_latents.sh

set -e  # Exit on error

# Configuration (hardcoded paths)
DATA_DIR="/data1/peijia/ct/processed/ct_pairs"
OUTPUT_DIR="./latents_cache_v2"
CONFIG_FILE="config_diffusion_.yaml"

# Activate conda environment
if [ ! -z "${CONDA_DEFAULT_ENV}" ]; then
    echo "Using conda environment: ${CONDA_DEFAULT_ENV}"
else
    echo "Activating bioagent conda environment..."
    eval "$(conda shell.bash hook)"
    conda activate bioagent
fi

echo "========================================================================"
echo "VQ-AE Latent Precomputation"
echo "========================================================================"
echo "Data directory: ${DATA_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Config file: ${CONFIG_FILE}"
echo ""

# Check number of CT pairs
NUM_LD=$(ls "${DATA_DIR}/low_dose" 2>/dev/null | wc -l)
NUM_HD=$(ls "${DATA_DIR}/high_dose" 2>/dev/null | wc -l)
echo "Found ${NUM_LD} low-dose and ${NUM_HD} high-dose CT scans"
echo ""

# Check if data directory exists
if [ ! -d "${DATA_DIR}" ]; then
    echo "ERROR: Data directory not found: ${DATA_DIR}"
    echo ""
    echo "Expected directory structure:"
    echo "  ${DATA_DIR}/"
    echo "    low_dose/"
    echo "      sample_000.nii.gz"
    echo "      sample_001.nii.gz"
    echo "      ..."
    echo "    high_dose/"
    echo "      sample_000.nii.gz"
    echo "      sample_001.nii.gz"
    echo "      ..."
    echo ""
    echo "NOTE: Data should be at /data1/peijia/ct/processed/ct_pairs/"
    exit 1
fi

# Check if config exists
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "ERROR: Config file not found: ${CONFIG_FILE}"
    exit 1
fi

# Run precomputation
echo "Starting precomputation..."
echo ""

# python precompute_latents.py \
#     --data_dir "${DATA_DIR}" \
#     --latent_cache_dir "${OUTPUT_DIR}" \
#     --config "${CONFIG_FILE}" \
#     --train_split 0.8 \
#     --device cuda \
#     --batch_size 32

# python precompute_latents.py \
#  --data_dir /data1/peijia/ct/processed/ct_pairs \
#  --latent_cache_dir latents_cache_v2 \
#  --vae_checkpoint ~/projects/BioAgent/3D-MedDiffusion/checkpoints/3DMedDiffusion_checkpoints/PatchVolume_8x_s2.ckpt \
#  --device cuda \
#  --config config_diffusion.yaml \
#  --target_shape 200 256 256 \
#  --batch_size 4

python precompute_latents.py \
 --data_dir /data1/peijia/ct/processed/ct_pairs \
 --latent_cache_dir latents_cache_v3 \
 --vae_checkpoint ~/projects/BioAgent/3D-MedDiffusion/checkpoints/3DMedDiffusion_checkpoints/PatchVolume_8x_s2.ckpt \
 --device cuda \
 --config config_diffusion_512.yaml \
 --target_shape 200 512 512 \
 --batch_size 1

echo ""
echo "========================================================================"
echo "Precomputation completed!"
echo "========================================================================"
echo ""
echo "Latents saved to: ${OUTPUT_DIR}"
echo ""
echo "Next steps:"
echo "  1. Verify the latents:"
echo "     python data/latent_dataset.py"
echo ""
echo "  2. Start training:"
echo "     ./scripts/run_train_diffusion.sh ${CONFIG_FILE}"
echo ""
