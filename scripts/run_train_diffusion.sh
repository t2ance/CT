#!/bin/bash
# Training launcher for CT Latent Diffusion Model
#
# This script automatically detects GPUs and launches training with appropriate settings.
# Uses HuggingFace Accelerate with FSDP for multi-GPU training.
#
# Usage:
#   ./scripts/run_train_diffusion.sh [CONFIG_FILE]
#
# Example:
#   ./scripts/run_train_diffusion.sh config_diffusion.yaml

set -e  # Exit on error

# Configuration
CONFIG_FILE="${1:-config_diffusion.yaml}"
ACCELERATE_CONFIG="accelerate_config_fsdp.yaml"

# Activate conda environment
if [ ! -z "${CONDA_DEFAULT_ENV}" ]; then
    echo "Using conda environment: ${CONDA_DEFAULT_ENV}"
else
    echo "Activating bioagent conda environment..."
    eval "$(conda shell.bash hook)"
    conda activate bioagent
fi

echo "========================================================================"
echo "CT Latent Diffusion Training"
echo "========================================================================"
echo "Config file: ${CONFIG_FILE}"
echo ""

# Check if config exists
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "ERROR: Config file not found: ${CONFIG_FILE}"
    exit 1
fi

# Detect number of GPUs
NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)

if [ ${NUM_GPUS} -eq 0 ]; then
    echo "ERROR: No CUDA GPUs detected!"
    exit 1
elif [ ${NUM_GPUS} -eq 1 ]; then
    echo "Single GPU detected, running without FSDP..."
    python train_diffusion.py --config "${CONFIG_FILE}"
else
    echo "Multiple GPUs detected, using FSDP..."

    # Check if accelerate config exists
    if [ ! -f "${ACCELERATE_CONFIG}" ]; then
        echo "WARNING: Accelerate config not found: ${ACCELERATE_CONFIG}"
        echo "Using default accelerate settings..."
        accelerate launch \
            --multi_gpu \
            --num_processes=${NUM_GPUS} \
            --mixed_precision=fp16 \
            train_diffusion.py \
            --config "${CONFIG_FILE}"
    else
        echo "Using accelerate config: ${ACCELERATE_CONFIG}"
        accelerate launch \
            --config_file "${ACCELERATE_CONFIG}" \
            train_diffusion.py \
            --config "${CONFIG_FILE}"
    fi
fi

echo ""
echo "========================================================================"
echo "Training completed!"
echo "========================================================================"
