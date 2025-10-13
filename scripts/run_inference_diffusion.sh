#!/bin/bash
# Inference launcher for CT Latent Diffusion Model
#
# Runs inference with trained diffusion model to convert low-dose CT to high-dose CT.
#
# Usage:
#   ./scripts/run_inference_diffusion.sh [CONFIG_FILE] [CHECKPOINT] [NUM_SAMPLES]
#
# Example:
#   ./scripts/run_inference_diffusion.sh config_diffusion.yaml outputs/checkpoints/best.pth 10

set -e  # Exit on error

# Configuration
CONFIG_FILE="${1:-config_diffusion.yaml}"
CHECKPOINT="${2:-outputs/checkpoints/best.pth}"
NUM_SAMPLES="${3:-}"

# Activate conda environment
if [ ! -z "${CONDA_DEFAULT_ENV}" ]; then
    echo "Using conda environment: ${CONDA_DEFAULT_ENV}"
else
    echo "Activating bioagent conda environment..."
    eval "$(conda shell.bash hook)"
    conda activate bioagent
fi

echo "========================================================================"
echo "CT Latent Diffusion Inference"
echo "========================================================================"
echo "Config file: ${CONFIG_FILE}"
echo "Checkpoint: ${CHECKPOINT}"
echo ""

# Check if files exist
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "ERROR: Config file not found: ${CONFIG_FILE}"
    exit 1
fi

# Check checkpoint size (should be >100 MB)
if [ -f "${CHECKPOINT}" ]; then
    CHECKPOINT_SIZE=$(stat -c%s "${CHECKPOINT}" 2>/dev/null || stat -f%z "${CHECKPOINT}" 2>/dev/null || echo 0)
    CHECKPOINT_SIZE_MB=$((CHECKPOINT_SIZE / 1024 / 1024))

    if [ ${CHECKPOINT_SIZE_MB} -lt 100 ]; then
        echo "WARNING: Checkpoint file is small (${CHECKPOINT_SIZE_MB} MB)"
        echo "This may indicate corruption. Expected >100 MB for diffusion model."
        echo ""
    fi
else
    echo "WARNING: Checkpoint not found: ${CHECKPOINT}"
    echo "Will attempt to find alternative checkpoint..."
    echo ""
fi

# Build command
CMD="python inference_diffusion.py --config ${CONFIG_FILE} --checkpoint ${CHECKPOINT}"

if [ ! -z "${NUM_SAMPLES}" ]; then
    CMD="${CMD} --num_samples ${NUM_SAMPLES}"
fi

# Run inference
echo "Running inference..."
echo "Command: ${CMD}"
echo ""
${CMD}

echo ""
echo "========================================================================"
echo "Inference completed!"
echo "========================================================================"
