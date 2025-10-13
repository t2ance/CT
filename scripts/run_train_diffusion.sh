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
fi

CMD=()

if [ ${NUM_GPUS} -eq 1 ]; then
    echo "Single GPU detected, running without FSDP..."
    CMD=(python train_diffusion.py --config "${CONFIG_FILE}")
else
    echo "Multiple GPUs detected, using FSDP..."

    if [ ! -f "${ACCELERATE_CONFIG}" ]; then
        echo "WARNING: Accelerate config not found: ${ACCELERATE_CONFIG}"
        echo "Using default accelerate settings..."
        CMD=(accelerate launch --multi_gpu --num_processes=${NUM_GPUS} --mixed_precision=fp16 train_diffusion.py --config "${CONFIG_FILE}")
    else
        echo "Using accelerate config: ${ACCELERATE_CONFIG}"
        CMD=(accelerate launch --config_file "${ACCELERATE_CONFIG}" train_diffusion.py --config "${CONFIG_FILE}")
    fi
fi
# accelerate launch --config_file accelerate_config_fsdp.yaml train_diffusion.py --config config_diffusion.yaml

if [ ${#CMD[@]} -eq 0 ]; then
    echo "ERROR: No command constructed for training."
    exit 1
fi

LOG_DIR="./log"
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"
PID_FILE="${LOG_DIR}/train_${TIMESTAMP}.pid"

echo ""
echo "Launching training with nohup..."
echo "  Command : ${CMD[*]}"
echo "  Log file: ${LOG_FILE}"

nohup "${CMD[@]}" > "${LOG_FILE}" 2>&1 &
PID=$!
echo "${PID}" > "${PID_FILE}"

echo "Training process started with PID ${PID} (saved to ${PID_FILE})."
echo "View live logs with: tail -f ${LOG_FILE}"
echo ""
echo "========================================================================"
echo "Training launched in background."
echo "========================================================================"
