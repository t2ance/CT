#!/bin/bash
# Simplified training launcher for CT Latent Diffusion Model

set -e

# Activate conda environment if needed
if [ -z "${CONDA_DEFAULT_ENV}" ]; then
    eval "$(conda shell.bash hook)"
    conda activate bioagent
fi

# Setup logging
LOG_DIR="./log"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_512.log"
PID_FILE="${LOG_DIR}/train_512.pid"

# Single accelerate launch command
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=2,3 nohup accelerate launch \
    --config_file accelerate_config_fsdp.yaml \
    train_diffusion.py \
    --config config_diffusion_512.yaml \
    > "${LOG_FILE}" 2>&1 &
PID=$!
echo "${PID}" > "${PID_FILE}"
