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
LOG_FILE="${LOG_DIR}/train_128.log"
PID_FILE="${LOG_DIR}/train_128.pid"

# Single accelerate launch command
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0,1 nohup accelerate launch \
#     --config_file accelerate_config_fsdp.yaml \
#     train_diffusion.py \
#     --config config_diffusion_128.yaml \
#     > "${LOG_FILE}" 2>&1 &
cuda_visible_devices=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup accelerate launch \
    --config_file accelerate_config_single_gpu.yaml \
    train_diffusion.py \
    --config config_diffusion_128.yaml \
    > "${LOG_FILE}" 2>&1 &
PID=$!
echo "${PID}" > "${PID_FILE}"
