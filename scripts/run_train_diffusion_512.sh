#!/bin/bash
# Simplified training launcher for CT Latent Diffusion Model

set -e

# Setup logging
LOG_DIR="./tmp/log"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_512.log"
PID_FILE="${LOG_DIR}/train_512.pid"

# Single accelerate launch command with explicit conda environment
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=2,3 nohup \
    /home/peijia/miniconda3/envs/bioagent/bin/accelerate launch \
    --config_file accelerate_config_fsdp.yaml \
    train_diffusion.py \
    --config config_diffusion_512.yaml \
    > "${LOG_FILE}" 2>&1 &
PID=$!
echo "${PID}" > "${PID_FILE}"

echo "Training started with PID: ${PID}"
echo "Logs: ${LOG_FILE}"
echo "To monitor: tail -f ${LOG_FILE}"
