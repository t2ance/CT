CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup accelerate launch \
    --config_file accelerate_config_single_gpu.yaml \
    train_diffusion.py \
    --config config_diffusion_256.yaml \
    > "./tmp/log/train_256.log" 2>&1 &
PID=$!
echo "${PID}" > "./tmp/log/train_256.pid"
