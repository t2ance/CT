CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup accelerate launch \
    --config_file accelerate_config_single_gpu.yaml \
    train_diffusion.py \
    --config config_diffusion_128_v1.yaml \
    > "./tmp/log/train_128_v1.log" 2>&1 &
PID=$!
echo "${PID}" > "./tmp/log/train_128_v1.pid"
