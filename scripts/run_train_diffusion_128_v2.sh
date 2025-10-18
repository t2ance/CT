CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup accelerate launch \
    --config_file accelerate_config_single_gpu.yaml \
    train_diffusion.py \
    --config config_diffusion_128_v2.yaml \
    > "./tmp/log/train_128_v2.log" 2>&1 &
PID=$!
echo "${PID}" > "./tmp/log/train_128_v2.pid"
