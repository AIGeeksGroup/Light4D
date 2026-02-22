#!/bin/bash

# ============ GPU Configuration ============
export CUDA_VISIBLE_DEVICES="0"
# ===========================================

echo "========================================"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Mode: Light4D Relighting"
echo "========================================"

python 4D_relighting.py \
    --color_video "/path/to/color.mp4" \
    --mask_video "/root/autodl-tmp/light4d_processed_512_recon/text/kling_video1_cat/mask_30.mp4" \
    --output_video "/path/to/mask.mp4" \
    --sd_model "models/realistic-vision-v51" \
    --ic_light_model "models/iclight_sd15_fc.safetensors" \
    --enable_relight \
    --light_direction LEFT \
    --relight_prompt "Pink Neon Lights, cinematic quality" \
    --gamma 0.7 \
    --seed 42 \
    --num_inference_steps 30 \
    --num_frames 49 \
    --height 384 \
    --width 384 \
    --device "cuda" \
    --save_intermediate_steps \
    --intermediate_dir "result/intermediate_fusion" \
    --use_progressive_fusion