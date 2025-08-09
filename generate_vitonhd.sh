#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

DATA_DIR=DATA
OUTPUT_DIR=results
MODEL_PATH=/path/to/ITA-MDT/experiments/ita-mdt_weights/ema_0.9999_2000000.pt
UNPAIR=false
BATCH_SIZE=18

NUM_SAMPLING_STEPS=30
CFG_SCALE=2.0
POW_SCALE=1.0
IMAGE_SIZE=512
VIT_IMG_SIZE=224
python generate_vitonhd.py \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --model_path $MODEL_PATH \
    --image_size $IMAGE_SIZE \
    --vit_img_size $VIT_IMG_SIZE \
    --num_sampling_steps $NUM_SAMPLING_STEPS \
    --cfg_scale $CFG_SCALE \
    --pow_scale $POW_SCALE \
    --batch_size $BATCH_SIZE \
    --unpair $UNPAIR
