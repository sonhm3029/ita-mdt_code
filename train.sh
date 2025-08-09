#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
NUM_GPUS=2

DATA_PATH=DATA
export OPENAI_LOGDIR=experiments/ITA-MDT

LR=1e-4
BATCH_SIZE=6
SAVE_INTERVAL=100000
MASTER_PORT=29971

SCRIPT_PATH=image_train.py
MODEL_FLAGS="--image_size 512 --vit_image_size 224 --mask_ratio 0.30 --decode_layer 4 --model MDT_IVTON_XL"
DIFFUSION_FLAGS="--diffusion_steps 1000"
TRAIN_FLAGS="--batch_size $BATCH_SIZE --save_interval $SAVE_INTERVAL"

python -m torch.distributed.launch \
    --master_port=$MASTER_PORT \
    --nproc_per_node=$NUM_GPUS \
    $SCRIPT_PATH \
    --data_dir $DATA_PATH \
    --work_dir $OPENAI_LOGDIR \
    --lr $LR \
    --n_gpus=$NUM_GPUS \
    $MODEL_FLAGS \
    $DIFFUSION_FLAGS \
    $TRAIN_FLAGS \
    # --resume_checkpoint experiments/ita-mdt_weights/model2000000.pt
