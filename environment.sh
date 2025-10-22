#!/bin/bash

# =======================================
# ITA-MDT Environment Setup Script
# =======================================
# IMPORTANT:
# This script assumes 'libopenmpi-dev' is already installed for 'mpi4py'
# If not, run the following with sudo before continuing:
#   sudo apt-get update
#   sudo apt-get install libopenmpi-dev

ENV_NAME=ITA-MDT
conda create -n $ENV_NAME python=3.10 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

conda install -y nvidia/label/cuda-12.1.0::cuda-toolkit
conda install -y pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia

pip install -e .
pip install ninja git+https://github.com/sail-sg/Adan.git
pip install mpi4py==3.1.6 diffusers==0.34.0 timm==0.9.16 einops==0.8.0 \
           numpy==1.23.5 protobuf==3.20.3 opencv-python==4.9.0.80 \
           albumentations==0.5.2 scikit-image==0.16.2 scipy==1.4.1 \
           tensorflow==2.2.0 accelerate==1.0.1 tqdm

pip install "git+https://github.com/facebookresearch/detectron2.git"