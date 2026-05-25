#!/bin/bash

cd "$(dirname "$0")"

python -m torch.distributed.launch \
  --nproc_per_node=1 \
  --master_port=12233 \
  --use_env run_train.py \
  --diff_steps 500 \
  --lr 0.0001 \
  --learning_steps 20000 \
  --save_interval 2000 \
  --seed 102 \
  --noise_schedule sqrt \
  --hidden_dim 128 \
  --bsz 64 \
  --microbatch 4 \
  --dataset cnndm \
  --data_dir ../datasets/CNNDM \
  --vocab bert \
  --seq_len 256 \
  --schedule_sampler lossaware \
  --notes cnndm_pp2 \
  --use_fp16 False