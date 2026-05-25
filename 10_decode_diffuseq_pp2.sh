#!/bin/bash

cd "$(dirname "$0")"

MODEL_DIR="../diffusion_models/PUT_YOUR_MODEL_FOLDER_HERE"

python run_decode.py \
  --model_dir $MODEL_DIR \
  --seed 101 \
  --step 500 \
  --bsz 8 \
  --split test \
  --pattern ema