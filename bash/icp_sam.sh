#!/bin/bash

python icp_sam.py \
    --sparsity 0.7 \
    --interaction_mode point \
    --checkpoint vit_b \
    --val_dataset coco \
    --tune_gamma 0.9 \
    --tune_epoch 8 \
    --tune_lr 0.0005 \
    --alpha 0.85 \
    --beta 0.0 \
    --nsamples 128 \
    --log_path ./path/to/your/log/root \
