#!/bin/bash

python icp_opt.py facebook/opt-125m c4 \
    --sparsity 0.7 \
    --tune_gamma 0.87 \
    --tune_epoch 12 \
    --tune_lr 0.0001 \
    --alpha 0.691 \
    --beta 0.2 \
    --log_path ./path/to/your/log/root