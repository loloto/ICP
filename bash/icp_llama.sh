
#!/bin/bash

python icp_llama.py meta-llama/Llama-2-7b-hf c4 \
    --sparsity 0.7 \
    --pruning_mode sparsegpt \
    --tune_epoch 1 \
    --tune_lr 0.0005 \
    --alpha 0.691 \
    --beta 0 \
    --log_path ./path/to/your/log/root