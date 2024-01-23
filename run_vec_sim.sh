#!/bin/bash

set -e

cuda=$1
if [ "$cuda" = "" ]; then
    echo "CUDA_VISIBLE_DEVICES is not set"
    exit 1
fi
export CUDA_VISIBLE_DEVICES=$cuda

python train.py --sim_name vec_sim \
    --task output_rte_bs-4_shot-200_sample-128_lr-2e-6_weight-decay-0.001_epoch-3 \
    --output_dir ./output \
    --max_epoch 300 \
    --train_bs 128 \
    --train_example_nums 200 \
    --test_example_nums 277 \
    --seed 42 \
    --hyper_parameter 0. \
    --lr 0.0001 \
    --valid_epoch_interval 1 \
    --save_epoch_interval 1 \
    --valid_num 2 \
    --test_num 0 \
    --weight_decay 1e-5