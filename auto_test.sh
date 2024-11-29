#!/bin/bash

set -e

cuda=$1
if [ "$cuda" = "" ]; then
    echo "CUDA_VISIBLE_DEVICES is not set"
    exit 1
fi
export CUDA_VISIBLE_DEVICES=$cuda

python test.py\
    --test_example_nums 200\
    --train_example_nums 200\
    --task rte\
    --metric loss\
    --sim_name enc_sim\
    --dataset_name rte\
    --checkpoint_path output/rte/enc_sim_task-rte_metric-loss_lr-0.0001_lambda-0.0_wd-1e-05_bs-128_train-sample-nums-200_test-sample-nums-277_seed-42_step_thres-None_max_epoch-1_frozen-True_use_initial-True_concate-False/best-checkpoint.pt\
    --save_dir output/rte/enc_sim_task-rte_metric-loss_lr-0.0001_lambda-0.0_wd-1e-05_bs-128_train-sample-nums-200_test-sample-nums-277_seed-42_step_thres-None_max_epoch-1_frozen-True_use_initial-True_concate-False\
    --test_example_start_id 0\
    --test_example_end_id 199