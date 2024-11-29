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
    --task webnlg\
    --metric rougeL\
    --sim_name enc_sim\
    --dataset_name webnlg\
    --checkpoint_path output/webnlg/enc_sim_task-webnlg_metric-bleu_lr-0.0001_lambda-0.0_wd-1e-05_bs-128_train-sample-nums-200_test-sample-nums-200_seed-42_step_thres-None_max_epoch-1_frozen-True_use_initial-True_concate-False/best-checkpoint.pt\
    --save_dir output/webnlg/enc_sim_task-webnlg_metric-bleu_lr-0.0001_lambda-0.0_wd-1e-05_bs-128_train-sample-nums-200_test-sample-nums-200_seed-42_step_thres-None_max_epoch-1_frozen-True_use_initial-True_concate-False\
    --test_example_start_id 0\
    --test_example_end_id 99