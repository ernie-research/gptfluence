from dataset.simfluence_dataset import SimfluenceDataset
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import json
from tqdm import tqdm

from scipy import stats

# 导入Simulator
from model.Simulator import Simulator
from model.XlmrSimulator import XlmrSimulator
from model.VectorSimulator import VectorSimulator

from utils.eval import eval_simulator

import logging
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("Simfuence.test")

data_paths = [
        'runs/rte/output_rte_bs-4_shot-200_sample-128_lr-2e-6_weight-decay-0.001_epoch-3_seed-28',
        'runs/rte/output_rte_bs-4_shot-200_sample-128_lr-2e-6_weight-decay-0.001_epoch-3_seed-29',
        'runs/rte/output_rte_bs-4_shot-200_sample-128_lr-2e-6_weight-decay-0.001_epoch-3_seed-30',
        'runs/rte/output_rte_bs-4_shot-200_sample-128_lr-2e-6_weight-decay-0.001_epoch-3_seed-31',
        'runs/rte/output_rte_bs-4_shot-200_sample-128_lr-2e-6_weight-decay-0.001_epoch-3_seed-32',
]

test_example_nums = 277
train_example_nums = 200
sim_name = "vec_sim" # "vec_sim" "original"
dataset_name = "rte"
check_point_path = "/root/paddlejob/workspace/liuqingyi01/code/Simfluence/output/vec_sim_task-output_rte_bs-4_shot-200_sample-128_lr-2e-6_weight-decay-0.001_epoch-3_lr-0.001_lambda-0.0_bs-128_train-sample-nums-200_test-sample-nums-277_seed-42_step_thres-None/checkpoint-288.pt"
save_dir = "/root/paddlejob/workspace/liuqingyi01/code/Simfluence/output/vec_sim_task-output_rte_bs-4_shot-200_sample-128_lr-2e-6_weight-decay-0.001_epoch-3_lr-0.001_lambda-0.0_bs-128_train-sample-nums-200_test-sample-nums-277_seed-42_step_thres-None"
hyper_parameter = 0.

from train import (
    DATASET,
    DATASET_ADDITIONAL_ARGS,
    SIMULATORS,
    SIMULATR_ADDIONAL_ARGS,
    INPUT_ADDITIONAL_KEYS,
)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   


test_dataset = SimfluenceDataset(data_paths, test_example_nums=test_example_nums, is_train=False, step_thres=None)
# 加载数据集
dataset = DATASET[dataset_name]
if dataset is None:
        logger.warning("dataset is None, use default dataset")
else:
    test_dataset = dataset(
        test_dataset,
        is_train=False,
        **DATASET_ADDITIONAL_ARGS[dataset_name]
    )


# 加载simulator
model = SIMULATORS[sim_name](train_example_nums=train_example_nums, hyper_parameter=hyper_parameter, test_example_nums=test_example_nums, **SIMULATR_ADDIONAL_ARGS[sim_name])
model.load_state_dict(torch.load(check_point_path))
model.to(device)

print("\n开始测试...")
input_kwargs_keys = INPUT_ADDITIONAL_KEYS[sim_name]
results = eval_simulator(test_dataset, model, device, input_kwargs_keys)
print("测试集mse:", results[0])

# 计算last-step spearman correlation
last_step_pred = {
     'pred': [],
     'gt': []
}
for test_sample_id, r in results[-1].items():
    for trajectory in r:
         last_step_pred['gt'].append(trajectory['gt_loss'][-1])
         last_step_pred['pred'].append(trajectory['pred_loss'][-1])

last_step_spearman = stats.spearmanr(last_step_pred['gt'], last_step_pred['pred']).statistic
print("测试集last step spearman:", last_step_spearman)
     
# 保存结果
print("\n保存结果...")
with open(os.path.join(save_dir, 'pred_and_gt_loss_trajectories.out'), 'w') as f:
    for test_sample_id, r in results[-1].items():
        r = json.dumps(r)
        print(r, file=f)
# 画图
print("\n画图...")
fig_save_path = os.path.join(save_dir, 'figs')
if not os.path.exists(fig_save_path):
    os.mkdir(fig_save_path)
for test_sample_id, r in tqdm(results[-1].items()):
    for i, item in enumerate(r):
        plt.plot(item['step'], item['gt_loss'], label='gt')
        plt.plot(item['step'], item['pred_loss'], label='predict')
        plt.xlabel('step')
        plt.ylabel('loss')
        plt.title('test sample id:' + str(test_sample_id) + '-' + str(i))
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(fig_save_path, str(test_sample_id) + '-' + str(i) + '.png'))
        plt.clf()

# plt.plot(results[-1][0]['step'], results[-1][0]['pred_loss'], label='predict')

# path = "/root/paddlejob/workspace/liuqingyi01/code/Simfluence/runs/rte/output_rte_bs-4_shot-200_sample-128_lr-2e-6_weight-decay-0.001_epoch-3_seed-32/all_loss_trajectory.out"
# with open(path, 'r') as f:
#     lines = f.readlines()
#     for line in lines[:1]:
#         line = json.loads(line)
#         sample_id = line['id']
#         steps = []
#         losses = []
#         for p in line['loss_trajectory']:
#             step = p['step']
#             loss = p['loss']
#             # if step < 49:
#             #     continue
#             steps.append(step)
#             losses.append(loss)
#         plt.plot(steps, losses, label='gt')
#         plt.xlabel('step')
#         plt.ylabel('loss')
#         plt.title(sample_id)
#         plt.legend()
#         plt.grid()
#         plt.show()
#         save_path = os.path.join('./', str(sample_id) + 'step_thresh.png')
#         print("save path is ", save_path)
#         plt.savefig(save_path)
#         plt.clf()

# print(results)
print('done')
