import matplotlib.pyplot as plt
import os
import json
from tqdm import tqdm
# 设置matplot的颜色
import matplotlib as mpl
from cycler import cycler
mpl.rcParams['axes.prop_cycle'] = cycler(color='bcrygmk')

# 设置绘图参数 #########################################
TASK='rte'
FIG_SAVE_DIR = f'output/figs/rte/n-order_cmp_loss'
if os.path.exists(FIG_SAVE_DIR) is False:
    os.makedirs(FIG_SAVE_DIR)
print("保存图片的文件夹名称：", FIG_SAVE_DIR.split('/')[-1])

METRIC='loss'

ORDER_1_FILE_PATH = f'output/rte/enc_sim_task-rte_metric-loss_lr-1e-06_lambda-0.0_wd-0.0005_bs-128_train-sample-nums-200_test-sample-nums-200_seed-42_step_thres-None_frozen-True_use_initial-True/pred_and_gt_loss_trajectories.out'
ORDER_2_FILE_PATH = "output/rte/norder_enc_sim_task-rte_metric-loss_lr-1e-06_lambda-0.0_wd-0.0005_bs-128_train-sample-nums-200_test-sample-nums-200_seed-42_step_thres-None_max_epoch-150_order_n-2_frozen-True_use_initial-True_concate-False/last/pred_and_gt_loss_trajectories.out"
ORDER_3_FILE_PATH = "output/rte/norder_enc_sim_task-rte_metric-loss_lr-1e-06_lambda-0.0_wd-0.0005_bs-128_train-sample-nums-200_test-sample-nums-200_seed-42_step_thres-None_max_epoch-150_order_n-3_frozen-True_use_initial-True_concate-False/last/pred_and_gt_loss_trajectories.out"
ORDER_5_FILE_PATH = "output/rte/norder_enc_sim_task-rte_metric-loss_lr-1e-06_lambda-0.0_wd-0.0005_bs-128_train-sample-nums-200_test-sample-nums-200_seed-42_step_thres-None_max_epoch-150_order_n-5_frozen-True_use_initial-True_concate-False/last/pred_and_gt_loss_trajectories.out"
ORDER_10_FILE_PATH = "output/rte/norder_enc_sim_task-rte_metric-loss_lr-1e-06_lambda-0.0_wd-0.0005_bs-128_train-sample-nums-200_test-sample-nums-200_seed-42_step_thres-None_max_epoch-150_order_n-10_frozen-True_use_initial-True_concate-False/last/pred_and_gt_loss_trajectories.out"

#########################################################

order_1_list = []
order_2_list = []
order_3_list = []
order_5_list = []
order_10_list = []

with open(ORDER_1_FILE_PATH, 'r', encoding='utf-8') as f:
    for line in f.readlines():
        line = json.loads(line)
        order_1_list.append(line)
with open(ORDER_2_FILE_PATH, 'r', encoding='utf-8') as f:
    for line in f.readlines():
        line = json.loads(line)
        order_2_list.append(line)
with open(ORDER_3_FILE_PATH, 'r', encoding='utf-8') as f:
    for line in f.readlines():
        line = json.loads(line)
        order_3_list.append(line)
with open(ORDER_5_FILE_PATH, 'r', encoding='utf-8') as f:
    for line in f.readlines():
        line = json.loads(line)
        order_5_list.append(line)
with open(ORDER_10_FILE_PATH, 'r', encoding='utf-8') as f:
    for line in f.readlines():
        line = json.loads(line)
        order_10_list.append(line)

for test_sample_id, (order_1s, order_2s, order_3s, order_5s, order_10s) in tqdm(enumerate(zip(order_1_list, order_2_list, order_3_list, order_5_list, order_10_list))):
    for run_id, (order_1, order_2, order_3, order_5, order_10) in enumerate(zip(order_1s, order_2s, order_3s, order_5s, order_10s)):
# for test_sample_id, (origins, encs) in tqdm(enumerate(zip(origin_list, enc_sim_list))):
#     for run_id, (origin, enc) in enumerate(zip(origins, encs)):
        # print('debug')
        # plt.plot(origin[])
        # 绘制gt曲线
        plt.plot(order_1['step'], order_1['gt_loss'], label='Ground Truth', linewidth=1.5, color='#0066fe')
        # 绘制origin曲线
        plt.plot(order_1['step'], order_1['pred_loss'], label='1-th order', linewidth=1.0, color='#dbdbdb', marker='X', markevery=6)
        plt.plot(order_2['step'], order_2['pred_loss'], label='2-th order', linewidth=1.0, color='#909090', marker='P', markevery=6)
        plt.plot(order_3['step'], order_3['pred_loss'], label='3-th order', linewidth=1, color='#59d65a')
        plt.plot(order_5['step'], order_5['pred_loss'], label='5-th order', linewidth=1, color='lime')
        plt.plot(order_10['step'], order_10['pred_loss'], label='10-th order', linewidth=1, color='thistle')
        
        plt.legend()
        plt.grid()
        plt.title(f'test_sample_id: {test_sample_id}, run_id: {run_id}')
        plt.xlabel('Training Steps')
        plt.ylabel(f'{METRIC}')
        plt.savefig(f'{FIG_SAVE_DIR}/test_sample_{test_sample_id}_run_{run_id}.png')
        plt.clf()
