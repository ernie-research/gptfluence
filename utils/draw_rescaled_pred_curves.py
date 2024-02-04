import matplotlib.pyplot as plt
import os
import json
from tqdm import tqdm
# 设置matplot的颜色
import matplotlib as mpl
from cycler import cycler
mpl.rcParams['axes.prop_cycle'] = cycler(color='bcrygmk')

# 设置绘图参数 #########################################
FIG_SAVE_DIR = 'test/figs/flan_tracincp_sst2_rescaled'
if os.path.exists(FIG_SAVE_DIR) is False:
    os.makedirs(FIG_SAVE_DIR)
print("保存图片的文件夹名称：", FIG_SAVE_DIR.split('/')[-1])

METRIC='loss'

ENC_FILE_PATH = f'test/tracincp/rescaled_pred_and_gt_loss_trajectories.out'
#########################################################

enc_sim_list = []

with open(ENC_FILE_PATH, 'r', encoding='utf-8') as f:
    for line in f.readlines():
        line = json.loads(line)
        enc_sim_list.append(line)
for test_sample_id, encs in tqdm(enumerate(enc_sim_list)):
    for run_id, enc in enumerate(encs):
        # print('debug')
        # plt.plot(origin[])
        # 绘制gt曲线
        plt.plot(enc['step'], enc['gt_loss'], label='Ground Truth', linewidth=1.5, color='#0066fe')
        # # 绘制origin曲线
        # plt.plot(origin['step'], origin['pred_loss'], label='Simfluence', linewidth=1.0, color='#dbdbdb', marker='X', markevery=6)
        # # 绘制vec_sim曲线
        # plt.plot(vec['step'], vec['pred_loss'], label='Simfluence+MLP', linewidth=1.0, color='#909090', marker='P', markevery=6)
        # 绘制enc_sim曲线
        plt.plot(enc['step'], enc['rescaled_pred_loss'], label='Ours', linewidth=1, color='#59d65a')
        plt.legend()
        plt.grid()
        plt.title(f'test_sample_id: {test_sample_id}, run_id: {run_id}')
        plt.xlabel('Training Steps')
        plt.ylabel(f'{METRIC}')
        plt.savefig(f'{FIG_SAVE_DIR}/test_sample_{test_sample_id}_run_{run_id}.png')
        plt.clf()
