import matplotlib.pyplot as plt
import os
import json
from tqdm import tqdm
# 设置matplot的颜色
import matplotlib as mpl
from cycler import cycler
mpl.rcParams['axes.prop_cycle'] = cycler(color='bcrygmk')

# 设置绘图参数 #########################################
FIG_SAVE_DIR = 'output/figs_boolq'
if os.path.exists(FIG_SAVE_DIR) is False:
    os.makedirs(FIG_SAVE_DIR)
print("保存图片的文件夹名称：", FIG_SAVE_DIR.split('/')[-1])

METRIC='loss'

ORIGINAL_FILE_PATH = f'output/original_task-output_boolq_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_lr-0.001_lambda-0.0_bs-128_train-sample-nums-200_test-sample-nums-200_seed-42_step_thres-None/pred_and_gt_{METRIC}_trajectories.out'

VEC_SIM_FILE_PATH = f'output/vec_sim_task-output_boolq_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_lr-0.0001_lambda-0.0_bs-128_train-sample-nums-200_test-sample-nums-200_seed-42_step_thres-None_emb_dim-2/pred_and_gt_{METRIC}_trajectories.out'

ENC_SIM_FILE_PATH = f'output/enc_sim_task-boolq_lr-0.0001_lambda-0.0_bs-128_train-sample-nums-200_test-sample-nums-200_seed-42_step_thres-None_frozen-True_use_initial-True/pred_and_gt_{METRIC}_trajectories.out'
#########################################################

origin_list = []
vec_sim_list = []
enc_sim_list = []

with open(ORIGINAL_FILE_PATH, 'r', encoding='utf-8') as f:
    for line in f.readlines():
        line = json.loads(line)
        origin_list.append(line)
with open(VEC_SIM_FILE_PATH, 'r', encoding='utf-8') as f:
    for line in f.readlines():
        line = json.loads(line)
        vec_sim_list.append(line)
with open(ENC_SIM_FILE_PATH, 'r', encoding='utf-8') as f:
    for line in f.readlines():
        line = json.loads(line)
        enc_sim_list.append(line)

for test_sample_id, (origins, vecs, encs) in tqdm(enumerate(zip(origin_list, vec_sim_list, enc_sim_list))):
    for run_id, (origin, vec, enc) in enumerate(zip(origins, vecs, encs)):
        # print('debug')
        # plt.plot(origin[])
        # 绘制gt曲线
        plt.plot(origin['step'], origin['gt_loss'], label='Ground Truth', linewidth=1.5, color='#0066fe')
        # 绘制origin曲线
        plt.plot(origin['step'], origin['pred_loss'], label='Simfluence', linewidth=1.0, color='#dbdbdb', marker='X', markevery=6)
        # 绘制vec_sim曲线
        plt.plot(vec['step'], vec['pred_loss'], label='Simfluence+MLP', linewidth=1.0, color='#909090', marker='P', markevery=6)
        # 绘制enc_sim曲线
        plt.plot(enc['step'], enc['pred_loss'], label='Ours', linewidth=1, color='#59d65a')
        plt.legend()
        plt.grid()
        plt.title(f'test_sample_id: {test_sample_id}, run_id: {run_id}')
        plt.xlabel('Training Steps')
        plt.ylabel(f'{METRIC}')
        plt.savefig(f'{FIG_SAVE_DIR}/test_sample_{test_sample_id}_run_{run_id}.png')
        plt.clf()
