import matplotlib.pyplot as plt
import json
import os
from tqdm import tqdm

if __name__ == '__main__':
    # 绘制评估样本的曲线
    # trajectory文件路径
    ROOR_DIR="/root/paddlejob/workspace/liuqingyi01/code/Simfluence/runs/wmt16_de_en/output_wmt16_de_en_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-1/"
    METRIC = 'rougeL'
    print(f'即将即将绘制测试样本的{METRIC}的变化曲线')
    PATH = f"{ROOR_DIR}/all_{METRIC}_trajectory.out"
    # 图片保存目录
    SAVE_DIR = os.path.join(ROOR_DIR, f'figs_{METRIC}_trajectory')
    # 绘制前N个样本的曲线
    N = 20
    print(f"绘制前{N}个样本的曲线")
    
    with open(PATH, 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines[:N]):
            line = json.loads(line)
            sample_id = line['id']
            steps = []
            losses = []
            for p in line['loss_trajectory']:
                step = p['step']
                loss = p['loss']
                # if step < 49:
                #     print("截断前49步")
                #     continue
                steps.append(step)
                losses.append(loss)
            plt.plot(steps, losses)
            plt.xlabel('step')
            plt.ylabel(f'{METRIC}')
            plt.title(sample_id)
            plt.show()
            if not os.path.exists(SAVE_DIR):
                os.mkdir(SAVE_DIR)
            plt.savefig(os.path.join(SAVE_DIR, str(sample_id) + '.png'))
            plt.clf()
            # print(line)
    print("done")
