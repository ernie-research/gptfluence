import matplotlib.pyplot as plt
import json
import os
from tqdm import tqdm

if __name__ == '__main__':
    # 绘制评估样本的曲线
    # trajectory文件路径
    ROOR_DIR="/root/paddlejob/workspace/liuqingyi01/code/Simfluence/runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-160m-deduped_lr-5e-7_weight-decay-0.001_epoch-2_loss-output-token_seed-29/"
    METRIC = 'loss'
    print(f'即将即将绘制测试样本的{METRIC}的变化曲线')
    PATH = f"{ROOR_DIR}/all_{METRIC}_trajectory.out"

    # 绘制前N个样本的曲线
    N = 100
    START = 855
    TASK='wmt16_de_en'

    # 图片保存目录
    SAVE_DIR = os.path.join(ROOR_DIR, f'figs_{TASK}_{METRIC}_trajectory')
    print(f"绘制前{N}个样本的曲线")
    
    # 记录前N个样本的测试集loss
    from collections import defaultdict
    tot_loss_dict = defaultdict(int)
    
    with open(PATH, 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines[START:N+START]):
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
                tot_loss_dict[step] += loss # 叠加不同测试样本在相同step的loss
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
    
    # 绘制前N个样本的loss曲线
    tot_loss_list = []
    step_list = []
    for step, loss_sum in tot_loss_dict.items():
        tot_loss_list.append(loss_sum / N)
        step_list.append(step)
    
    plt.plot(step_list, tot_loss_list)
    plt.xlabel('step')
    plt.ylabel(f'{METRIC}')
    plt.title('Test Loss curves')
    plt.show()
    if not os.path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)
    plt.savefig(os.path.join(SAVE_DIR, 'test_loss.png'))
    plt.clf()

        
    print("done")
