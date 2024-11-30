import json
import os
from collections import defaultdict
import tqdm
import time

start = time.time()

################################################################################
# 文件夹结构
# 以 rte 任务为例
# output_rte_bs-4_shot-200_sample-128_lr-2e-6_weight-decay-0.001_epoch-3_seed-1 (output file name)
# ├─-loss_rte (评估输出文件)
# │ ├─-loss_rte_checkpoint-1.out
# │ ├─-loss_rte_checkpoint-2.out
# |__log.out (训练log)
################################################################################
# `runs-data` 保存路径
# TEMPLATE="rte_bs-4_shot-200_sample-128_lr-2e-6_weight-decay-0.001_epoch-4_seed"

# 任务名称
TASK='rte'

# run数据本地保存根目录
SAVE_ROOT_DIR = f"/home/zhuliyi/code/tmp_gptfluence/gptfluence/data/runs/{TASK}"

# 输出根目录
OURPUT_ROOT_DIR = "/home/zhuliyi/code/tmp_gptfluence/gptfluence/output/tmp"



# 输出文件
OUTPUT_FILE_NAME_LIST = [
    # sst2
    # 'output_sst2_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-last-token_seed-1/',
    # 'output_sst2_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-last-token_seed-2/',
    # 'output_sst2_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-last-token_seed-3/',
    # 'output_sst2_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-last-token_seed-4/',
    # 'output_sst2_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-last-token_seed-5/',
    # 'output_sst2_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-last-token_seed-6/',
    # 'output_sst2_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-last-token_seed-7/',
    # 'output_sst2_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-last-token_seed-8/',
    # 'output_sst2_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-last-token_seed-9/',
    # 'output_sst2_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-last-token_seed-10/',
    # 'output_sst2_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-last-token_seed-11/',
    # 'output_sst2_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-last-token_seed-12/',
    # 'output_sst2_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-last-token_seed-13/',
    # 'output_sst2_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-last-token_seed-14/',
    # 'output_sst2_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-last-token_seed-15/',
    # 'output_sst2_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-last-token_seed-16/',
    # 'output_sst2_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-last-token_seed-17/',
    # 'output_sst2_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-last-token_seed-18/',
    # 'output_sst2_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-last-token_seed-19/',
    # 'output_sst2_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-last-token_seed-20/',
    # 'output_sst2_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-last-token_seed-21/',
    # 'output_sst2_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-last-token_seed-22/',
    # 'output_sst2_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-last-token_seed-23/',
    # 'output_sst2_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-last-token_seed-24/',
    # 'output_sst2_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-last-token_seed-25/',
    # 'output_sst2_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-last-token_seed-26/',
    # 'output_sst2_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-last-token_seed-27/',
    # 'output_sst2_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-last-token_seed-28/',
    # 'output_sst2_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-last-token_seed-29/',
    # 'output_sst2_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-last-token_seed-30/',
    # 'output_sst2_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-last-token_seed-31/',
    # 'output_sst2_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-last-token_seed-32/',

    # rte
    # 'output_rte-ood_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-1',
    # 'output_rte-ood_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-2',
    # 'output_rte-ood_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-3'
    'output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-1',
    'output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-2',
    'output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-3',
    'output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-4',
    'output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-5',
    'output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-6',
    'output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-7',
    'output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-8',
    'output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-9',
    'output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-10',
    'output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-11',
    'output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-12',
    'output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-13',
    'output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-14',
    'output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-15',
    'output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-16',
    'output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-17',
    'output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-18',
    'output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-19',
    'output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-20',
    'output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-21',
    'output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-22',
    'output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-23',
    'output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-24',
    'output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-25',
    'output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-26',
    'output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-27',
    'output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-28',
    'output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-29',
    'output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-30',
    'output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-31',
    'output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-32',
]
# 训练的step数  
STEP_NUM = 96

# 对每个输出文件下的训练log和评估结果进行处理
for output_file_name in OUTPUT_FILE_NAME_LIST:
    print(f"正在处理：{output_file_name}",)
    
    save_path = os.path.join(SAVE_ROOT_DIR, output_file_name)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 处理训练log
    # 提取step和sample id, 保存为train_samples_id.json
    # train_samples_id.json 格式：
    #    {'step': step1, 'samples_id': samples_id1}
    #    {'step': step2, 'samples_id': samples_id2}
    if not os.path.exists(OURPUT_ROOT_DIR):
        # 如果文件夹不存在，则创建
        os.makedirs(OURPUT_ROOT_DIR)
        print(f"文件夹 '{OURPUT_ROOT_DIR}' 已创建。")
    else:
        # 如果文件夹存在，则跳过
        print(f"文件夹 '{OURPUT_ROOT_DIR}' 已存在，无需创建。")
    
    log_path = os.path.join(OURPUT_ROOT_DIR, output_file_name, 'log.out')
    train_data_list = []
    with open(log_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().replace('\'', '\"')
            if "\"samples_id\"" in line and "\"loss\"" in line and "\"step\"" in line:
                train_data_list.append(json.loads(line))

    train_processed_result_path = os.path.join(SAVE_ROOT_DIR, output_file_name, 'train_samples_id.json')
    if os.path.exists(train_processed_result_path):
        print(f'{train_processed_result_path} 已存在，跳过')
    else:
        with open(train_processed_result_path, 'w', encoding='utf-8') as f:
            for i in range(len(train_data_list)):
                o = train_data_list[i]
                step = o['step']
                samples_id = o['samples_id']
                print(json.dumps({'step': step, 'samples_id':samples_id}), file=f)
        print(f"total training steps: {len(train_data_list)}")


    # 处理评估out文件
    # 保存为 all_loss_trajectory.out
    # all_loss_trajectory.out 格式：
    # {'id': 1, 'loss_trajectory': [{'step': step1, 'loss': loss11}, {'step': step2, 'loss': loss12}]}
    # {'id': 2, 'loss_trajectory': [{'step': step1, 'loss': loss21}, {'step': step2, 'loss': loss22}]}
    eval_processed_result_path = os.path.join(SAVE_ROOT_DIR, output_file_name, 'all_loss_trajectory.out')

    if os.path.exists(eval_processed_result_path):
        print(f'{eval_processed_result_path} 已存在，跳过')
    else:
        eval_data_list = defaultdict(list)
        for step in range(1, STEP_NUM+1):
            loss_trajectory_file_path = os.path.join(OURPUT_ROOT_DIR, output_file_name, f"loss_{TASK}/loss_{TASK}_checkpoint-{step}.out")
            if not os.path.exists(loss_trajectory_file_path):
                print(f"step: {step} 评估文件不存在，跳过")
                continue
            with open(loss_trajectory_file_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    line = json.loads(line)
                    id = line['id']
                    loss = line['loss']
                    eval_data_list[id].append({'step': step, 'loss': loss})
        with open(eval_processed_result_path, 'w', encoding='utf-8') as f:
            for k, v in eval_data_list.items():
                print(json.dumps({'id': k, 'loss_trajectory': v}), file=f)
        print(f"测试样本数量: {len(eval_data_list)}")
        print("\n")

end = time.time()
print(f"总共处理时间: {end - start}s")