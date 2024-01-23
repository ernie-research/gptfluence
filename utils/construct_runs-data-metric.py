import json
import os
from collections import defaultdict
import tqdm
import time

start = time.time()

################################################################################
# 文件夹结构
# 以 wmt16_de_en 任务为例
# output_wmt16_de_en_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-1e-6_weight-decay-0.001_epoch-3_loss-output-token_seed-1 (output file name)
# ├─-metric_rte (评估输出文件)
# │ ├─-metric_rte_checkpoint-1.out
# │ ├─-metric_rte_checkpoint-2.out
# |__log.out (训练log)
################################################################################
# `runs-data` 保存路径
# TEMPLATE="rte_bs-4_shot-200_sample-128_lr-2e-6_weight-decay-0.001_epoch-4_seed"

# 任务名称
TASK='wmt16_de_en' # wmt16_de_en | webnlg

# run数据本地保存根目录
SAVE_ROOT_DIR = f"runs/{TASK}_test"

# 输出根目录
OURPUT_ROOT_DIR = "/root/paddlejob/workspace/liuqingyi01/code/alpaca-lora-main/test"

# 输出文件
OUTPUT_FILE_NAME_LIST = [
    # pythia-1.4b-deduped
    
    # webnlg
    # "output_webnlg_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-2e-6_weight-decay-0.001_epoch-3_loss-output-token_seed-1",
    # "output_webnlg_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-1e-6_weight-decay-0.001_epoch-3_loss-output-token_seed-1"
    
    # wmt16_de_en
    # 'output_wmt16_de_en_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-1e-6_weight-decay-0.001_epoch-3_loss-output-token_seed-1'
    'output_wmt16_de_en_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-1'

]
# 训练的step数  
STEP_NUM = 128

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
    log_path = os.path.join(OURPUT_ROOT_DIR, output_file_name, 'log.out')
    train_data_list = []
    with open(log_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().replace('\'', '\"')
            if "\"samples_id\"" in line and "\"loss\"" in line and "\"step\"" in line:
                if ']{' in line: # 进度条和日志混在一行，需要分开
                    line = '{' + line.split(']{')[-1].strip()
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
            loss_trajectory_file_path = os.path.join(OURPUT_ROOT_DIR, output_file_name, f"metric_{TASK}/metric_{TASK}_checkpoint-{step}.out")
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