import json
import os
from collections import defaultdict
from tqdm import tqdm
import time
from concurrent.futures import ProcessPoolExecutor


start = time.time()

################################################################################
# 文件夹结构
# 多个评估任务
# output_flan_bs-8_shot-200_sample-128_model-pythia-410m-deduped_lr-2e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-1 (output file name)
# ├─-loss_sst2 (评估输出文件)
# │ ├─-loss_rte_checkpoint-1.out
# │ ├─-loss_rte_checkpoint-2.out
# |--loss_rte
# |--loss_boolq
# |--metric_wmt16_de_en
# |--metric_webnlg
# |__log.out (训练log)
################################################################################
# `runs-data` 保存路径
# TEMPLATE="rte_bs-4_shot-200_sample-128_lr-2e-6_weight-decay-0.001_epoch-4_seed"

# 任务名称
TASK='flan_ood'

# 训练样本id offeset
TRAIN_SAMPLE_OFFSET = 1200

# 子任务信息
SUB_TASKS = {
    # 'rte': {
    #     'file_name': 'loss_rte',
    #     'test_sample_id_offset': 0 + 200,
    # },
    'webnlg': {
        'file_name': 'metric_webnlg',
        'test_sample_id_offset': 0 + 200 + 277 + 200,
    },
}
print(f'flan中评估的子任务：{SUB_TASKS.keys()}')
# run数据本地保存根目录
SAVE_ROOT_DIR = f"runs/{TASK}"

# 输出根目录
OURPUT_ROOT_DIR = "/root/paddlejob/workspace/env_run/liuqingyi01/tda_output/webnlg/"

# 输出文件
OUTPUT_FILE_NAME_LIST = [
    #rte ood
    # 'output_rte-ood_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-1/',
    # 'output_rte-ood_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-2/',
    # 'output_rte-ood_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-3/',
    
    # wenlg ood
    'output_webnlg-ood_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-1e-6_weight-decay-0.001_epoch-3_loss-output-token_seed-1',
    'output_webnlg-ood_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-1e-6_weight-decay-0.001_epoch-3_loss-output-token_seed-2',
    'output_webnlg-ood_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-1e-6_weight-decay-0.001_epoch-3_loss-output-token_seed-3',
]
# 训练的step数  
STEP_NUM = 96

def process_file(args):
    step, file_path, test_sample_id_offset = args
    result = defaultdict(list)
    if not os.path.exists(file_path):
        print(f"评估文件 {file_path} 不存在，跳过")
        return result
    with open(file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            line = json.loads(line)
            id = line['id'] + test_sample_id_offset
            loss = line['loss']
            result[id].append({'step': step, 'loss': loss})
    return result


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
                samples_id = [ sample_id + TRAIN_SAMPLE_OFFSET for sample_id in o['samples_id'] ]
                
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
        tasks = []
        for step in tqdm(range(1, STEP_NUM+1)):
            
            # 读取flan所有子任务的测试集结果
            for task_name, task_info in SUB_TASKS.items():

                file_name = task_info['file_name']
                file_prefix = file_name
                loss_trajectory_file_path = os.path.join(OURPUT_ROOT_DIR, output_file_name, f"{file_name}/{file_prefix}_checkpoint-{step}.out")

                test_sample_id_offset = task_info['test_sample_id_offset']
                tasks.append((step, loss_trajectory_file_path, test_sample_id_offset))

                # loss_trajectory_file_path = os.path.join(OURPUT_ROOT_DIR, output_file_name, f"loss_{TASK}/loss_{TASK}_checkpoint-{step}.out")
                # if not os.path.exists(loss_trajectory_file_path):
                #     print(f"step: {step} 评估文件不存在，跳过")
                #     continue
                # with open(loss_trajectory_file_path, "r") as f:
                #     lines = f.readlines()
                #     for line in lines:
                #         line = line.strip()
                #         line = json.loads(line)
                #         id = line['id'] + test_sample_id_offset
                #         loss = line['loss']
                #         eval_data_list[id].append({'step': step, 'loss': loss})

        with ProcessPoolExecutor() as executor:
            for result in executor.map(process_file, tasks):
                for k, v in result.items():
                    eval_data_list[k].extend(v)

        sorted_eval_data_list = {}
        # 对eval_data_list中测试样本的trajectory按照step排序
        for test_sample_id, v in eval_data_list.items():
            trajectory = sorted(eval_data_list[test_sample_id], key=lambda x:x['step'])
            sorted_eval_data_list[test_sample_id] = trajectory

        with open(eval_processed_result_path, 'w', encoding='utf-8') as f:
            for k, v in eval_data_list.items():
                print(json.dumps({'id': k, 'loss_trajectory': v}), file=f)
        print(f"测试样本数量: {len(eval_data_list)}")
        print("\n")

end = time.time()
print(f"总共处理时间: {end - start}s")