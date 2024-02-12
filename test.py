from dataset.simfluence_dataset import SimfluenceDataset
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import json
from tqdm import tqdm
import fire
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
    # boolq
    
]
data_paths_dict = {
    'boolq': [
        './runs/boolq/output_boolq_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-28/',
        './runs/boolq/output_boolq_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-29/',
        './runs/boolq/output_boolq_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-30/',
        './runs/boolq/output_boolq_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-31/',
        './runs/boolq/output_boolq_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-32/',
    ],
    'sst2': [
        './runs/sst2/output_sst2_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-last-token_seed-28',
        './runs/sst2/output_sst2_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-last-token_seed-29',
        './runs/sst2/output_sst2_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-last-token_seed-30',
        './runs/sst2/output_sst2_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-last-token_seed-31',
        './runs/sst2/output_sst2_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-last-token_seed-32',
    ],
    'rte': [
        './runs/rte/output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-28',
        './runs/rte/output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-29',
        './runs/rte/output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-30',
        './runs/rte/output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-31',
        './runs/rte/output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-32',
    ],
    'webnlg': [
        './runs/webnlg/output_webnlg_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-1e-6_weight-decay-0.001_epoch-3_loss-output-token_seed-28',
        './runs/webnlg/output_webnlg_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-1e-6_weight-decay-0.001_epoch-3_loss-output-token_seed-29',
        './runs/webnlg/output_webnlg_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-1e-6_weight-decay-0.001_epoch-3_loss-output-token_seed-30',
        './runs/webnlg/output_webnlg_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-1e-6_weight-decay-0.001_epoch-3_loss-output-token_seed-31',
        './runs/webnlg/output_webnlg_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-1e-6_weight-decay-0.001_epoch-3_loss-output-token_seed-32',
    ],
    'wmt16_de_en': [
        'runs/wmt16_de_en/output_wmt16_de_en_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-5',
        'runs/wmt16_de_en/output_wmt16_de_en_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-6',
        'runs/wmt16_de_en/output_wmt16_de_en_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-7',
        'runs/wmt16_de_en/output_wmt16_de_en_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-8',
        'runs/wmt16_de_en/output_wmt16_de_en_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-9',
    ],
    'webnlg_ood_train': [
        './runs/webnlg/output_webnlg-ood_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-1e-6_weight-decay-0.001_epoch-3_loss-output-token_seed-1',
        './runs/webnlg/output_webnlg-ood_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-1e-6_weight-decay-0.001_epoch-3_loss-output-token_seed-2',
        './runs/webnlg/output_webnlg-ood_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-1e-6_weight-decay-0.001_epoch-3_loss-output-token_seed-3',
    ],
    'rte_ood_train': [
        './runs/rte/output_rte-ood_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-1',
        './runs/rte/output_rte-ood_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-2',
        './runs/rte/output_rte-ood_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-3',
    ],
    'flan': [
        'runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-410m-deduped_lr-2e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-5/',
        'runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-410m-deduped_lr-2e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-6/',
        'runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-410m-deduped_lr-2e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-7/',
        'runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-410m-deduped_lr-2e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-8/',
        'runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-410m-deduped_lr-2e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-9/',
    ],
    'flan_pythia-1b': [
    "runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-1b-deduped_lr-2e-7_weight-decay-0.001_epoch-2_loss-output-token_seed-5/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-1b-deduped_lr-2e-7_weight-decay-0.001_epoch-2_loss-output-token_seed-6/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-1b-deduped_lr-2e-7_weight-decay-0.001_epoch-2_loss-output-token_seed-7/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-1b-deduped_lr-2e-7_weight-decay-0.001_epoch-2_loss-output-token_seed-8/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-1b-deduped_lr-2e-7_weight-decay-0.001_epoch-2_loss-output-token_seed-9/",
    ],
    'flan_rte_ood': [
        'runs/flan_ood/output_rte-ood_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-1/',
        'runs/flan_ood/output_rte-ood_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-2/',
        'runs/flan_ood/output_rte-ood_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-3/',
    ],
    'flan_webnlg_ood': [
        'runs/flan_ood/output_webnlg-ood_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-1e-6_weight-decay-0.001_epoch-3_loss-output-token_seed-1',
        'runs/flan_ood/output_webnlg-ood_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-1e-6_weight-decay-0.001_epoch-3_loss-output-token_seed-2',
        'runs/flan_ood/output_webnlg-ood_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-1e-6_weight-decay-0.001_epoch-3_loss-output-token_seed-3',
    ],
    'flan_pythia-160m': [
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-160m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-5",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-160m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-6",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-160m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-7",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-160m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-8",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-160m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-9",
    ],
    'flan_pythia-70m': [
        "runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-70m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-5",
        "runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-70m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-6",
        "runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-70m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-7",
        "runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-70m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-8",
        "runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-70m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-9",
    ],
    'flan_pythia-14m': [
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-14m_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-5/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-14m_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-6/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-14m_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-7/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-14m_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-8/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-14m_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-9/",
    ]
}
def main(
    test_example_nums = 200,
    train_example_nums = 200,
    task = '',
    metric = "",
    sim_name = "original", # "vec_sim" "original"
    dataset_name = "boolq",
    check_point_path = "/root/paddlejob/workspace/liuqingyi01/code/Simfluence/output/original_task-output_boolq_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_lr-0.001_lambda-0.0_bs-128_train-sample-nums-200_test-sample-nums-200_seed-42_step_thres-None/checkpoint-233.pt",
    save_dir = "/root/paddlejob/workspace/liuqingyi01/code/Simfluence/output/original_task-output_boolq_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_lr-0.001_lambda-0.0_bs-128_train-sample-nums-200_test-sample-nums-200_seed-42_step_thres-None",
    hyper_parameter = 0.,
    test_example_start_id=-1,
    test_example_end_id=-1,
    order_n=None,
    concate=None,
):
    print("task:", task)
    print("metric:", metric)
    print("dataset:", dataset_name)
    print('simulator', sim_name)
    print()

    from train import (
        DATASET,
        DATASET_ADDITIONAL_ARGS,
        SIMULATORS,
        SIMULATR_ADDIONAL_ARGS,
        INPUT_ADDITIONAL_KEYS,
    )

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   

    simulator_args = SIMULATR_ADDIONAL_ARGS[sim_name]
    # 命令行参数将重写`simulator_args`
    if order_n is not None and 'order_n' in simulator_args.keys():
        print(f"重写order_n: {order_n}")
        simulator_args['order_n'] = order_n
    if concate is not None and 'concate' in simulator_args.keys():
        print(f'重写concate: {concate}')
        if concate == True:
            simulator_args['concate'] = True
        elif concate == False:
            simulator_args['concate'] = False
        else:
            raise NotImplementedError
            
    if sim_name == 'norder_enc_sim':
        order_n  = simulator_args['order_n']
    else:
        order_n = -1
    test_dataset = SimfluenceDataset(data_paths_dict[task], test_example_nums=test_example_nums, test_example_start_id=test_example_start_id, test_example_end_id=test_example_end_id, is_train=False, step_thres=None, metric=metric, order_n=order_n)
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
    model = SIMULATORS[sim_name](train_example_nums=train_example_nums, hyper_parameter=hyper_parameter, test_example_nums=test_example_nums, **simulator_args).to(device)

    if sim_name == 'enc_sim' or sim_name == 'norder_enc_sim' or sim_name == 'enc_cp_sim' or sim_name == 'gpt_sim':
        if simulator_args['use_initial']:
            model._get_initial_embeds(test_dataset, device)
    model.load_state_dict(torch.load(check_point_path))
    model.to(device)

    print("\n开始测试...")
    input_kwargs_keys = INPUT_ADDITIONAL_KEYS[sim_name]
    results = eval_simulator(test_dataset, model, device, input_kwargs_keys)
    # print("测试集mse:", results[0][0])
    # print("测试集mae:", results[0][1])
    print(f'指标: {metric}')
    print(f"测试集mse 均值:{results['all_steps_mse_mean']} 标准差:{results['all_steps_mse_std']}")
    print(f"测试集mae 均值:{results['all_steps_mae_mean']} 标准差:{results['all_steps_mae_std']}")
        
    # 保存结果
    print("\n保存结果...")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, f'pred_and_gt_{metric}_trajectories.out'), 'w', encoding='utf-8') as f:
        for test_sample_id, r in results['pred_loss_dict'].items():
            r = json.dumps(r)
            print(r, file=f)

    # 计算last-step spearman correlation ################################
    from collections import defaultdict
    last_step = {
        'pred': defaultdict(list),
        'gt': defaultdict(list)
    }
    with open(os.path.join(save_dir, f'pred_and_gt_{metric}_trajectories.out'), 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = json.loads(line)
            for runs_id, trajectory in enumerate(line):
                last_step['pred'][runs_id].append(trajectory['pred_loss'][-1])
                last_step['gt'][runs_id].append(trajectory['gt_loss'][-1])
    
    total_runs_num = len(last_step['pred'])
    last_step_spearman_list = []
    for i in range(total_runs_num):
        last_step_spearman_list.append(
            stats.spearmanr(last_step['gt'][i], last_step['pred'][i] ).statistic
        )
    # import pdb; pdb.set_trace()
    last_step_spearman_np = np.array(last_step_spearman_list)
    last_step_spearman_mean = last_step_spearman_np.mean()
    last_step_spearman_std = last_step_spearman_np.std()
    print(f"测试集last step spearman 均值: {last_step_spearman_mean}, 标准差: {last_step_spearman_std}")




    # # 画图
    # print("\n画图...")
    # fig_save_path = os.path.join(save_dir, 'figs')
    # if not os.path.exists(fig_save_path):
    #     os.mkdir(fig_save_path)
    # for test_sample_id, r in tqdm(results['pred_loss_dict].items()):
    #     for i, item in enumerate(r):
    #         plt.plot(item['step'], item['gt_loss'], label='gt')
    #         plt.plot(item['step'], item['pred_loss'], label='predict')
    #         plt.xlabel('step')
    #         plt.ylabel('loss')
    #         plt.title('test sample id:' + str(test_sample_id) + '-' + str(i))
    #         plt.legend()
    #         plt.grid()
    #         plt.savefig(os.path.join(fig_save_path, str(test_sample_id) + '-' + str(i) + '.png'))
    #         plt.clf()
    print('done')

if __name__ == "__main__":
    fire.Fire(main)