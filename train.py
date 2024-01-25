# coding: utf-8

import torch
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
from tensorboardX import SummaryWriter

# 导入dataset
from dataset.simfluence_dataset import SimfluenceDataset
from dataset.rte_dataset import RteDataset
from dataset.base_dataset import BaseDataset

# 导入Simulator
from model.Simulator import Simulator
from model.XlmrSimulator import XlmrSimulator
from model.VectorSimulator import VectorSimulator
from model.EncSimulator import EncSimulator

from utils.eval import eval_simulator

from torch.utils.data import DataLoader
import random
from tqdm import tqdm
from transformers import (
    get_linear_schedule_with_warmup,
    get_constant_schedule,
)
import fire
import logging
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("Simfuence.train")

DATASET = {
    "": None,
    'rte': BaseDataset,
    'boolq': BaseDataset,
    'sst2': BaseDataset,
    'webnlg': BaseDataset,
    'wmt16_de_en': BaseDataset,
}

DATASET_ADDITIONAL_ARGS = {
    "": {},
    'rte': {
        'train_data_path': '/root/paddlejob/workspace/env_run/liuqingyi01/data/eval_data/tda_tasks/rte/rte_0_399_train.json',
        'eval_data_path': '/root/paddlejob/workspace/env_run/liuqingyi01/data/eval_data/tda_tasks/rte/rte_0_276_eval.json',
    },
    'boolq': {
        'train_data_path': '/root/paddlejob/workspace/env_run/liuqingyi01/data/eval_data/tda_tasks/boolq/boolq_train_0-199.json',
        'eval_data_path': '/root/paddlejob/workspace/env_run/liuqingyi01/data/eval_data/tda_tasks/boolq/boolq_validation_0-199.json',
    },
    'sst2': {
        'train_data_path': '/root/paddlejob/workspace/env_run/liuqingyi01/data/eval_data/tda_tasks/sst2/sst2_train_0-199.json',
        'eval_data_path': '/root/paddlejob/workspace/env_run/liuqingyi01/data/eval_data/tda_tasks/sst2/sst2_validation_0-199.json'
    },
    'webnlg': {
        'train_data_path': '/root/paddlejob/workspace/env_run/liuqingyi01/data/eval_data/tda_tasks/webnlg/webnlg_train_0-399.json',
        'eval_data_path': '/root/paddlejob/workspace/env_run/liuqingyi01/data/eval_data/tda_tasks/webnlg/webnlg_dev_0-177.json'

    },
    'wmt16_de_en': {
        'train_data_path': '/root/paddlejob/workspace/env_run/liuqingyi01/data/eval_data/tda_tasks/wmt16_de_en/wmt16_de_en_train_0-199.json',
        'eval_data_path': '/root/paddlejob/workspace/env_run/liuqingyi01/data/eval_data/tda_tasks/wmt16_de_en/wmt16_de_en_test_0-199.json'

    }
}

SIMULATORS = {
    'original': Simulator,
    'vec_sim': VectorSimulator,
    'xlmr_sim': XlmrSimulator,
    'enc_sim': EncSimulator,
}

SIMULATR_ADDIONAL_ARGS = {
    'original': {},
    'vec_sim': {
        'emb_dim': 2
    },
    'xlmr_sim': {
        'xlmr_model_name_or_path': '/root/paddlejob/workspace/env_run/liuqingyi01/data/model/xlm_roberta_base',
        'train_xlm_ids_file': '/root/paddlejob/workspace/env_run/liuqingyi01/data/eval_data/wmt18/tr-en/en-tr_tgt-en_xlmr-ids.json',
        'test_xlm_ids_file': '/root/paddlejob/workspace/env_run/liuqingyi01/data/eval_data/wmt18/tr-en/dev/newsdev2016-tren-tren_xlmr-ids.json',
    },
    'enc_sim': {
        'enc_model_name_or_path': '/root/paddlejob/workspace/env_run/liuqingyi01/data/model/models--sentence-transformers--all-MiniLM-L6-v2/',
        'frozen': True,
        'use_initial': True,
    }
}

INPUT_ADDITIONAL_KEYS ={
    'original': {},
    'vec_sim': {},
    'xlmr_sim': {},
    'enc_sim': {
        'samples_texts',
        'test_sample_text',
    }
}

# 维护自动生成save dir时需要忽略的参数
SAVE_DIR_IGNORED_ARG_NAME = {
    'original': [],
    'vec_sim': [],
    'xlmr_sim': [
        'xlmr_model_name_or_path',
        'train_xlm_ids_file',
        'test_xlm_ids_file',
    ],
    'enc_sim': [
        'enc_model_name_or_path',
    ]
}

def train(
    sim_name = "original",
    dataset_name = "",
    task="",
    max_epoch = 2000,
    # train_run = 22,
    # val_run = 10,
    train_bs = 128,
    # valid_bs = 16,
    train_example_nums = 205756,
    test_example_nums = 10,
    # num_samples_to_valid = 100,
    # num_samples_to_select = 64,
    seed=42,
    hyper_parameter=0.,
    lr=0.00001,
    valid_epoch_interval=100,
    save_epoch_interval=100,
    output_dir='.',
    valid_num = 2,
    test_num = 2,
    weight_decay=0.,
    step_thres=None,
):

    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
    # 设置随机数种子
    setup_seed(seed)
    # 加载模拟器训练数据
    data_paths = [

        # boolq 
        # './runs/boolq/output_boolq_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-1/',
        # './runs/boolq/output_boolq_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-2/',
        # './runs/boolq/output_boolq_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-3/',
        # './runs/boolq/output_boolq_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-4/',
        # './runs/boolq/output_boolq_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-5/',
        # './runs/boolq/output_boolq_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-6/',
        # './runs/boolq/output_boolq_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-7/',
        # './runs/boolq/output_boolq_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-8/',
        # './runs/boolq/output_boolq_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-9/',
        # './runs/boolq/output_boolq_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-10/',
        # './runs/boolq/output_boolq_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-11/',
        # './runs/boolq/output_boolq_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-12/',
        # './runs/boolq/output_boolq_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-13/',
        # './runs/boolq/output_boolq_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-14/',
        # './runs/boolq/output_boolq_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-15/',
        # './runs/boolq/output_boolq_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-16/',
        # './runs/boolq/output_boolq_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-17/',
        # './runs/boolq/output_boolq_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-18/',
        # './runs/boolq/output_boolq_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-19/',
        # './runs/boolq/output_boolq_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-20/',
        # './runs/boolq/output_boolq_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-21/',
        # './runs/boolq/output_boolq_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-22/',
        # './runs/boolq/output_boolq_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-23/',
        # './runs/boolq/output_boolq_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-24/',
        # './runs/boolq/output_boolq_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-25/',
        # './runs/boolq/output_boolq_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-26/',
        # './runs/boolq/output_boolq_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-27/',

        # sst2
        './runs/sst2/output_sst2_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-last-token_seed-1',
        './runs/sst2/output_sst2_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-last-token_seed-2',
        './runs/sst2/output_sst2_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-last-token_seed-3',
        './runs/sst2/output_sst2_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-last-token_seed-4',
        './runs/sst2/output_sst2_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-last-token_seed-5',
        './runs/sst2/output_sst2_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-last-token_seed-6',
        './runs/sst2/output_sst2_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-last-token_seed-7',
        './runs/sst2/output_sst2_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-last-token_seed-8',
        './runs/sst2/output_sst2_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-last-token_seed-9',
        './runs/sst2/output_sst2_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-last-token_seed-10',
        './runs/sst2/output_sst2_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-last-token_seed-11',
        './runs/sst2/output_sst2_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-last-token_seed-12',
        './runs/sst2/output_sst2_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-last-token_seed-13',
        './runs/sst2/output_sst2_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-last-token_seed-14',
        './runs/sst2/output_sst2_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-last-token_seed-15',
        './runs/sst2/output_sst2_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-last-token_seed-16',
        './runs/sst2/output_sst2_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-last-token_seed-17',
        './runs/sst2/output_sst2_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-last-token_seed-18',
        './runs/sst2/output_sst2_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-last-token_seed-19',
        './runs/sst2/output_sst2_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-last-token_seed-20',
        './runs/sst2/output_sst2_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-last-token_seed-21',
        './runs/sst2/output_sst2_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-last-token_seed-22',
        './runs/sst2/output_sst2_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-last-token_seed-23',
        './runs/sst2/output_sst2_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-last-token_seed-24',
        './runs/sst2/output_sst2_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-last-token_seed-25',
        './runs/sst2/output_sst2_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-last-token_seed-26',
        './runs/sst2/output_sst2_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-last-token_seed-27',

        # webnlg
        # './runs/webnlg/output_webnlg_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-1e-6_weight-decay-0.001_epoch-3_loss-output-token_seed-1',
        # './runs/webnlg/output_webnlg_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-1e-6_weight-decay-0.001_epoch-3_loss-output-token_seed-2',
        # './runs/webnlg/output_webnlg_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-1e-6_weight-decay-0.001_epoch-3_loss-output-token_seed-3',
        # './runs/webnlg/output_webnlg_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-1e-6_weight-decay-0.001_epoch-3_loss-output-token_seed-4',
        # './runs/webnlg/output_webnlg_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-1e-6_weight-decay-0.001_epoch-3_loss-output-token_seed-5',
        # './runs/webnlg/output_webnlg_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-1e-6_weight-decay-0.001_epoch-3_loss-output-token_seed-6',
        # './runs/webnlg/output_webnlg_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-1e-6_weight-decay-0.001_epoch-3_loss-output-token_seed-7',
        # './runs/webnlg/output_webnlg_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-1e-6_weight-decay-0.001_epoch-3_loss-output-token_seed-8',
        # './runs/webnlg/output_webnlg_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-1e-6_weight-decay-0.001_epoch-3_loss-output-token_seed-9',
        # './runs/webnlg/output_webnlg_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-1e-6_weight-decay-0.001_epoch-3_loss-output-token_seed-10',
        # './runs/webnlg/output_webnlg_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-1e-6_weight-decay-0.001_epoch-3_loss-output-token_seed-11',
        # './runs/webnlg/output_webnlg_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-1e-6_weight-decay-0.001_epoch-3_loss-output-token_seed-12',
        # './runs/webnlg/output_webnlg_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-1e-6_weight-decay-0.001_epoch-3_loss-output-token_seed-13',
        # './runs/webnlg/output_webnlg_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-1e-6_weight-decay-0.001_epoch-3_loss-output-token_seed-14',
        # './runs/webnlg/output_webnlg_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-1e-6_weight-decay-0.001_epoch-3_loss-output-token_seed-15',
        # './runs/webnlg/output_webnlg_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-1e-6_weight-decay-0.001_epoch-3_loss-output-token_seed-16',
        # './runs/webnlg/output_webnlg_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-1e-6_weight-decay-0.001_epoch-3_loss-output-token_seed-17',
        # './runs/webnlg/output_webnlg_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-1e-6_weight-decay-0.001_epoch-3_loss-output-token_seed-18',
        # './runs/webnlg/output_webnlg_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-1e-6_weight-decay-0.001_epoch-3_loss-output-token_seed-19',
        # './runs/webnlg/output_webnlg_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-1e-6_weight-decay-0.001_epoch-3_loss-output-token_seed-20',
        # './runs/webnlg/output_webnlg_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-1e-6_weight-decay-0.001_epoch-3_loss-output-token_seed-21',
        # './runs/webnlg/output_webnlg_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-1e-6_weight-decay-0.001_epoch-3_loss-output-token_seed-22',
        # './runs/webnlg/output_webnlg_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-1e-6_weight-decay-0.001_epoch-3_loss-output-token_seed-23',
        # './runs/webnlg/output_webnlg_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-1e-6_weight-decay-0.001_epoch-3_loss-output-token_seed-24',
        # './runs/webnlg/output_webnlg_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-1e-6_weight-decay-0.001_epoch-3_loss-output-token_seed-25',
        # './runs/webnlg/output_webnlg_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-1e-6_weight-decay-0.001_epoch-3_loss-output-token_seed-26',
        # './runs/webnlg/output_webnlg_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-1e-6_weight-decay-0.001_epoch-3_loss-output-token_seed-27',

        # rte
        # './runs/rte/output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-1/',
        # './runs/rte/output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-2/',
        # './runs/rte/output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-3/',
        # './runs/rte/output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-4/',
        # './runs/rte/output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-5/',
        # './runs/rte/output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-6/',
        # './runs/rte/output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-7/',
        # './runs/rte/output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-8/',
        # './runs/rte/output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-9/',
        # './runs/rte/output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-10/',
        # './runs/rte/output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-11/',
        # './runs/rte/output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-12/',
        # './runs/rte/output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-13/',
        # './runs/rte/output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-14/',
        # './runs/rte/output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-15/',
        # './runs/rte/output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-16/',
        # './runs/rte/output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-17/',
        # './runs/rte/output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-18/',
        # './runs/rte/output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-19/',
        # './runs/rte/output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-20/',
        # './runs/rte/output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-21/',
        # './runs/rte/output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-22/',
        # './runs/rte/output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-23/',
        # './runs/rte/output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-24/',
        # './runs/rte/output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-25/',
        # './runs/rte/output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-26/',
        # './runs/rte/output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-27/',

        # wmt16_de_en
        # 'runs/wmt16_de_en/output_wmt16_de_en_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-1',
        # 'runs/wmt16_de_en/output_wmt16_de_en_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-10',
        # 'runs/wmt16_de_en/output_wmt16_de_en_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-11',
        # 'runs/wmt16_de_en/output_wmt16_de_en_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-17',
        # 'runs/wmt16_de_en/output_wmt16_de_en_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-18',
        # 'runs/wmt16_de_en/output_wmt16_de_en_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-19',
        # 'runs/wmt16_de_en/output_wmt16_de_en_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-2',
        # 'runs/wmt16_de_en/output_wmt16_de_en_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-20',
        # 'runs/wmt16_de_en/output_wmt16_de_en_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-21',
        # 'runs/wmt16_de_en/output_wmt16_de_en_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-22',
        # 'runs/wmt16_de_en/output_wmt16_de_en_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-23',
        # 'runs/wmt16_de_en/output_wmt16_de_en_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-24',
        # 'runs/wmt16_de_en/output_wmt16_de_en_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-25',
        # 'runs/wmt16_de_en/output_wmt16_de_en_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-26',
        # 'runs/wmt16_de_en/output_wmt16_de_en_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-27',
        # 'runs/wmt16_de_en/output_wmt16_de_en_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-28',
        # 'runs/wmt16_de_en/output_wmt16_de_en_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-29',
        # 'runs/wmt16_de_en/output_wmt16_de_en_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-3',
        # 'runs/wmt16_de_en/output_wmt16_de_en_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-33',
        # 'runs/wmt16_de_en/output_wmt16_de_en_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-34',
        # 'runs/wmt16_de_en/output_wmt16_de_en_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-35',
        # 'runs/wmt16_de_en/output_wmt16_de_en_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-36',
        # 'runs/wmt16_de_en/output_wmt16_de_en_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-4',
        # 'runs/wmt16_de_en/output_wmt16_de_en_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-43',
        # 'runs/wmt16_de_en/output_wmt16_de_en_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-44',
        # 'runs/wmt16_de_en/output_wmt16_de_en_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-45',
        # 'runs/wmt16_de_en/output_wmt16_de_en_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-46',
    ]
    print(
        f"data_paths: {data_paths}\n",
        f"dataset_name: {dataset_name}\n",
        f"task: {task}\n",
        f"sim_name: {sim_name}\n",
        f"max_epoch: {max_epoch}\n",
        f"train_bs: {train_bs}\n",
        f"lr: {lr}\n",
        f"train_example_nums: {train_example_nums}\n",
        f"test_example_nums: {test_example_nums}\n",
        # f"valid_bs: {valid_bs}\n",
        f"seed: {seed}\n",
        f"hyper_parameter: {hyper_parameter}\n"
        f"valid_epoch_interval: {valid_epoch_interval}\n",
        f"save_epoch_interval: {save_epoch_interval}\n",
        f"valid_num: {valid_num}\n",
        f"test_num: {test_num}\n",
        f"weight_decay: {weight_decay}\n",
        f"output_dir: {output_dir}\n",
        f"step_thres: {step_thres}\n"
    )

    log_dir = "tb-logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 设置tensorboard日志保存路径
    save_dir_name = f'/{sim_name}_task-{task}_lr-{lr}_lambda-{hyper_parameter}_bs-{train_bs}_train-sample-nums-{train_example_nums}_test-sample-nums-{test_example_nums}_seed-{seed}_step_thres-{step_thres}'
    simulator_args = SIMULATR_ADDIONAL_ARGS[sim_name]
    ignore_args = SAVE_DIR_IGNORED_ARG_NAME[sim_name]
    for args_name, args_value in simulator_args.items():
        if args_name in ignore_args:
            continue
        save_dir_name += f'_{args_name}-{args_value}'
    writer = SummaryWriter(log_dir=log_dir + save_dir_name)


    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   


    random.shuffle(data_paths)
    valid_num = valid_num
    test_num = test_num
    
    valid_dataset = SimfluenceDataset(data_paths[:valid_num], is_train=False, test_example_nums=test_example_nums, step_thres=step_thres)
    # test_dataset = SimfluenceDataset(data_paths[valid_num: valid_num + test_num], is_train=False, test_example_nums=test_example_nums, step_thres=step_thres)
    train_dataset = SimfluenceDataset(data_paths[valid_num + test_num:], test_example_nums=test_example_nums, step_thres=step_thres)
    
    print('')
    print(f'valid dataset: {data_paths[:valid_num]}')
    # print(f'test dataset: {data_paths[valid_num: valid_num + test_num]}')
    print(f'train dataset: {data_paths[valid_num + test_num:]}')
    print('')

    # 加载数据集
    dataset = DATASET[dataset_name]
    if dataset is None:
        logger.warning("dataset is None, use default dataset")
    else:
        valid_dataset = dataset(
            valid_dataset,
            is_train=False,
            **DATASET_ADDITIONAL_ARGS[dataset_name]
        )
        train_dataset = dataset(
            train_dataset,
            is_train=True,
            **DATASET_ADDITIONAL_ARGS[dataset_name]
        )

    # debug
    # -----------------------------------
    # n = 3
    # logger.warning(f"only {n} training samples, this is for debugging")
    # train_dataset = train_dataset[10:10+n]
    # -----------------------------------
    train_data_loader = DataLoader(train_dataset, batch_size=train_bs, shuffle=True, collate_fn=lambda x: train_dataset.collate_fn(x, device=device))

    # 加载simulator
    model = SIMULATORS[sim_name](train_example_nums=train_example_nums, hyper_parameter=hyper_parameter, test_example_nums=test_example_nums, **SIMULATR_ADDIONAL_ARGS[sim_name])
    model.to(device).train()

    if sim_name == 'enc_sim':
        if SIMULATR_ADDIONAL_ARGS[sim_name]['use_initial']:
            model._get_initial_embeds(train_dataset, device)

    # criterion = nn.MSELoss(reduction="mean")                                                
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)  
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=200, num_training_steps=max_epoch*len(train_data_loader))    
    # scheduler = get_constant_schedule(optimizer)

    train_losses = []
    tot_step = 0
    for epoch in range(max_epoch):
        pbar = tqdm(train_data_loader)
        pbar.set_description("Epoch: [{}/{}]".format(epoch + 1, max_epoch))
        for step, data in enumerate(pbar):
            
            input_kwargs_keys = INPUT_ADDITIONAL_KEYS[sim_name]
            input_kwargs = {key: data[key] for key in input_kwargs_keys}
            outputs = model(
                orders=data["samples_id"],
                before_loss=data["prev_loss"],
                after_loss=data["cur_loss"],
                test_sample_ids=data["test_sample_id"],
                device=device,
                **input_kwargs,
            )
            
            loss = outputs['tot_loss']        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_losses.append(loss.item())   
            
            # print("Training: step[{:0>3}/{:0>3}] Loss: {:.4f} ".format(step + 1, len(train_data_loader), loss))
            pbar.set_postfix(loss=loss.item(), lr=scheduler.get_last_lr()[0])
            tot_step += 1
            writer.add_scalar('loss', loss.item(), tot_step)
            writer.add_scalar('learning_rate', scheduler.get_last_lr()[0], tot_step)

        # 验证集
        if epoch % valid_epoch_interval == 0:
            all_steps_mse, _ = eval_simulator(
                eval_dataset=valid_dataset,
                model=model,
                device=device,
                input_kwargs_keys=input_kwargs_keys,
            )
            writer.add_scalar('valid_all_steps_mse', all_steps_mse, epoch)
            model.train()
            print(f'valid mse loss {all_steps_mse}')
        
        # 保存模型
        if epoch % save_epoch_interval == 0:
            net_save_dir = f'{output_dir}/{save_dir_name}'
            if not os.path.exists(net_save_dir):
                os.makedirs(net_save_dir)
            net_save_path = os.path.join(net_save_dir, f'checkpoint-{epoch}.pt')
            
            torch.save(model.state_dict(), net_save_path)

if __name__ == "__main__":
    fire.Fire(train)
