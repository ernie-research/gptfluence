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
from model.NOrder_EncSimulator import NOrder_EncSimulator
from model.TracInCPSimulator import TracInCPSimulator
from model.GPTSimulator import GPTSimulator

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
    'flan': BaseDataset,
    'dataset_debug_sst2': BaseDataset,
}

DATASET_ADDITIONAL_ARGS = {
    "": {},
    'rte': {
        'train_data_path': './gptdynamics/sft_tasks/rte/rte_0_399_train.json',
        'eval_data_path': './gptdynamics/sft_tasks/rte/rte_0_276_eval.json',
    },
    'boolq': {
        'train_data_path': './gptdynamics/sft_tasks/boolq/boolq_train_0-199.json',
        'eval_data_path': './gptdynamics/sft_tasks/boolq/boolq_validation_0-199.json',
    },
    'sst2': {
        'train_data_path': './gptdynamics/sft_tasks/sst2/sst2_train_0-199.json',
        'eval_data_path': './gptdynamics/sft_tasks/sst2/sst2_validation_0-199.json'
    },
    'webnlg': {
        'train_data_path': './gptdynamics/sft_tasks/webnlg/webnlg_train_0-399.json',
        'eval_data_path': './gptdynamics/sft_tasks/webnlg/webnlg_dev_0-177.json'
    },
    'wmt16_de_en': {
        'train_data_path': './gptdynamics/sft_tasks/wmt16_de_en/wmt16_de_en_train_0-199.json',
        'eval_data_path': './gptdynamics/sft_tasks/wmt16_de_en/wmt16_de_en_test_0-199.json'
    },
    'flan': {
        'train_data_path': './gptdynamics/it_tasks/flan_train_0_1599.json',
        'eval_data_path': './gptdynamics/it_tasks/flan_eval_0_1054.json'
    },
    'dataset_debug_sst2': {
        'train_data_path': './gptdynamics/sft_tasks/sst2/sst2_flan-prompt_dataset-debug_mislabelled-0.4_train_0-127.json',
        'eval_data_path': './gptdynamics/sft_tasks/sst2/sst2_flan-prompt_dataset-debug_eval_0-127.json'
    }
}

SIMULATORS = {
    'original': Simulator,
    'vec_sim': VectorSimulator,
    'xlmr_sim': XlmrSimulator,
    'enc_sim': EncSimulator,
    'norder_enc_sim': NOrder_EncSimulator,
    'tracincp_sim': TracInCPSimulator,
    'enc_cp_sim': EncSimulator,
    "gpt_sim": GPTSimulator,
}

SIMULATR_ADDIONAL_ARGS = {
    'original': {
        'eval_task': None
    },
    'vec_sim': {
        'emb_dim': 2
    },
    'xlmr_sim': {
        'xlmr_model_name_or_path': './model/xlm_roberta_base',
        'train_xlm_ids_file': './eval_data/wmt18/tr-en/en-tr_tgt-en_xlmr-ids.json',
        'test_xlm_ids_file': './eval_data/wmt18/tr-en/dev/newsdev2016-tren-tren_xlmr-ids.json',
    },
    'enc_sim': {
        'enc_model_name_or_path': '../hf_hub/models--sentence-transformers--all-MiniLM-L6-v2/',
        'frozen': True,
        'use_initial': True,
        'concate': False
    },
    'norder_enc_sim': {
        'order_n': 10,
        'enc_model_name_or_path': './model/models--sentence-transformers--all-MiniLM-L6-v2/',
        'frozen': True,
        'use_initial': True,
        'concate': True
    },
    "tracincp_sim": {},
    "enc_cp_sim": {
        'enc_model_name_or_path': './model/models--sentence-transformers--all-MiniLM-L6-v2/',
        'frozen': True,
        'use_initial': True,
        'concate': False,
        'cp_interval': 1,
    },
    "gpt_sim": {
        'enc_model_name_or_path': {
            '160m': './models--EleutherAI--pythia-160m-deduped/',
            '410m': '../alpaca-lora-main/models--EleutherAI--pythia-410m-deduped/',
            '1b': '../alpaca-lora-main/models--EleutherAI--pythia-1b-deduped/',
        },
        'frozen': True,
        'use_initial': True,
        'concate': False,
        'max_input_length': {
            '160m': 2048,
            '410m': 2048,
            '1b': 2048,
        },
        'model_size': None
    }
}

INPUT_ADDITIONAL_KEYS ={
    'original': {},
    'vec_sim': {},
    'xlmr_sim': {},
    'enc_sim': {
        'samples_texts',
        'test_sample_text',
    },
    'norder_enc_sim': {
        'samples_texts',
        'test_sample_text',
        'prev_n_steps',
        'prev_n_losses',
    },
    'tracincp_sim': {
        'samples_texts',
        'test_sample_text',
        'samples_contexts',
        'test_sample_context',
    },
    'enc_cp_sim': {
        'samples_texts',
        'test_sample_text',
    },
    'gpt_sim': {
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
    ],
    'norder_enc_sim': [
        'enc_model_name_or_path',
    ],
    'tracincp_sim': [],
    'enc_cp_sim': [
        'enc_model_name_or_path',
    ],
    'gpt_sim': [
        'enc_model_name_or_path',
        'max_input_length'
    ],
}

def train(
    sim_name, # [Options]: enc_sim | original
    dataset_name, # [Options]: rte | boolq | sst2 | webnlg | wmt16_de_en | flan | dataset_debug_sst2
    metric, # [Options]: loss | bleu | rougeL
    task, # [Options]: boolq | sst2 | rte | webnlg | wmt16_de_en | flan | flan_pythia-14m | flan_pythia-70m | flan_pythia-1b
    test_example_start_id,
    test_example_end_id,
    max_epoch = 2000,
    train_bs = 128,
    train_example_nums = 205756,
    test_example_nums = 10,
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
    order_n=None,
    concate=None,
    cp_interval=None,
    eval_task=None,
    model_size=None,
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
    data_paths_dict = {
        'boolq': [
            './runs/boolq/output_boolq_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-1/',
        './runs/boolq/output_boolq_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-2/',
        './runs/boolq/output_boolq_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-3/',
        './runs/boolq/output_boolq_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-4/',
        './runs/boolq/output_boolq_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-5/',
        './runs/boolq/output_boolq_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-6/',
        './runs/boolq/output_boolq_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-7/',
        './runs/boolq/output_boolq_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-8/',
        './runs/boolq/output_boolq_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-9/',
        './runs/boolq/output_boolq_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-10/',
        './runs/boolq/output_boolq_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-11/',
        './runs/boolq/output_boolq_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-12/',
        './runs/boolq/output_boolq_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-13/',
        './runs/boolq/output_boolq_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-14/',
        './runs/boolq/output_boolq_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-15/',
        './runs/boolq/output_boolq_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-16/',
        './runs/boolq/output_boolq_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-17/',
        './runs/boolq/output_boolq_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-18/',
        './runs/boolq/output_boolq_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-19/',
        './runs/boolq/output_boolq_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-20/',
        './runs/boolq/output_boolq_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-21/',
        './runs/boolq/output_boolq_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-22/',
        './runs/boolq/output_boolq_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-23/',
        './runs/boolq/output_boolq_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-24/',
        './runs/boolq/output_boolq_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-25/',
        './runs/boolq/output_boolq_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-26/',
        './runs/boolq/output_boolq_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-27/',
        ],
        'sst2': [
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
        ],
        'rte': [
            './runs/rte/output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-1/',
            './runs/rte/output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-2/',
            './runs/rte/output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-3/',
            './runs/rte/output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-4/',
            './runs/rte/output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-5/',
            './runs/rte/output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-6/',
            './runs/rte/output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-7/',
            './runs/rte/output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-8/',
            './runs/rte/output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-9/',
            './runs/rte/output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-10/',
            './runs/rte/output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-11/',
            './runs/rte/output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-12/',
            './runs/rte/output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-13/',
            './runs/rte/output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-14/',
            './runs/rte/output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-15/',
            './runs/rte/output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-16/',
            './runs/rte/output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-17/',
            './runs/rte/output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-18/',
            './runs/rte/output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-19/',
            './runs/rte/output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-20/',
            './runs/rte/output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-21/',
            './runs/rte/output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-22/',
            './runs/rte/output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-23/',
            './runs/rte/output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-24/',
            './runs/rte/output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-25/',
            './runs/rte/output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-26/',
            './runs/rte/output_rte_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_seed-27/',
        ],
        'webnlg': [
            './runs/webnlg/output_webnlg_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-1e-6_weight-decay-0.001_epoch-3_loss-output-token_seed-1',
        './runs/webnlg/output_webnlg_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-1e-6_weight-decay-0.001_epoch-3_loss-output-token_seed-2',
        './runs/webnlg/output_webnlg_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-1e-6_weight-decay-0.001_epoch-3_loss-output-token_seed-3',
        './runs/webnlg/output_webnlg_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-1e-6_weight-decay-0.001_epoch-3_loss-output-token_seed-4',
        './runs/webnlg/output_webnlg_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-1e-6_weight-decay-0.001_epoch-3_loss-output-token_seed-5',
        './runs/webnlg/output_webnlg_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-1e-6_weight-decay-0.001_epoch-3_loss-output-token_seed-6',
        './runs/webnlg/output_webnlg_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-1e-6_weight-decay-0.001_epoch-3_loss-output-token_seed-7',
        './runs/webnlg/output_webnlg_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-1e-6_weight-decay-0.001_epoch-3_loss-output-token_seed-8',
        './runs/webnlg/output_webnlg_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-1e-6_weight-decay-0.001_epoch-3_loss-output-token_seed-9',
        './runs/webnlg/output_webnlg_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-1e-6_weight-decay-0.001_epoch-3_loss-output-token_seed-10',
        './runs/webnlg/output_webnlg_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-1e-6_weight-decay-0.001_epoch-3_loss-output-token_seed-11',
        './runs/webnlg/output_webnlg_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-1e-6_weight-decay-0.001_epoch-3_loss-output-token_seed-12',
        './runs/webnlg/output_webnlg_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-1e-6_weight-decay-0.001_epoch-3_loss-output-token_seed-13',
        './runs/webnlg/output_webnlg_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-1e-6_weight-decay-0.001_epoch-3_loss-output-token_seed-14',
        './runs/webnlg/output_webnlg_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-1e-6_weight-decay-0.001_epoch-3_loss-output-token_seed-15',
        './runs/webnlg/output_webnlg_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-1e-6_weight-decay-0.001_epoch-3_loss-output-token_seed-16',
        './runs/webnlg/output_webnlg_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-1e-6_weight-decay-0.001_epoch-3_loss-output-token_seed-17',
        './runs/webnlg/output_webnlg_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-1e-6_weight-decay-0.001_epoch-3_loss-output-token_seed-18',
        './runs/webnlg/output_webnlg_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-1e-6_weight-decay-0.001_epoch-3_loss-output-token_seed-19',
        './runs/webnlg/output_webnlg_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-1e-6_weight-decay-0.001_epoch-3_loss-output-token_seed-20',
        './runs/webnlg/output_webnlg_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-1e-6_weight-decay-0.001_epoch-3_loss-output-token_seed-21',
        './runs/webnlg/output_webnlg_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-1e-6_weight-decay-0.001_epoch-3_loss-output-token_seed-22',
        './runs/webnlg/output_webnlg_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-1e-6_weight-decay-0.001_epoch-3_loss-output-token_seed-23',
        './runs/webnlg/output_webnlg_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-1e-6_weight-decay-0.001_epoch-3_loss-output-token_seed-24',
        './runs/webnlg/output_webnlg_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-1e-6_weight-decay-0.001_epoch-3_loss-output-token_seed-25',
        './runs/webnlg/output_webnlg_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-1e-6_weight-decay-0.001_epoch-3_loss-output-token_seed-26',
        './runs/webnlg/output_webnlg_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-1e-6_weight-decay-0.001_epoch-3_loss-output-token_seed-27',
        ],
        'wmt16_de_en': [
            'runs/wmt16_de_en/output_wmt16_de_en_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-1',
        'runs/wmt16_de_en/output_wmt16_de_en_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-10',
        'runs/wmt16_de_en/output_wmt16_de_en_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-11',
        'runs/wmt16_de_en/output_wmt16_de_en_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-17',
        'runs/wmt16_de_en/output_wmt16_de_en_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-18',
        'runs/wmt16_de_en/output_wmt16_de_en_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-19',
        'runs/wmt16_de_en/output_wmt16_de_en_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-2',
        'runs/wmt16_de_en/output_wmt16_de_en_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-20',
        'runs/wmt16_de_en/output_wmt16_de_en_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-21',
        'runs/wmt16_de_en/output_wmt16_de_en_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-22',
        'runs/wmt16_de_en/output_wmt16_de_en_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-23',
        'runs/wmt16_de_en/output_wmt16_de_en_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-24',
        'runs/wmt16_de_en/output_wmt16_de_en_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-25',
        'runs/wmt16_de_en/output_wmt16_de_en_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-26',
        'runs/wmt16_de_en/output_wmt16_de_en_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-27',
        'runs/wmt16_de_en/output_wmt16_de_en_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-28',
        'runs/wmt16_de_en/output_wmt16_de_en_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-29',
        'runs/wmt16_de_en/output_wmt16_de_en_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-3',
        'runs/wmt16_de_en/output_wmt16_de_en_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-33',
        'runs/wmt16_de_en/output_wmt16_de_en_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-34',
        'runs/wmt16_de_en/output_wmt16_de_en_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-35',
        'runs/wmt16_de_en/output_wmt16_de_en_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-36',
        'runs/wmt16_de_en/output_wmt16_de_en_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-4',
        'runs/wmt16_de_en/output_wmt16_de_en_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-43',
        'runs/wmt16_de_en/output_wmt16_de_en_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-44',
        'runs/wmt16_de_en/output_wmt16_de_en_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-45',
        'runs/wmt16_de_en/output_wmt16_de_en_bs-4_shot-200_sample-128_model-pythia-410m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-46',
        ],
        'flan': [
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-410m-deduped_lr-2e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-17/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-410m-deduped_lr-2e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-18/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-410m-deduped_lr-2e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-19/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-410m-deduped_lr-2e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-20/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-410m-deduped_lr-2e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-21/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-410m-deduped_lr-2e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-22/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-410m-deduped_lr-2e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-23/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-410m-deduped_lr-2e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-24/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-410m-deduped_lr-2e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-25/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-410m-deduped_lr-2e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-26/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-410m-deduped_lr-2e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-3/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-410m-deduped_lr-2e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-34/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-410m-deduped_lr-2e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-35/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-410m-deduped_lr-2e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-36/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-410m-deduped_lr-2e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-38/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-410m-deduped_lr-2e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-39/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-410m-deduped_lr-2e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-4/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-410m-deduped_lr-2e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-40/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-410m-deduped_lr-2e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-41/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-410m-deduped_lr-2e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-42/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-410m-deduped_lr-2e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-43/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-410m-deduped_lr-2e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-44/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-410m-deduped_lr-2e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-45/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-410m-deduped_lr-2e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-46/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-410m-deduped_lr-2e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-47/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-410m-deduped_lr-2e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-49/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-410m-deduped_lr-2e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-50/",
        ],
        'flan_pythia-1b': [
        "runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-1b-deduped_lr-2e-7_weight-decay-0.001_epoch-2_loss-output-token_seed-1/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-1b-deduped_lr-2e-7_weight-decay-0.001_epoch-2_loss-output-token_seed-10/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-1b-deduped_lr-2e-7_weight-decay-0.001_epoch-2_loss-output-token_seed-11/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-1b-deduped_lr-2e-7_weight-decay-0.001_epoch-2_loss-output-token_seed-12/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-1b-deduped_lr-2e-7_weight-decay-0.001_epoch-2_loss-output-token_seed-13/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-1b-deduped_lr-2e-7_weight-decay-0.001_epoch-2_loss-output-token_seed-14/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-1b-deduped_lr-2e-7_weight-decay-0.001_epoch-2_loss-output-token_seed-15/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-1b-deduped_lr-2e-7_weight-decay-0.001_epoch-2_loss-output-token_seed-16/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-1b-deduped_lr-2e-7_weight-decay-0.001_epoch-2_loss-output-token_seed-17/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-1b-deduped_lr-2e-7_weight-decay-0.001_epoch-2_loss-output-token_seed-18/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-1b-deduped_lr-2e-7_weight-decay-0.001_epoch-2_loss-output-token_seed-19/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-1b-deduped_lr-2e-7_weight-decay-0.001_epoch-2_loss-output-token_seed-2/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-1b-deduped_lr-2e-7_weight-decay-0.001_epoch-2_loss-output-token_seed-20/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-1b-deduped_lr-2e-7_weight-decay-0.001_epoch-2_loss-output-token_seed-21/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-1b-deduped_lr-2e-7_weight-decay-0.001_epoch-2_loss-output-token_seed-22/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-1b-deduped_lr-2e-7_weight-decay-0.001_epoch-2_loss-output-token_seed-23/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-1b-deduped_lr-2e-7_weight-decay-0.001_epoch-2_loss-output-token_seed-24/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-1b-deduped_lr-2e-7_weight-decay-0.001_epoch-2_loss-output-token_seed-25/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-1b-deduped_lr-2e-7_weight-decay-0.001_epoch-2_loss-output-token_seed-26/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-1b-deduped_lr-2e-7_weight-decay-0.001_epoch-2_loss-output-token_seed-27/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-1b-deduped_lr-2e-7_weight-decay-0.001_epoch-2_loss-output-token_seed-28/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-1b-deduped_lr-2e-7_weight-decay-0.001_epoch-2_loss-output-token_seed-29/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-1b-deduped_lr-2e-7_weight-decay-0.001_epoch-2_loss-output-token_seed-3/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-1b-deduped_lr-2e-7_weight-decay-0.001_epoch-2_loss-output-token_seed-30/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-1b-deduped_lr-2e-7_weight-decay-0.001_epoch-2_loss-output-token_seed-31/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-1b-deduped_lr-2e-7_weight-decay-0.001_epoch-2_loss-output-token_seed-32/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-1b-deduped_lr-2e-7_weight-decay-0.001_epoch-2_loss-output-token_seed-4/",
        ],
        'flan_pythia-160m': [
            "runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-160m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-1",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-160m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-10",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-160m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-11",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-160m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-12",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-160m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-13",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-160m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-14",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-160m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-15",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-160m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-16",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-160m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-17",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-160m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-18",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-160m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-19",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-160m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-2",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-160m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-20",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-160m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-21",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-160m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-22",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-160m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-23",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-160m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-24",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-160m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-25",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-160m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-26",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-160m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-27",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-160m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-28",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-160m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-29",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-160m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-3",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-160m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-30",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-160m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-31",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-160m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-32",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-160m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-4",
        ],
        'flan_pythia-70m': [
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-70m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-10/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-70m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-11/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-70m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-12/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-70m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-13/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-70m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-14/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-70m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-15/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-70m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-16/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-70m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-17/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-70m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-18/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-70m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-19/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-70m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-2/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-70m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-20/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-70m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-21/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-70m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-22/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-70m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-23/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-70m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-24/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-70m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-25/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-70m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-26/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-70m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-27/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-70m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-28/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-70m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-29/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-70m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-3/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-70m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-30/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-70m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-31/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-70m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-32/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-70m-deduped_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-4/",
        ],
        'flan_pythia-14m': [
            "runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-14m_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-1/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-14m_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-10/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-14m_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-11/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-14m_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-12/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-14m_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-13/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-14m_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-14/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-14m_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-15/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-14m_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-16/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-14m_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-17/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-14m_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-18/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-14m_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-19/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-14m_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-2/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-14m_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-20/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-14m_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-21/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-14m_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-22/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-14m_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-23/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-14m_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-24/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-14m_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-25/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-14m_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-26/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-14m_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-27/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-14m_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-28/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-14m_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-29/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-14m_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-3/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-14m_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-30/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-14m_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-31/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-14m_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-32/",
"runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-14m_lr-5e-7_weight-decay-0.001_epoch-3_loss-output-token_seed-4/",
        ],
        'flan_pythia-2.8b': [
            "runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-2.8b-deduped_lr-1e-5_weight-decay-0.001_epoch-2_loss-output-token_seed-1/",
    "runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-2.8b-deduped_lr-1e-5_weight-decay-0.001_epoch-2_loss-output-token_seed-10/",
    "runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-2.8b-deduped_lr-1e-5_weight-decay-0.001_epoch-2_loss-output-token_seed-11/",
    "runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-2.8b-deduped_lr-1e-5_weight-decay-0.001_epoch-2_loss-output-token_seed-12/",
    "runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-2.8b-deduped_lr-1e-5_weight-decay-0.001_epoch-2_loss-output-token_seed-13/",
    "runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-2.8b-deduped_lr-1e-5_weight-decay-0.001_epoch-2_loss-output-token_seed-14/",
    "runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-2.8b-deduped_lr-1e-5_weight-decay-0.001_epoch-2_loss-output-token_seed-15/",
    "runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-2.8b-deduped_lr-1e-5_weight-decay-0.001_epoch-2_loss-output-token_seed-16/",
    "runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-2.8b-deduped_lr-1e-5_weight-decay-0.001_epoch-2_loss-output-token_seed-17/",
    "runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-2.8b-deduped_lr-1e-5_weight-decay-0.001_epoch-2_loss-output-token_seed-18/",
    "runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-2.8b-deduped_lr-1e-5_weight-decay-0.001_epoch-2_loss-output-token_seed-19/",
    "runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-2.8b-deduped_lr-1e-5_weight-decay-0.001_epoch-2_loss-output-token_seed-2/",
    "runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-2.8b-deduped_lr-1e-5_weight-decay-0.001_epoch-2_loss-output-token_seed-20/",
    "runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-2.8b-deduped_lr-1e-5_weight-decay-0.001_epoch-2_loss-output-token_seed-21/",
    "runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-2.8b-deduped_lr-1e-5_weight-decay-0.001_epoch-2_loss-output-token_seed-22/",
    "runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-2.8b-deduped_lr-1e-5_weight-decay-0.001_epoch-2_loss-output-token_seed-23/",
    "runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-2.8b-deduped_lr-1e-5_weight-decay-0.001_epoch-2_loss-output-token_seed-24/",
    "runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-2.8b-deduped_lr-1e-5_weight-decay-0.001_epoch-2_loss-output-token_seed-25/",
    "runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-2.8b-deduped_lr-1e-5_weight-decay-0.001_epoch-2_loss-output-token_seed-26/",
    "runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-2.8b-deduped_lr-1e-5_weight-decay-0.001_epoch-2_loss-output-token_seed-27/",
    "runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-2.8b-deduped_lr-1e-5_weight-decay-0.001_epoch-2_loss-output-token_seed-28/",
    "runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-2.8b-deduped_lr-1e-5_weight-decay-0.001_epoch-2_loss-output-token_seed-29/",
    "runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-2.8b-deduped_lr-1e-5_weight-decay-0.001_epoch-2_loss-output-token_seed-3/",
    "runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-2.8b-deduped_lr-1e-5_weight-decay-0.001_epoch-2_loss-output-token_seed-30/",
    "runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-2.8b-deduped_lr-1e-5_weight-decay-0.001_epoch-2_loss-output-token_seed-31/",
    "runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-2.8b-deduped_lr-1e-5_weight-decay-0.001_epoch-2_loss-output-token_seed-32/",
    "runs/flan/output_flan_bs-8_shot-200_sample-128_model-pythia-2.8b-deduped_lr-1e-5_weight-decay-0.001_epoch-2_loss-output-token_seed-4/",
        ],
        'debug': [
            'runs/rte/output_rte_bs-4_shot-200_sample-128_lr-2e-6_weight-decay-0.001_epoch-3_seed-1',
            'runs/rte/output_rte_bs-4_shot-200_sample-128_lr-2e-6_weight-decay-0.001_epoch-3_seed-1',
            'runs/rte/output_rte_bs-4_shot-200_sample-128_lr-2e-6_weight-decay-0.001_epoch-3_seed-1',
        ],
        'dataset_debug_sst2': [
            "runs/dataset_debug_sst2/output_sst2-dataset-debug-is-mislabelled_bs-4_shot-128_sample-128_model-pythia-410m-deduped_lr-1e-06_weight-decay-0.001_epoch-1_loss-output-token_seed-1/",
"runs/dataset_debug_sst2/output_sst2-dataset-debug-is-mislabelled_bs-4_shot-128_sample-128_model-pythia-410m-deduped_lr-1e-06_weight-decay-0.001_epoch-1_loss-output-token_seed-10/",
"runs/dataset_debug_sst2/output_sst2-dataset-debug-is-mislabelled_bs-4_shot-128_sample-128_model-pythia-410m-deduped_lr-1e-06_weight-decay-0.001_epoch-1_loss-output-token_seed-11/",
"runs/dataset_debug_sst2/output_sst2-dataset-debug-is-mislabelled_bs-4_shot-128_sample-128_model-pythia-410m-deduped_lr-1e-06_weight-decay-0.001_epoch-1_loss-output-token_seed-12/",
"runs/dataset_debug_sst2/output_sst2-dataset-debug-is-mislabelled_bs-4_shot-128_sample-128_model-pythia-410m-deduped_lr-1e-06_weight-decay-0.001_epoch-1_loss-output-token_seed-13/",
"runs/dataset_debug_sst2/output_sst2-dataset-debug-is-mislabelled_bs-4_shot-128_sample-128_model-pythia-410m-deduped_lr-1e-06_weight-decay-0.001_epoch-1_loss-output-token_seed-14/",
"runs/dataset_debug_sst2/output_sst2-dataset-debug-is-mislabelled_bs-4_shot-128_sample-128_model-pythia-410m-deduped_lr-1e-06_weight-decay-0.001_epoch-1_loss-output-token_seed-15/",
"runs/dataset_debug_sst2/output_sst2-dataset-debug-is-mislabelled_bs-4_shot-128_sample-128_model-pythia-410m-deduped_lr-1e-06_weight-decay-0.001_epoch-1_loss-output-token_seed-16/",
"runs/dataset_debug_sst2/output_sst2-dataset-debug-is-mislabelled_bs-4_shot-128_sample-128_model-pythia-410m-deduped_lr-1e-06_weight-decay-0.001_epoch-1_loss-output-token_seed-17/",
"runs/dataset_debug_sst2/output_sst2-dataset-debug-is-mislabelled_bs-4_shot-128_sample-128_model-pythia-410m-deduped_lr-1e-06_weight-decay-0.001_epoch-1_loss-output-token_seed-18/",
"runs/dataset_debug_sst2/output_sst2-dataset-debug-is-mislabelled_bs-4_shot-128_sample-128_model-pythia-410m-deduped_lr-1e-06_weight-decay-0.001_epoch-1_loss-output-token_seed-19/",
"runs/dataset_debug_sst2/output_sst2-dataset-debug-is-mislabelled_bs-4_shot-128_sample-128_model-pythia-410m-deduped_lr-1e-06_weight-decay-0.001_epoch-1_loss-output-token_seed-2/",
"runs/dataset_debug_sst2/output_sst2-dataset-debug-is-mislabelled_bs-4_shot-128_sample-128_model-pythia-410m-deduped_lr-1e-06_weight-decay-0.001_epoch-1_loss-output-token_seed-20/",
"runs/dataset_debug_sst2/output_sst2-dataset-debug-is-mislabelled_bs-4_shot-128_sample-128_model-pythia-410m-deduped_lr-1e-06_weight-decay-0.001_epoch-1_loss-output-token_seed-21/",
"runs/dataset_debug_sst2/output_sst2-dataset-debug-is-mislabelled_bs-4_shot-128_sample-128_model-pythia-410m-deduped_lr-1e-06_weight-decay-0.001_epoch-1_loss-output-token_seed-22/",
"runs/dataset_debug_sst2/output_sst2-dataset-debug-is-mislabelled_bs-4_shot-128_sample-128_model-pythia-410m-deduped_lr-1e-06_weight-decay-0.001_epoch-1_loss-output-token_seed-23/",
"runs/dataset_debug_sst2/output_sst2-dataset-debug-is-mislabelled_bs-4_shot-128_sample-128_model-pythia-410m-deduped_lr-1e-06_weight-decay-0.001_epoch-1_loss-output-token_seed-24/",
"runs/dataset_debug_sst2/output_sst2-dataset-debug-is-mislabelled_bs-4_shot-128_sample-128_model-pythia-410m-deduped_lr-1e-06_weight-decay-0.001_epoch-1_loss-output-token_seed-25/",
"runs/dataset_debug_sst2/output_sst2-dataset-debug-is-mislabelled_bs-4_shot-128_sample-128_model-pythia-410m-deduped_lr-1e-06_weight-decay-0.001_epoch-1_loss-output-token_seed-26/",
"runs/dataset_debug_sst2/output_sst2-dataset-debug-is-mislabelled_bs-4_shot-128_sample-128_model-pythia-410m-deduped_lr-1e-06_weight-decay-0.001_epoch-1_loss-output-token_seed-27/",
"runs/dataset_debug_sst2/output_sst2-dataset-debug-is-mislabelled_bs-4_shot-128_sample-128_model-pythia-410m-deduped_lr-1e-06_weight-decay-0.001_epoch-1_loss-output-token_seed-28/",
"runs/dataset_debug_sst2/output_sst2-dataset-debug-is-mislabelled_bs-4_shot-128_sample-128_model-pythia-410m-deduped_lr-1e-06_weight-decay-0.001_epoch-1_loss-output-token_seed-29/",
"runs/dataset_debug_sst2/output_sst2-dataset-debug-is-mislabelled_bs-4_shot-128_sample-128_model-pythia-410m-deduped_lr-1e-06_weight-decay-0.001_epoch-1_loss-output-token_seed-3/",
"runs/dataset_debug_sst2/output_sst2-dataset-debug-is-mislabelled_bs-4_shot-128_sample-128_model-pythia-410m-deduped_lr-1e-06_weight-decay-0.001_epoch-1_loss-output-token_seed-30/",
"runs/dataset_debug_sst2/output_sst2-dataset-debug-is-mislabelled_bs-4_shot-128_sample-128_model-pythia-410m-deduped_lr-1e-06_weight-decay-0.001_epoch-1_loss-output-token_seed-31/",
"runs/dataset_debug_sst2/output_sst2-dataset-debug-is-mislabelled_bs-4_shot-128_sample-128_model-pythia-410m-deduped_lr-1e-06_weight-decay-0.001_epoch-1_loss-output-token_seed-32/",
"runs/dataset_debug_sst2/output_sst2-dataset-debug-is-mislabelled_bs-4_shot-128_sample-128_model-pythia-410m-deduped_lr-1e-06_weight-decay-0.001_epoch-1_loss-output-token_seed-4/",
"runs/dataset_debug_sst2/output_sst2-dataset-debug-is-mislabelled_bs-4_shot-128_sample-128_model-pythia-410m-deduped_lr-1e-06_weight-decay-0.001_epoch-1_loss-output-token_seed-5/",
"runs/dataset_debug_sst2/output_sst2-dataset-debug-is-mislabelled_bs-4_shot-128_sample-128_model-pythia-410m-deduped_lr-1e-06_weight-decay-0.001_epoch-1_loss-output-token_seed-6/",
"runs/dataset_debug_sst2/output_sst2-dataset-debug-is-mislabelled_bs-4_shot-128_sample-128_model-pythia-410m-deduped_lr-1e-06_weight-decay-0.001_epoch-1_loss-output-token_seed-7/",
"runs/dataset_debug_sst2/output_sst2-dataset-debug-is-mislabelled_bs-4_shot-128_sample-128_model-pythia-410m-deduped_lr-1e-06_weight-decay-0.001_epoch-1_loss-output-token_seed-8/",
"runs/dataset_debug_sst2/output_sst2-dataset-debug-is-mislabelled_bs-4_shot-128_sample-128_model-pythia-410m-deduped_lr-1e-06_weight-decay-0.001_epoch-1_loss-output-token_seed-9/",
        ]
    }
    # data_paths = [
    # ]

    print(
        # f"data_paths: {data_paths}\n",
        f"dataset_name: {dataset_name}\n",
        f"metric: {metric}\n", # loss | bleu | rouge
        f"task: {task}\n",
        f"sim_name: {sim_name}\n",
        f"max_epoch: {max_epoch}\n",
        f"train_bs: {train_bs}\n",
        f"lr: {lr}\n",
        f"train_example_nums: {train_example_nums}\n",
        f"test_example_nums: {test_example_nums}\n",
        # f"valid_bs: {valid_bs}\n",
        f"seed: {seed}\n", # 种子
        f"hyper_parameter: {hyper_parameter}\n"
        f"valid_epoch_interval: {valid_epoch_interval}\n",
        f"save_epoch_interval: {save_epoch_interval}\n",
        f"valid_num: {valid_num}\n",
        f"test_num: {test_num}\n",
        f"weight_decay: {weight_decay}\n",
        f"output_dir: {output_dir}\n",
        f"step_thres: {step_thres}\n"
        f"test_example_start_id: {test_example_start_id}\n",
        f"test_example_end_id: {test_example_end_id}\n",
    )

    log_dir = "tb-logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 设置tensorboard日志保存路径
    save_dir_name = f'/{task}/{sim_name}_task-{task}_metric-{metric}_lr-{lr}_lambda-{hyper_parameter}_wd-{weight_decay}_bs-{train_bs}_train-sample-nums-{train_example_nums}_test-sample-nums-{test_example_nums}_seed-{seed}_step_thres-{step_thres}_max_epoch-{max_epoch}'
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
            raise NotImplementedError()
    if cp_interval is not None and 'cp_interval' in simulator_args.keys():
        print(f"重写cp_interval: {cp_interval}")
        simulator_args['cp_interval'] = cp_interval
    if eval_task is not None:
        if sim_name == 'original':
            print(f"重写eval_task： {eval_task}")
            simulator_args['eval_task'] = eval_task
        else:
            raise NotImplementedError()
    if model_size is not None and 'model_size' in simulator_args.keys():
        print(f"重写model_size: {model_size}")
        simulator_args['model_size'] = model_size
        simulator_args['enc_model_name_or_path'] = simulator_args['enc_model_name_or_path'][model_size]
        simulator_args['max_input_length'] = simulator_args['max_input_length'][model_size]

    ignore_args = SAVE_DIR_IGNORED_ARG_NAME[sim_name]
    for args_name, args_value in simulator_args.items():
        if args_name in ignore_args:
            continue
        save_dir_name += f'_{args_name}-{args_value}'
    writer = SummaryWriter(log_dir=log_dir + save_dir_name)


    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   

    data_paths = data_paths_dict[task]

    random.shuffle(data_paths)
    valid_num = valid_num
    test_num = test_num
    
    if sim_name == 'norder_enc_sim':
        order_n = simulator_args['order_n']
        print(f"order_n: {order_n}\n")
    else:
        order_n = -1
    valid_dataset = SimfluenceDataset(data_paths[:valid_num], is_train=False, test_example_nums=test_example_nums, test_example_start_id=test_example_start_id, test_example_end_id=test_example_end_id, step_thres=step_thres, metric=metric, order_n=order_n)
    train_dataset = SimfluenceDataset(data_paths[valid_num + test_num:], test_example_nums=test_example_nums, test_example_start_id=test_example_start_id, test_example_end_id=test_example_end_id, step_thres=step_thres, metric=metric, order_n=order_n, cp_interval=cp_interval)
    


    print('')
    print(f'valid dataset: {data_paths[:valid_num]}')
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

    train_data_loader = DataLoader(train_dataset, batch_size=train_bs, shuffle=True, collate_fn=lambda x: train_dataset.collate_fn(x, device=device))

    # 加载simulator
    model = SIMULATORS[sim_name](train_example_nums=train_example_nums, hyper_parameter=hyper_parameter, test_example_nums=test_example_nums, **simulator_args)
    model.to(device).train()

    if sim_name == 'enc_sim' or sim_name == 'norder_enc_sim' or sim_name == 'enc_cp_sim' or sim_name == 'gpt_sim':
        if simulator_args['use_initial']:
            model._get_initial_embeds(train_dataset, device)

    # criterion = nn.MSELoss(reduction="mean")                                                
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)  
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=200, num_training_steps=max_epoch*len(train_data_loader))    
    # scheduler = get_constant_schedule(optimizer)

    train_losses = []
    tot_step = 0
    min_valid_mse = float('inf')
    prev_step_valid_mse = float('inf')
    early_stop_N = 10
    early_stop_count = 0
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
                kwargs=input_kwargs,
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
            res = eval_simulator(
                eval_dataset=valid_dataset,
                model=model,
                device=device,
                input_kwargs_keys=input_kwargs_keys,
            )
            all_steps_mse = res['all_steps_mse_mean']
            all_steps_mae  =res['all_steps_mae_mean']
            writer.add_scalar('valid_all_steps_mse', all_steps_mse, epoch)
            writer.add_scalar('valid_all_steps_mae', all_steps_mae, epoch)
            model.train()
            print(f'valid mse loss {all_steps_mse}')
            print(f'valid mae loss {all_steps_mae}')
        
        # 保存模型
        if epoch % save_epoch_interval == 0:
            net_save_dir = f'{output_dir}/{save_dir_name}'
            if not os.path.exists(net_save_dir):
                os.makedirs(net_save_dir)
            # 每一步保存模型 ###############################
            # net_save_path = os.path.join(net_save_dir, f'checkpoint-{epoch}.pt')
            #############################################
            
            if all_steps_mse < min_valid_mse:
                min_valid_mse = all_steps_mse
                net_save_path = os.path.join(net_save_dir, 'best-checkpoint.pt')
                torch.save(model.state_dict(), net_save_path)
                print(f'save best checkpoint at epoch: {epoch}')

            if all_steps_mse > prev_step_valid_mse:
                early_stop_count += 1
            else:
                early_stop_count = 0

            prev_step_valid_mse = all_steps_mse
            
            # early stop
            if early_stop_count == early_stop_N:
                print('early stop, exit。')
                net_save_path = os.path.join(net_save_dir, 'last-checkpoint.pt')
                torch.save(model.state_dict(), net_save_path)
                print('save last checkpoint')
                return
            
            # 保存最后一个epoch的checkpoint
            if epoch == (max_epoch-1):
                net_save_path = os.path.join(net_save_dir, 'last-checkpoint.pt')
                torch.save(model.state_dict(), net_save_path)
                print('save last checkpoint')


if __name__ == "__main__":
    fire.Fire(train)
