from typing import Any
from torch.utils.data import Dataset
from datasets import load_dataset
import os
import sys
import logging
import torch
import json
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("dataset.tracincp_dataset")

class TracInCPDataset(Dataset):
    def __init__(self, simfluencedataset, is_train, **kwargs):
        self.is_train = is_train
        self.simfluencedataset = simfluencedataset

        # 加载训练文本数据，包括`input`和`context`
        train_data_path = kwargs['train_data_path']
        self.id_to_text_train = {}
        self.id_to_context_train = {}
        with open(train_data_path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                line = json.loads(line)
                self.id_to_text_train[line["id"]] = line["input"]
                self.id_to_context_train[line["id"]] = line["context"]
        
        # 加载评估文本数据，包括`input`和`context`
        eval_data_path = kwargs['eval_data_path']
        self.id_to_text_eval = {}
        self.id_to_context_eval = {}
        with open(eval_data_path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                line = json.loads(line)
                self.id_to_text_eval[line["id"]] = line["input"]
                self.id_to_context_eval[line["id"]] = line["context"]

        
    def __getitem__(self, index) -> Any:
        '''将数据集中的样本索引映射到原始数据集的样本，并添加到example中'''
        
        simfluence_data = self.simfluencedataset[index]

        # 模拟器的验证集或测试集按照list组织，需要将list中的每个样本转换为text
        if self.is_train == False:
            # assert not isinstance(simfluence_data, list)
            for i, d in enumerate(simfluence_data):
                simfluence_data[i]['samples_texts'] = [self.id_to_text_train[sample_id] for sample_id in d['samples_id']] 
                simfluence_data[i]['test_sample_text'] = [self.id_to_text_eval[d['test_sample_id']]]
                simfluence_data[i]['samples_contexts'] = [self.id_to_context_train[sample_id] for sample_id in d['samples_id']] 
                simfluence_data[i]['test_sample_context'] = [self.id_to_context_eval[d['test_sample_id']]]
            return simfluence_data
        else: # 模拟器训练集按照dict组织，不需要做任何处理
            samples_id = simfluence_data['samples_id']
            samples_texts = [ self.id_to_text_train[sample_id] for sample_id in samples_id ]
            samples_contexts = [ self.id_to_context_train[sample_id] for sample_id in samples_id ]

            test_sample_id = simfluence_data['test_sample_id']
            test_sample_text = [self.id_to_text_eval[test_sample_id]]
            test_sample_context = [self.id_to_context_eval[test_sample_id]]
            
            simfluence_data['samples_texts'] = samples_texts
            simfluence_data['test_sample_text'] = test_sample_text
            simfluence_data['samples_contexts'] = samples_contexts
            simfluence_data['test_sample_context'] = test_sample_context
            return simfluence_data
        
    def __len__(self) -> int:
        return len(self.simfluencedataset)
    
    # def collate_fn(self, data, device=None):
    #     res = {}
    #     prev_step = [d['prev_step'] for d in data]
    #     res['prev_step'] = prev_step
    #     prev_loss = torch.tensor([d['prev_loss'] for d in data]).to(device)
    #     res['prev_loss'] = prev_loss
    #     cur_step = [d['cur_step'] for d in data]
    #     res['cur_step'] = cur_step
    #     cur_loss = torch.tensor([d['cur_loss'] for d in data]).to(device)
    #     res['cur_loss'] = cur_loss
    #     samples_id = torch.tensor([d['samples_id'] for d in data]).to(device)
    #     res['samples_id'] = samples_id
    #     test_sample_id = torch.tensor([d['test_sample_id'] for d in data]).to(device)
    #     res['test_sample_id'] = test_sample_id
    #     samples_texts = [d['samples_texts'] for d in data]
    #     res['samples_texts'] = samples_texts
    #     test_sample_text = [d['test_sample_text'] for d in data]
    #     res['test_sample_text'] = test_sample_text
    #     # 处理n阶markov
    #     keys = data[0].keys()
    #     if 'prev_n_steps' in keys and 'prev_n_losses' in keys:
    #         prev_steps = [d['prev_n_steps'] for d in data]
    #         res['prev_n_steps'] = prev_steps
    #         prev_losses = torch.tensor([d['prev_n_losses'] for d in data]).to(device)
    #         res['prev_n_losses'] = prev_losses
    #     return res

if __name__ == "__main__":
    from dataset.simfluence_dataset import SimfluenceDataset
    simfluencedataset = SimfluenceDataset(
        paths=['/root/paddlejob/workspace/liuqingyi01/code/Simfluence/runs/rte/output_rte_bs-4_shot-200_sample-128_lr-2e-6_weight-decay-0.001_epoch-3_seed-1'],
        test_example_start_id=0,
        test_example_end_id=199,
        metric='loss'
    )
    
    kwargs = {
        'train_data_path': '/root/paddlejob/workspace/env_run/liuqingyi01/data/eval_data/tda_tasks/rte/rte_0_199_train.json',
        'eval_data_path': '/root/paddlejob/workspace/env_run/liuqingyi01/data/eval_data/tda_tasks/rte/rte_0_199_eval.json'
    }
    dataset = TracInCPDataset(simfluencedataset, is_train=True, **kwargs)
    print(len(dataset))
    print(dataset[0])
    print('done')