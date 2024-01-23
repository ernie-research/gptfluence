from typing import Any
from torch.utils.data import Dataset
from datasets import load_dataset
import os
import sys
import logging
import torch
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("dataset.rte_dataset")

class RteDataset(Dataset):
    def __init__(self, simfluencedataset, is_train, **kwargs):
        self.is_train = is_train
        self.simfluencedataset = simfluencedataset
        data_path = kwargs['data_path']
        self.rte_dataset = load_dataset(data_path, 'rte')
        # rte数据选择验证集样本进行评估
        logger.warning('请检查RTE任务的评估样本的数据集split来源，当前默认使用 `validation` 样本')
        self.rte_train = self.rte_dataset['train']
        self.rte_eval = self.rte_dataset['validation']

        self.id_to_sample_train = {sample['idx']: sample for sample in self.rte_train}
        self.id_to_sample_eval = {sample['idx']: sample for sample in self.rte_eval}
    
    def __getitem__(self, index) -> Any:
        '''将数据集中的样本索引映射到原始数据集的样本，并添加到example中'''
        
        simfluence_data = self.simfluencedataset[index]

        # 模拟器的验证集或测试集按照list组织，需要将list中的每个样本转换为text
        if self.is_train == False:
            # assert not isinstance(simfluence_data, list)
            for i, d in enumerate(simfluence_data):
                simfluence_data[i]['samples_texts'] = [self._get_text(self.id_to_sample_train[sample_id]) for sample_id in d['samples_id']] 
                simfluence_data[i]['test_sample_text'] = [self._get_text(self.id_to_sample_eval[d['test_sample_id']])]
            return simfluence_data
        else: # 模拟器训练集按照dict组织，不需要做任何处理
            samples_id = simfluence_data['samples_id']
            samples_texts = [ self._get_text(self.id_to_sample_train[sample_id]) for sample_id in samples_id ]
            

            test_sample_id = simfluence_data['test_sample_id']
            test_sample_text = [self._get_text(self.id_to_sample_eval[test_sample_id])]
            
            simfluence_data['samples_texts'] = samples_texts
            simfluence_data['test_sample_text'] = test_sample_text
            return simfluence_data

    def __len__(self) -> int:
        return len(self.simfluencedataset)
    
    def _get_text(self, sample):
        '''将sample转换为text
        Args:
            sample (dict): {'setence1': '...', 'sentence2': '...', ''label': 0/1, ''idx': ...}
        '''
        sentence1 = sample['sentence1']
        sentence2 = sample['sentence2']
        label_id = sample['label']
        if label_id == 1:
            label = False
        else:
            label = True
        text = f"{{{sentence1}}}\nQuestion: {{{sentence2}}} True or False?\nAnswer: {label}."
        
        return text
    
    def collate_fn(self, data, device=None):
        prev_step = [d['prev_step'] for d in data]
        prev_loss = torch.tensor([d['prev_loss'] for d in data]).to(device)
        cur_step = [d['cur_step'] for d in data]
        cur_loss = torch.tensor([d['cur_loss'] for d in data]).to(device)
        samples_id = torch.tensor([d['samples_id'] for d in data]).to(device)
        test_sample_id = torch.tensor([d['test_sample_id'] for d in data]).to(device)
        samples_texts = [d['samples_texts'] for d in data]
        test_sample_text = [d['test_sample_text'] for d in data]
        return {
            'prev_step': prev_step,
            'prev_loss': prev_loss,
            'cur_step': cur_step,
            'cur_loss': cur_loss,
            'samples_id': samples_id,
            'test_sample_id': test_sample_id,
            'samples_texts': samples_texts,
            'test_sample_text': test_sample_text,
        }
    
if __name__ == '__main__':
    from dataset.simfluence_dataset import SimfluenceDataset
    simfluencedataset = SimfluenceDataset(
        paths=['/root/paddlejob/workspace/liuqingyi01/code/Simfluence/runs/rte/output_rte_bs-4_shot-200_sample-128_lr-2e-6_weight-decay-0.001_epoch-3_seed-1'],
    )
    
    dataset = RteDataset(simfluencedataset, '/root/paddlejob/workspace/env_run/liuqingyi01/data/eval_data/datasets--glue')
    print(len(dataset))
    print(dataset[0])
    print('done')
