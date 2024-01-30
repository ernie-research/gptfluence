from torch.utils.data import Dataset
import json
import os
import torch
import numpy as np

class SimfluenceDataset(Dataset):
    def __init__(self, paths, is_train=True, test_example_nums=10, step_thres=None, test_example_start_id=-1, test_example_end_id=-1, metric=""):
        self.test_example_nums = test_example_nums
        # test_example_start_id和test_example_end_id是字符串，通过`,`分隔，表示多个区间
        if isinstance(test_example_start_id, str) and isinstance(test_example_end_id, str):
            self.test_example_start_id = list(map(int, test_example_start_id.split(',')))
            self.test_example_end_id = list(map(int, test_example_end_id.split(',')))
        elif isinstance(test_example_start_id, int) and isinstance(test_example_end_id, int):
            assert test_example_start_id != -1 and test_example_end_id != -1, '请设置test_example_start_id和test_example_end_id'
            # assert (test_example_end_id - test_example_start_id + 1) == test_example_nums, 'test_example_nums与test_example_start_id和test_example_end_id不匹配'
            self.test_example_start_id = [test_example_start_id]
            self.test_example_end_id = [test_example_end_id]
        self.is_train = is_train
        self.dataset = list()
        self.step_thres = step_thres
        assert metric != "", "metric未设置"
        self.metric = metric

        if is_train:
            for path in paths:
                cur_dataset = self._load_dataset(path)
                self.dataset += cur_dataset
                print(path, len(cur_dataset))
        else:
            for path in paths:
                cur_dataset = self._load_dataset(path)
                self.dataset.extend(cur_dataset)
                print(path, len(cur_dataset))
                
            

    def __getitem__(self, index):
        return self.dataset[index]
    def __len__(self):
        return len(self.dataset)

    def _load_dataset(self, path):
        '''
        
        Args:
            path
                train_samples_id.json

                all_loss_trajectory.out
            
            
            step: list
            sample_id: list
            metric1: list
            metric2: list
        '''
        # 加载train samples id
        train_samples_id_path = os.path.join(path, 'train_samples_id.json')
        train_samples_id = {}
        with open(train_samples_id_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = json.loads(line)
                # 将line id转换为0, 1, 2, ...
                # train_samples_id[line['step']] = (np.array(line['samples_id']) // 5 - 1).tolist()
                train_samples_id[line['step']] = line['samples_id']
        
        # 加载metric trajectory
        all_loss_trajectory_path = os.path.join(path, f'all_{self.metric}_trajectory.out')
        # test_sample_id = 0
        simulator_train_data = list()
        if self.is_train:
            with open(all_loss_trajectory_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    test_sample_loss_trajectory = list()
                    line = json.loads(line)
                    test_sample_id = line['id']

                    in_the_range_flag = False
                    for start_id, end_id in zip(self.test_example_start_id, self.test_example_end_id):
                        # 如果在区间之内
                        if test_sample_id >= start_id and test_sample_id <= end_id:
                            in_the_range_flag = True
                            break
                    if not in_the_range_flag:
                        
                        continue

                    loss_trajectory = line['loss_trajectory']
                    # if line['id'] == test_sample_id:
                    #     loss_trajectory = line['loss_trajectory']
                    # # debug
                    # if 'seed-37' in path and i == 682:
                    #     for i in range(1,len(loss_trajectory)):
                    #         if loss_trajectory[i]['step'] - loss_trajectory[i-1]['step'] > 1:
                    #             print(loss_trajectory[i]['step'])
                    #             import pdb; pdb.set_trace()
                    for l in loss_trajectory:
                        test_sample_loss_trajectory.append({'step': l['step'], 'loss': l['loss']})
                        # break


                    # 构造训练数据
                    for prev, cur in zip(test_sample_loss_trajectory[:-1], test_sample_loss_trajectory[1:]):
                        prev_step, prev_loss = prev['step'], prev['loss']
                        cur_step, cur_loss = cur['step'], cur['loss']
                        if self.step_thres is not None:
                            if cur_step < self.step_thres:
                                continue
                        samples_id = train_samples_id[cur_step]
                        simulator_train_data.append({'prev_step': prev_step, 'prev_loss': prev_loss, 'cur_step': cur_step, 'cur_loss': cur_loss, 'samples_id': samples_id, 'test_sample_id': test_sample_id})
        else:
            with open(all_loss_trajectory_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    simulator_train_data_list = list()
                    test_sample_loss_trajectory = list()
                    line = json.loads(line)
                    test_sample_id = line['id']
                    # if test_sample_id < self.test_example_start_id or test_sample_id > self.test_example_end_id:
                    # # if i >= self.test_example_nums:
                    #     continue

                    in_the_range_flag = False
                    for start_id, end_id in zip(self.test_example_start_id, self.test_example_end_id):
                        # 如果在区间之内
                        if test_sample_id >= start_id and test_sample_id <= end_id:
                            in_the_range_flag = True
                            break
                    if not in_the_range_flag:
                        continue

                    loss_trajectory = line['loss_trajectory']
                    # if line['id'] == test_sample_id:
                    #     loss_trajectory = line['loss_trajectory']
                    for l in loss_trajectory:
                        test_sample_loss_trajectory.append({'step': l['step'], 'loss': l['loss']})
                        # break
                    # 构造训练数据
                    for prev, cur in zip(test_sample_loss_trajectory[:-1], test_sample_loss_trajectory[1:]):
                        prev_step, prev_loss = prev['step'], prev['loss']
                        cur_step, cur_loss = cur['step'], cur['loss']
                        if self.step_thres is not None:
                            if cur_step < self.step_thres:
                                continue
                        samples_id = train_samples_id[cur_step]
                        simulator_train_data_list.append({'prev_step': prev_step, 'prev_loss': prev_loss, 'cur_step': cur_step, 'cur_loss': cur_loss, 'samples_id': samples_id, 'test_sample_id': test_sample_id})
                    simulator_train_data.append(simulator_train_data_list)
        return simulator_train_data
    
    @staticmethod
    def collate_fn(data, device=None):
        prev_step = [d['prev_step'] for d in data]
        prev_loss = torch.tensor([d['prev_loss'] for d in data]).to(device)
        cur_step = [d['cur_step'] for d in data]
        cur_loss = torch.tensor([d['cur_loss'] for d in data]).to(device)
        samples_id = torch.tensor([d['samples_id'] for d in data]).to(device)
        test_sample_id = [d['test_sample_id'] for d in data]
        return {
            'prev_step': prev_step,
            'prev_loss': prev_loss,
            'cur_step': cur_step,
            'cur_loss': cur_loss,
            'samples_id': samples_id,
            'test_sample_id': test_sample_id,
        }
        
    

if __name__ == '__main__':
    # 测试SimfluenceDataset
    path = [
        '/root/paddlejob/workspace/liuqingyi01/code/Simfluence/runs/wmt18-tr-en_tgt-en_loss-only-tgt_bs-1024_seed-42'
    ]
    dataset = SimfluenceDataset(path)
    print("finished")