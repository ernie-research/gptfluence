import torch
from torch import nn
from typing import List, Optional, Tuple, Union
import torch.nn as nn
import json
from transformers import XLMRobertaTokenizer, XLMRobertaModel

MSE = nn.MSELoss(reduction='mean')


class XlmrSimulator(nn.Module):
    def __init__(self, train_example_nums, hyper_parameter=0, **kwargs):
        super(XlmrSimulator, self).__init__()
        self.train_xlmr_id_to_ids_dict = self._load_xlmr_id_to_ids_dict(kwargs['train_xlm_ids_file'])
        self.test_xlmr_id_to_ids_dict = self._load_xlmr_id_to_ids_dict(kwargs['test_xlm_ids_file'])
        
        # 加载xmlr的tokenizer和model
        self.xlmr_tokenizer = XLMRobertaTokenizer.from_pretrained(kwargs['xlmr_model_name_or_path'])
        self.xlmr_model = XLMRobertaModel.from_pretrained(kwargs['xlmr_model_name_or_path'])
        self.xlmr_model.eval()
        # 冻结xlmr模型参数
        for param in self.xlmr_model.parameters():
            param.requires_grad = False

        self.A = nn.Parameter(torch.rand(train_example_nums))
        self.B = nn.Parameter(torch.rand(train_example_nums))
        
        # self.emb = nn.Embedding(train_example_nums, 2)
        # self.fc1 = nn.Linear(2, 100)
        # self.relu = nn.ReLU()
        # self.fc2 = nn.Linear(100, 1)
        # self.fc3 = nn.Linear(100, 1)

        # self.embed =  nn.Embedding(train_example_nums, 1024)
        self.hyper_parameter = hyper_parameter
        nn.init.normal_(self.A, 0, 1)
        nn.init.normal_(self.B, 0, 1)

    def forward(
        self,
        orders: Optional[torch.Tensor] = None,  # tensor:(cur_bsz, finetune_gpt2_bsz)
        before_loss: Optional[torch.Tensor] = None,
        after_loss: Optional[torch.Tensor] = None,
        test_sample_ids: List = None,
        is_train=True
    ):
        # a = torch.gather(self.A.unsqueeze(0).repeat(before_loss.shape[0], 1), 1, orders)
        # b = torch.gather(self.B.unsqueeze(0).repeat(before_loss.shape[0], 1), 1, orders)

        fine_tuned_gpt2_bs = orders.shape[1]
        x_train_sample = self._orders_to_xlmr_emb(orders=orders) # (cur_bsz, finetune_gpt2_bsz, xlmr_hid_dim)
        x_test_sample = self._test_sample_ids_to_xmlr_emb(test_sample_ids=test_sample_ids)
        x_test_sample = x_test_sample.unsqueeze(dim=1).repeat(1, fine_tuned_gpt2_bs, 1)


        # 每一个训练样本对应两个参数

        # x = self.emb(orders)
        # x = self.fc1(x)
        # a = self.fc2(x)
        # b = self.fc3(x)

        
        
        predict_loss = torch.sum(a, dim=1) * before_loss + torch.sum(b, dim=1)


        mse_loss = MSE(after_loss, predict_loss) 
        L2_loss = 0.

        if is_train:
            L2_loss = self.hyper_parameter * (torch.norm(self.A, p=2, dim=0) + torch.norm(self.B, p=2, dim=0))   

        return {
            "mse_loss": mse_loss, 
            "L2_loss": L2_loss,
            "predict_loss": predict_loss,
            "tot_loss": mse_loss + L2_loss,
        }
    
    with torch.no_grad():
        def _orders_to_xlmr_emb(self, orders):
            hidden_states_list = []
            for order in orders:
                input_ids_list = []
                for sample_id in order:
                    input_ids_list.append(self.train_xlmr_id_to_ids_dict[str(sample_id.item())])
                
                # 批量转化为tensor，然后送入xlmr模型
                bs = 256
                hidden_list = []
                for start in range(0, len(input_ids_list), bs):
                    input_ids, attention_mask = self._pad_to_batch(input_ids_list[start:start+bs])
                    input_ids = input_ids.to(orders.device)
                    attention_mask = attention_mask.to(orders.device)
                    output = self.xlmr_model(input_ids, attention_mask)
                    hidden = output.last_hidden_state[:, 0]
                    hidden_list.append(hidden)
                hidden_states = torch.cat(hidden_list)
                hidden_states_list.append(hidden_states)
            return torch.stack(hidden_states_list, dim=0)
        
    with torch.no_grad():
        def _test_sample_ids_to_xmlr_emb(self, test_sample_ids):
            input_ids_list = []
            for test_sample_id in test_sample_ids:
                input_ids_list.append(self.test_xlmr_id_to_ids_dict[str(test_sample_id)])
                
            # 批量转化为tensor，然后送入xlmr模型
            bs = 256
            hidden_list = []
            for start in range(0, len(input_ids_list), bs):
                input_ids, attention_mask = self._pad_to_batch(input_ids_list[start:start+bs])
                input_ids = input_ids.to(self.xlmr_model.device)
                attention_mask = attention_mask.to(self.xlmr_model.device)
                output = self.xlmr_model(input_ids, attention_mask)
                hidden = output.last_hidden_state[:, 0]
                hidden_list.append(hidden)
            hidden_states = torch.cat(hidden_list)
            return hidden_states



    def _pad_to_batch(self, input_ids_list):
        '''
        right padding to input ids list, then return padded input ids tensors and attention mask tensors
        
        Args:
            input_ids_list: list of input ids
        
        Returns:
            input_ids_pt: padded input ids tensor
            attention_mask_pt: attention mask tensor
        '''
        max_len = max([len(input_ids) for input_ids in input_ids_list])
        for i, input_ids in enumerate(input_ids_list):
            if len(input_ids) < max_len:
                input_ids = input_ids + [self.xlmr_tokenizer.pad_token_id] * (max_len - len(input_ids))
                input_ids_list[i] = input_ids

        input_ids_pt = torch.tensor(input_ids_list)
        attention_mask_pt = (input_ids_pt != self.xlmr_tokenizer.pad_token_id).long()
        
        return input_ids_pt, attention_mask_pt
    
    def _load_xlmr_id_to_ids_dict(self, xlmr_ids_file):
        xlmr_id_to_ids_dict = {}
        with open(xlmr_ids_file, 'r') as f:
            for line in f.readlines():
                line = json.loads(line)
                id = str(line['id'])
                input_ids = line['input_ids']
                xlmr_id_to_ids_dict[id] = input_ids
        return xlmr_id_to_ids_dict
                
                

    
if __name__ == "__main__":
    simulator = Simulator(10)
    print(simulator)
    print('finish')