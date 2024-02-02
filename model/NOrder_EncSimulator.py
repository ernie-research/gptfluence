import torch
from torch import nn
from typing import List, Optional, Tuple, Union
import torch.nn as nn
import json
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

MSE = nn.MSELoss(reduction='mean')


class NOrder_EncSimulator(nn.Module):
    def __init__(self, train_example_nums, test_example_nums, hyper_parameter=0, **kwargs):
        super(NOrder_EncSimulator, self).__init__()

        # 建模的阶数
        self.order_n = kwargs['order_n']

        # 编码器和tokenzier
        self.encoder = AutoModel.from_pretrained(kwargs['enc_model_name_or_path'])
        self.tokenizer = AutoTokenizer.from_pretrained(kwargs['enc_model_name_or_path'])
        
        self.encoder.eval()
        # 冻结编码器参数
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.hidden_dim = self.encoder.config.hidden_size
        # self.fc = nn.Linear(self.hidden_dim, self.hidden_dim)
        # self.relu = nn.ReLU()
        # self.fca = nn.Linear(self.hidden_dim, 1)
        # self.fcb = nn.Linear(self.hidden_dim, 1)

        # MLP
        # self.mlp_a = nn.Sequential(
        #     nn.Linear(self.hidden_dim, self.hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.hidden_dim, self.hidden_dim),
        # )

        # self.mlp_b = nn.Sequential(
        #     nn.Linear(self.hidden_dim, self.hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.hidden_dim, self.hidden_dim),
        # )

        

        self.concate = kwargs['concate']
        if self.concate:
            # mlp-1 ##################################
            # self.fc = nn.Linear(self.hidden_dim * 2, 100)
            # self.relu = nn.ReLU()
            #########################################
            self.mlp = nn.Sequential(
                nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(self.hidden_dim * 2, 100),
                nn.ReLU(),
            )
            # self.fca = nn.Linear(100, 1)
            # self.fcb = nn.Linear(100, 1)
            self.fc_layers = nn.ModuleList([ nn.Linear(100, 1) for _ in range(self.order_n + 1)])
        else:
            self.mlp_layers = nn.ModuleList([ nn.Sequential(
                        nn.Linear(self.hidden_dim, self.hidden_dim),
                        nn.ReLU(),
                        nn.Linear(self.hidden_dim, self.hidden_dim),
            ) for _ in range(self.order_n + 1)]) # `1` for addtion fluence

        # 1.：mlp embed纬度，a,b参数共享
        # 2.：不同交互方式
        # 3.：Bert初始化

    

        self.hyper_parameter = hyper_parameter
        
        # 初始化模型时初始化embedding
        self.frozen = kwargs['frozen']
        self.embed = nn.Embedding(train_example_nums,  self.hidden_dim)
        self.test_embed = nn.Embedding(test_example_nums, self.hidden_dim)
        
    
    def _get_initial_embeds(self, dataset, device):
        sample_id_dict = dataset.id_to_text_train
        test_sample_id_dict = dataset.id_to_text_eval
        # sample_id_dict = {}
        # test_sample_id_dict = {}
        print(f'initial embeds ...')
        # for d in tqdm(dataset):
        #     # 获取评估样本的文本和id
        #     test_sample_id, test_sample_text = d['test_sample_id'], d['test_sample_text']
        #     if test_sample_id_dict.get(test_sample_id) is None:
        #         test_sample_id_dict[test_sample_id] = test_sample_text[0]
        #     # 获取训练集样本的文本和id
        #     for sample_id, sample_text in zip(d['samples_id'], d['samples_texts']):
        #         if sample_id_dict.get(sample_id) is None:
        #             sample_id_dict[sample_id] = sample_text

        sample_embed_tensors = self._sample_id_dict_to_embed(sample_id_dict, device)
        self.embed = nn.Embedding.from_pretrained(sample_embed_tensors, freeze=self.frozen)

        test_sample_embed_tensors = self._sample_id_dict_to_embed(test_sample_id_dict, device)
        self.test_embed = nn.Embedding.from_pretrained(test_sample_embed_tensors, freeze=self.frozen)
                    
        # 冻结embedding参数 
        print(f'freeze embeds? {self.frozen}')
            
        print(f'initial embeds done')

    def _sample_id_dict_to_embed(self, sample_id_dict, device):
        '''根据sample_id_dict构建样本embedding'''
        sorted_sample_id_tuple = sorted(sample_id_dict.items())
        embed_tensor_list = []
        for sample_id, sample_text in tqdm(sorted_sample_id_tuple):
            embeds = self._get_sample_embeds(
                texts=[sample_text],
                device=device,
            )
            embed_tensor_list.append(embeds[0].detach())
        
        return torch.stack(embed_tensor_list, dim=0)
        

    def forward(
        self,
        orders: Optional[torch.Tensor] = None,  # tensor:(cur_bsz, finetune_gpt2_bsz)
        before_loss: Optional[torch.Tensor] = None,
        after_loss: Optional[torch.Tensor] = None,
        test_sample_ids: List = None,
        is_train=True,
        device='cuda:0',
        **kwargs
    ):  
        # 获取训练样本的embedding
        # samples_texts = kwargs['samples_texts']
        # samples_embeds = [ self._get_sample_embeds(texts=texts, device=device) for texts in samples_texts]
        # x = torch.stack(samples_embeds, dim=0)

        bs, ft_bs = orders.size()
        x_src = self.embed(orders) # (bs, ft_bs, hid_dim)
        x_tgt = self.test_embed(test_sample_ids) # (bs, hid_dim)

        score_list = []
        if not self.concate:
            # 计算
            for mlp_layer in self.mlp_layers:
                h_src = mlp_layer(x_src)
                h_tgt = mlp_layer(x_tgt)
                score_list.append(torch.mul(h_src, h_tgt.unsqueeze(1)).sum(dim=-1))


            
        else:
            x = torch.cat([x_src, x_tgt.unsqueeze(1).repeat(1, ft_bs, 1)], dim=-1) # (bs, ft_bs+1, hid_dim)])
            # mlp-1 ##################################
            # x = self.fc(x)
            # x = self.relu(x)
            ##########################################
            x = self.mlp(x)
            score_list = [ layer(x).squeeze(-1) for layer in self.fc_layers ]
            # a = self.fca(x).squeeze(-1)
            # b = self.fcb(x).squeeze(-1)

        a = torch.stack(score_list[:-1], dim=0).transpose(0, 1)
        b = score_list[-1]
        
        before_loss = kwargs['prev_n_losses']
        if isinstance(before_loss, list):
            before_loss = torch.tensor(before_loss).to(device)
        predict_loss = torch.sum(torch.sum(a, dim=-1) * before_loss, dim=-1) + torch.sum(b, dim=1)


        mse_loss = MSE(after_loss, predict_loss)
        L2_loss = 0.

        if is_train and self.hyper_parameter != 0.0:
            # L2_loss = self.hyper_parameter * (torch.norm(self.A, p=2, dim=0) + torch.norm(self.B, p=2, dim=0))   
            raise NotImplementedError()

        return {
            "mse_loss": mse_loss, 
            "L2_loss": L2_loss,
            "predict_loss": predict_loss,
            "tot_loss": mse_loss + L2_loss,
        }
    
    with torch.no_grad():
        def _get_sample_embeds(self, texts, device):
            '''
            Args:
                `texts`: list of str
            Returns:
                `embeds`: tensor of shape (len(texts), hidden_dim)
            '''
            tokenized = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
            print('\r 编码器最大长度为：512', end="")
            input_ids = tokenized['input_ids'].to(device)
            attention_mask = tokenized['attention_mask'].to(device)
            token_type_ids = tokenized['token_type_ids'].to(device)
            output = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids = token_type_ids
            )
            embeds = output.last_hidden_state[:, 0]
            return embeds
                

    
if __name__ == "__main__":
    simulator = Simulator(10)
    print(simulator)
    print('finish')