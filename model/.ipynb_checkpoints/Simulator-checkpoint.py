import torch
from torch import nn
from typing import List, Optional, Tuple, Union
import torch.nn as nn

MSE = nn.MSELoss(reduction='mean')


class Simulator(nn.Module):
    def __init__(self, train_example_nums, test_example_nums, hyper_parameter=0, **kargs):
        super(Simulator, self).__init__()
        
        self.training_example_nums = train_example_nums
        self.test_example_nums = test_example_nums

        self.emb = nn.Embedding(train_example_nums * test_example_nums, 2)
        '''
        param = 2 * m * n
        '''
        # self.fc1 = nn.Linear(2, 100)
        # self.relu = nn.ReLU()
        # self.fc2 = nn.Linear(100, 1)
        # self.fc3 = nn.Linear(100, 1)

        self.hyper_parameter = hyper_parameter

    def forward(
        self,
        orders: Optional[torch.Tensor] = None,  # tensor:(cur_bsz, finetune_gpt2_bsz)
        before_loss: Optional[torch.Tensor] = None,
        after_loss: Optional[torch.Tensor] = None,
        test_sample_ids: List = None,
        is_train=True,
        device=None,
    ):
        """
        orders: (bs, finetune_gpt2_bsz, )
        before_loss: (bs, )
        after_loss: (bs, )
        test_sample_ids: (bs, )
        """
        orders = self._adjust_orders(
            orders=orders,
            test_sample_ids=test_sample_ids,
            training_sample_nums=self.training_example_nums
        )
        hidden_states = self.emb(orders)
        a = hidden_states[:, :, 0]
        b = hidden_states[:, :, 1]
        # a = torch.gather(self.A.unsqueeze(0).repeat(before_loss.shape[0], 1), 1, orders)
        # b = torch.gather(self.B.unsqueeze(0).repeat(before_loss.shape[0], 1), 1, orders)
        

        # x = self.emb(orders)
        # x = self.fc1(x)
        # a = self.fc2(x)
        # b = self.fc3(x)
        
        predict_loss = torch.sum(a, dim=1) * before_loss + torch.sum(b, dim=1)


        mse_loss = MSE(after_loss, predict_loss) 
        L2_loss = 0.

        if is_train:
            L2_loss = self.hyper_parameter * (torch.norm(self.emb.weight[:, 0], p=2, dim=0) + torch.norm(self.emb.weight[:, 1], p=2, dim=0))   

        return {
            "mse_loss": mse_loss, 
            "L2_loss": L2_loss,
            "predict_loss": predict_loss,
            "tot_loss": mse_loss + L2_loss,
        }

    def _adjust_orders(self, orders: torch.Tensor, test_sample_ids: torch.Tensor, training_sample_nums: int):
        """
        converts the original orders to actual embedding index 

        args:
            
            orders: (bs, finetune_gpt2_bsz), id of batch training samples

            test_sample_ids: (bs, ), id of test sample

            training_sample_nums: int, number of all training samples
        
        returns:
            new_orders: (bs, finetune_gpt2_bsz), index of batch training samples
        """
        for i, (order, test_sample_id)in enumerate(zip(orders, test_sample_ids)):
            orders[i] = training_sample_nums * test_sample_id + order
        new_orders = orders
        return new_orders

        



if __name__ == "__main__":
    simulator = Simulator(10)
    print(simulator)
    print('finish')