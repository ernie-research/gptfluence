import torch
from torch import nn




class MetricSimulator(nn.Module):
    def __init__(self,num_train_samples, data_to_index):
        super(MetricSimulator, self).__init__()
        self.params = nn.Parameter(torch.zeros(num_train_samples, 2))
        self.data_to_index = data_to_index
    def forward(self, tot_step, M_prev):
        # predicted_losses = []
        # for step in tot_step:
        #     alpha_ct = 0
        #     beta_ct = 0
        #     for data in step:
        #         data_idx = self.data_to_index[data]
        #         alpha_ct += self.params[data_idx][0]
        #         beta_ct += self.params[data_idx][1]
        #     pred_loss = alpha_ct * M_prev + beta_ct
        #     M_prev = pred_loss
        #     predicted_losses.append(pred_loss)
        # return torch.tensor(predicted_losses, requires_grad=True)
        
      
        predicted_losses = []
        for step in tot_step:
            indices = torch.tensor([self.data_to_index[data] for data in step], dtype=torch.long)
            step_params = self.params[indices]
            
            alpha_ct = step_params[:, 0].sum()
            beta_ct = step_params[:, 1].sum()
            
            pred_loss = alpha_ct * M_prev + beta_ct
            M_prev = pred_loss
            predicted_losses.append(pred_loss.unsqueeze(0))
        predicted_losses = torch.cat(predicted_losses)
        return predicted_losses
        # # Concatenate all loss predictions into a single tensor
        # predicted_losses = torch.cat(predicted_losses)
        # return predicted_losses
    
            # for index, value  in enumerate(train_indices):
        #     if index == 0:
        #         M_prev = labels[0]
        #     else:
        #         M_prev = labels[index - 1]

        #     alpha_ct = self.params[value][0]
        #     beta_ct = self.params[value][1]
        #     predicted_loss = alpha_ct * M_prev + beta_ct

        #     predicted_losses.append(predicted_loss)

    def initialize_weights(self):
        # pass
        
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()

   

class MetricSimulator2(nn.Module):
    def __init__(self,num_train_samples):
        super(MetricSimulator, self).__init__()
        self.params = nn.Parameter(torch.randn(num_train_samples, 3))

    def forward(self, train_indices, M_prev, M_prev_prev, labels):
    
        
        predicted_losses = []
        for index, value  in enumerate(train_indices):
            if index == 0:
                M_prev = labels[0]
                M_prev_prev = labels[0]
            else:
                M_prev_prev = M_prev
                M_prev = labels[index - 1]

            alpha_ct = self.params[value][0]
            beta_ct = self.params[value][1]
            gamma_ct = self.params[value][2]
            predicted_loss = alpha_ct * M_prev + beta_ct * M_prev_prev + gamma_ct

            
            
            predicted_losses.append(predicted_loss)
        
        return torch.tensor(predicted_losses)

    def initialize_weights(self):
        pass
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         torch.nn.init.xavier_normal_(m.weight.data)
        #         if m.bias is not None:
        #             m.bias.data.zero_()
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        #     elif isinstance(m, nn.Linear):
        #         torch.nn.init.normal_(m.weight.data, 0, 0.01)
        #         m.bias.data.zero_()

        
class MetricSimulator1(nn.Module):
    def __init__(self, num_samples):
        super(MetricSimulator1, self).__init__()
        
        self.A = nn.Parameter(torch.randn(num_samples))
        self.B = nn.Parameter(torch.randn(num_samples))
        self.C = nn.Parameter(torch.randn(num_samples))

    def forward(self, c_t_indices, M_prev):
        alpha = self.A[c_t_indices].sum()
        beta = self.B[c_t_indices].sum()
        gamma = self.C[c_t_indices].sum()
        M_pred = alpha * M_prev + gamma * M_prev + beta
        return M_pred

    # 定义权值初始化
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()