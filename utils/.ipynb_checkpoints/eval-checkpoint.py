import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
with torch.no_grad():
    def eval_simulator(eval_dataset, model, device, input_kwargs_keys):
        model.eval()
        all_steps_mse = 0.
        all_steps_mse_list = []
        pred_loss_dict = defaultdict(list)
        for eval_data in tqdm(eval_dataset):
            step_mse_loss_list = []
            predict_loss = None
            test_sample_id = eval_data[0]["test_sample_id"]
            start_step = eval_data[0]["prev_step"]
            start_loss = eval_data[0]["prev_loss"]
            
            # pred_loss_dict[test_sample_id] = {
            #     'pred_loss': [start_loss],
            #     'step': [start_step]
            # }
            pred_loss_dict[test_sample_id].append(
                {
                    'gt_loss': [start_loss],
                    'pred_loss': [start_loss],
                    'step': [start_step]
                }
            )
            for step, data in enumerate(eval_data):
                before_loss = None
                if step == 0:
                    before_loss = torch.tensor([data["prev_loss"]]).to(device)
                else: 
                    before_loss = torch.tensor([predict_loss]).to(device)
                
                input_kwargs = {key: data[key] for key in input_kwargs_keys}
                output = model(
                    orders=torch.tensor([data["samples_id"]]).to(device),
                    before_loss=before_loss,
                    after_loss=torch.tensor([data["cur_loss"]]).to(device),
                    test_sample_ids=torch.tensor([data["test_sample_id"]]).to(device),
                    is_train=False,
                    device=device,
                    **input_kwargs
                )
                
                if torch.isnan(output['mse_loss']).all():
                    raise ValueError(f"mse_loss is nan, before loss is {predict_loss}")

                if torch.isnan(output['predict_loss']).all():
                    raise ValueError(f"predict_loss is nan, before loss is {predict_loss}")

                predict_loss = output['predict_loss'].item()
                pred_loss_dict[test_sample_id][-1]['gt_loss'].append(data['cur_loss'])
                pred_loss_dict[test_sample_id][-1]['pred_loss'].append(predict_loss)
                pred_loss_dict[test_sample_id][-1]['step'].append(data['cur_step'])

                step_mse_loss = output['mse_loss'].item()
                step_mse_loss_list.append(step_mse_loss)
            all_steps_mse += np.array(step_mse_loss_list).mean()
        return all_steps_mse / len(eval_dataset), pred_loss_dict