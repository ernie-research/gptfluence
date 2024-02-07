import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import os

with torch.no_grad():
    def eval_simulator(eval_dataset, model, device, input_kwargs_keys):
        model.eval()
        all_steps_mse = 0.
        all_steps_mae = 0.
        all_steps_mse_list = []
        all_steps_mae_list = []
        pred_loss_dict = defaultdict(list)
        for eval_data in tqdm(eval_dataset):
            step_mse_loss_list = []
            steps_mae_list = []
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

                # # 如果是n-th order markov
                if 'prev_n_losses' in input_kwargs_keys and 'prev_n_steps' in input_kwargs_keys:
                    # 取出预测的前N步loss
                    N = model.order_n
                    prev_pred_n_losses = pred_loss_dict[test_sample_id][-1]['pred_loss'][-N:]
                    prev_pred_n_losses = [0] * (N - len(prev_pred_n_losses)) + prev_pred_n_losses
                    input_kwargs['prev_n_losses'] = torch.tensor([prev_pred_n_losses]).to(device)

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
                steps_mae_list.append(abs(predict_loss - data['cur_loss']))
            # all_steps_mse += np.array(step_mse_loss_list).mean()
            # all_steps_mae += np.array(steps_mae_list).mean()
            all_steps_mse_list.append(np.array(step_mse_loss_list).mean())
            all_steps_mae_list.append(np.array(steps_mae_list).mean())
        
        # 计算不同runs的指标均值和方差
        # all steps mse
        all_steps_mse_np = np.array(all_steps_mse_list)
        all_steps_mse_mean = all_steps_mse_np.mean()
        all_steps_mse_std = all_steps_mse_np.std()
        # all steps mae
        all_steps_mae_np = np.array(all_steps_mae_list)
        all_steps_mae_mean = all_steps_mae_np.mean()
        all_steps_mae_std = all_steps_mae_np.std()
        
        return {
            'all_steps_mse_mean': all_steps_mse_mean,
            'all_steps_mse_std': all_steps_mse_std,
            'all_steps_mae_mean': all_steps_mae_mean,
            'all_steps_mae_std': all_steps_mae_std,
            'pred_loss_dict': pred_loss_dict
        }
    

from transformers import AutoTokenizer, AutoModelForCausalLM
def eval_tracincp_simulator(eval_dataset, model, device, step_ckpt_dir, input_kwargs_keys, ckpt_steps=None):
    all_steps_mse_list = []
    all_steps_mae_list = []
    pred_loss_dict = defaultdict(list)

    eval_dataset_wrt_prev_step = defaultdict(list)
    for eval_data in eval_dataset:
        for data in eval_data:
            prev_step = data['prev_step']
            eval_dataset_wrt_prev_step[prev_step].append(data)

    gpt_model = None
    tokenizer = None
    model_name_list = []
    
    # 处理ckpt_steps
    ### ckpt_steps 要么是`None`要么是`str`
    if ckpt_steps is not None:
        assert isinstance(ckpt_steps, str), "请通过字符串设置ckpt_steps"
        ckpt_steps = list(map(int, ckpt_steps.split(',')))
        # import pdb; pdb.set_trace()
    for prev_step, eval_data in tqdm(eval_dataset_wrt_prev_step.items()):
        if ckpt_steps is not None:
            for ckpt_step in ckpt_steps:
                if prev_step <= ckpt_step:
                    model_name_or_path = os.path.join(step_ckpt_dir, f'checkpoint-{ckpt_step}')
                    print(f'load checkpoint-{ckpt_step}')
                    break
        else:
            model_name_or_path = os.path.join(step_ckpt_dir, f'checkpoint-{prev_step}')
            print(f'load checkpoint-{prev_step}')

        
        if model_name_or_path not in model_name_list:
            model_name_list.append(model_name_or_path)
            gpt_model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device)
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            tokenizer.pad_token_id = tokenizer.eos_token_id
            tokenizer.padding_side = 'left'
        
        run_id = 0
        for i, data in enumerate(eval_data):
            test_sample_id = data["test_sample_id"]
            
            if i != 0:
                prev_test_sample_id = eval_data[i-1]["test_sample_id"]
                if test_sample_id < prev_test_sample_id:
                    run_id += 1

            if prev_step == 1:
                start_step = prev_step
                start_loss = data["prev_loss"]                 
                pred_loss_dict[test_sample_id].append(
                    {
                        'gt_loss': [start_loss],
                        'pred_loss': [start_loss],
                        'step': [start_step]
                    }
                )

            before_loss = None
            if prev_step == 1:
                before_loss = torch.tensor([data["prev_loss"]]).to(device)
            else: 
                before_loss = torch.tensor([predict_loss]).to(device)
            
            input_kwargs = {key: data[key] for key in input_kwargs_keys}

            # 通过kwargs设置TracInCPSimulator的参数 ######################
            input_kwargs['model'] = gpt_model
            input_kwargs['tokenizer'] = tokenizer
            input_kwargs['path'] = [model_name_or_path]
            ############################################################

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
            pred_loss_dict[test_sample_id][run_id]['gt_loss'].append(data['cur_loss'])
            pred_loss_dict[test_sample_id][run_id]['pred_loss'].append(predict_loss)
            pred_loss_dict[test_sample_id][run_id]['step'].append(data['cur_step'])


    return {
        'all_steps_mse_mean': None,
        'all_steps_mse_std': None,
        'all_steps_mae_mean': None,
        'all_steps_mae_std': None,
        'pred_loss_dict': pred_loss_dict
    }



        




    # for eval_data in tqdm(eval_dataset):
    #     step_mse_loss_list = []
    #     steps_mae_list = []
    #     predict_loss = None
    #     test_sample_id = eval_data[0]["test_sample_id"]
    #     start_step = eval_data[0]["prev_step"]
    #     start_loss = eval_data[0]["prev_loss"]
        
    #     # pred_loss_dict[test_sample_id] = {
    #     #     'pred_loss': [start_loss],
    #     #     'step': [start_step]
    #     # }
    #     pred_loss_dict[test_sample_id].append(
    #         {
    #             'gt_loss': [start_loss],
    #             'pred_loss': [start_loss],
    #             'step': [start_step]
    #         }
    #     )
    #     for step, data in enumerate(eval_data):
    #         before_loss = None
    #         if step == 0:
    #             before_loss = torch.tensor([data["prev_loss"]]).to(device)
    #         else: 
    #             before_loss = torch.tensor([predict_loss]).to(device)
            
    #         input_kwargs = {key: data[key] for key in input_kwargs_keys}

    #         # 通过kwargs设置TracInCPSimulator的参数 ######################
    #         ### 设置checkpoint path
    #         cur_step = data['cur_step']
    #         # input_kwargs['path'] = [os.path.join(step_ckpt_dir, f'checkpoint-{cur_step}')]
    #         input_kwargs['path'] = [os.path.join(step_ckpt_dir, f'checkpoint-2')]
    #         ############################################################

    #         output = model(
    #             orders=torch.tensor([data["samples_id"]]).to(device),
    #             before_loss=before_loss,
    #             after_loss=torch.tensor([data["cur_loss"]]).to(device),
    #             test_sample_ids=torch.tensor([data["test_sample_id"]]).to(device),
    #             is_train=False,
    #             device=device,
    #             **input_kwargs
    #         )            
            
    #         if torch.isnan(output['mse_loss']).all():
    #             raise ValueError(f"mse_loss is nan, before loss is {predict_loss}")

    #         if torch.isnan(output['predict_loss']).all():
    #             raise ValueError(f"predict_loss is nan, before loss is {predict_loss}")

    #         predict_loss = output['predict_loss'].item()
    #         pred_loss_dict[test_sample_id][-1]['gt_loss'].append(data['cur_loss'])
    #         pred_loss_dict[test_sample_id][-1]['pred_loss'].append(predict_loss)
    #         pred_loss_dict[test_sample_id][-1]['step'].append(data['cur_step'])

        #     step_mse_loss = output['mse_loss'].item()
        #     step_mse_loss_list.append(step_mse_loss)
        #     steps_mae_list.append(abs(predict_loss - data['cur_loss']))
        # # all_steps_mse += np.array(step_mse_loss_list).mean()
        # # all_steps_mae += np.array(steps_mae_list).mean()
        # all_steps_mse_list.append(np.array(step_mse_loss_list).mean())
        # all_steps_mae_list.append(np.array(steps_mae_list).mean())
    
    # # 计算不同runs的指标均值和方差
    # # all steps mse
    # all_steps_mse_np = np.array(all_steps_mse_list)
    # all_steps_mse_mean = all_steps_mse_np.mean()
    # all_steps_mse_std = all_steps_mse_np.std()
    # # all steps mae
    # all_steps_mae_np = np.array(all_steps_mae_list)
    # all_steps_mae_mean = all_steps_mae_np.mean()
    # all_steps_mae_std = all_steps_mae_np.std()
    
    # return {
    #     'all_steps_mse_mean': all_steps_mse_mean,
    #     'all_steps_mse_std': all_steps_mse_std,
    #     'all_steps_mae_mean': all_steps_mae_mean,
    #     'all_steps_mae_std': all_steps_mae_std,
    #     'pred_loss_dict': pred_loss_dict
    # }