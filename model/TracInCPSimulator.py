from collections import defaultdict
import threading
from captum.influence import TracInCPFast
from typing import Any, Callable, Dict, Optional, Tuple, Union, List, cast

from torch.nn import Module
from torch import Tensor, device
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

from captum.influence._utils.common import (
    _jacobian_loss_wrt_inputs,
    _tensor_batch_dot,
)
from captum._utils.common import _sort_key_list
from captum._utils.gradient import _gather_distributed_tensors

MSE = nn.MSELoss(reduction='mean')

# class TraceInCPSimulator
class HFTracInCPFast(TracInCPFast):
    def _influence_batch_tracincp_fast(
        self,
        test_batch: Tuple[Any, ...],
        train_batch: Tuple[Any, ...],
    ):
        """
        computes influence scores for a single training batch, when only considering
        gradients in the last fully-connected layer, using the computation trick
        described in the `TracInCPFast` class description.
        """

        def get_checkpoint_contribution(checkpoint):

            assert (
                checkpoint is not None
            ), "None returned from `checkpoints`, cannot load."

            learning_rate = self.checkpoints_load_func(self.model, checkpoint)

            input_jacobians, input_layer_inputs = self._basic_computation_tracincp_fast(
                test_batch[0:-1],
                test_batch[-1],
                self.test_loss_fn,
                self.test_reduction_type,
            ) # (test_bs*test_seq, vocab_size)

            src_jacobian, src_layer_input = self._basic_computation_tracincp_fast(
                train_batch[0:-1],
                train_batch[-1],
                self.loss_fn,
                self.reduction_type,
            ) # (train_bs*train_seq, vocab_size)
            return (
                _tensor_batch_dot(
                    input_jacobians, src_jacobian
                )  # shape is (test batch size, training batch size), containing x^T x'
                # for every example x in the training batch and example x' in the test
                # batch
                * _tensor_batch_dot(input_layer_inputs, src_layer_input)
                # shape is (test batch size, training batch size), containing
                # (\nabla_y f(y)^T \nabla_{y'} f(y')) for every label y in the training
                # batch and label y' in the test batch
                * learning_rate
            )

        batch_tracin_scores = get_checkpoint_contribution(self.checkpoints[0])

        for checkpoint in self.checkpoints[1:]:
            batch_tracin_scores += get_checkpoint_contribution(checkpoint)

        return batch_tracin_scores
    
    def _basic_computation_tracincp_fast(
        self,
        inputs: Tuple[Any, ...],
        targets: Tensor,
        loss_fn: Optional[Union[Module, Callable]] = None,
        reduction_type: Optional[str] = None,
    ):
        layer_inputs: Dict[device, Tuple[Tensor, ...]] = defaultdict()
        lock = threading.Lock()

        def hook_wrapper(original_module):
            def _capture_inputs(layer, input, output) -> None:
                r"""Save activations into layer_inputs in forward pass"""
                with lock:
                    is_eval_tuple = isinstance(input, tuple)
                    if is_eval_tuple:
                        layer_inputs_val = tuple(inp.detach() for inp in input)
                    else:
                        layer_inputs_val = input.detach()
                    layer_inputs[layer_inputs_val[0].device] = layer_inputs_val

            return _capture_inputs

        assert isinstance(self.final_fc_layer, Module)
        handle = self.final_fc_layer.register_forward_hook(
            hook_wrapper(self.final_fc_layer)
        )

        input_ids, attention_mask = tuple(inputs)
        # input_ids_list, attention_mask_list = tuple(inputs)
        # labels_list = targets
        device = self.model.device
        # input_ids = torch.stack(input_ids_list).to(device)
        # attention_mask = torch.stack(attention_mask_list).to(device)
        # targets = torch.stack(labels_list).to(device) # (bs, seq_len)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        targets = targets.to(device)

        # out = self.model(*inputs)
        out = self.model(
            input_ids,
            attention_mask,
        ).logits # (bs, seq_len, vocab_size)

        # 计算target的loos
        # shift logits
        shifted_out = out[..., :-1, :].contiguous()
        shifted_targets = targets[..., 1:].contiguous()

        vocab_size = self.model.config.vocab_size
        shifted_out = shifted_out.view(-1, vocab_size)
        shifted_targets = shifted_targets.view(-1)
        # # Enable model parallelism
        # tot_shift_labels = tot_shift_labels.to(shift_logits.device)

        assert loss_fn is not None, "loss function is required"
        assert reduction_type in [
            "sum",
            "mean",
        ], 'reduction_type must be either "mean" or "sum"'
        input_jacobians = _jacobian_loss_wrt_inputs(
            loss_fn,
            shifted_out,
            shifted_targets,
            self.vectorize,
            reduction_type,
        )

        handle.remove()

        device_ids = cast(
            Union[None, List[int]],
            self.model.device_ids
            if hasattr(self.model, "device_ids")
            else None,
        )
        key_list = _sort_key_list(list(layer_inputs.keys()), device_ids)

        _layer_inputs = _gather_distributed_tensors(layer_inputs, key_list=key_list)[0]

        assert len(input_jacobians.shape) == 2

        bs = input_ids.shape[0]
        # input_jacobians = input_jacobians.reshape(bs, -1)
        # _layer_inputs = _layer_inputs.reshape(bs, -1)
        input_jacobians = input_jacobians.reshape(bs, -1, vocab_size).sum(dim=1)
        _layer_inputs = _layer_inputs.sum(dim=1)
        return input_jacobians, _layer_inputs


class EveryStepDataset(Dataset):
    def __init__(self, data, tokenizer, **kwargs):
        '''
        
        Args:
            data (`dict`): 
        '''
        samples_texts = data['samples_texts']
        samples_texts_without_tgt = data['samples_texts_without_tgt']
        
        input_tokenized = tokenizer(
            samples_texts,
            padding=True,
            truncation=True,
            max_length=kwargs['max_length'],
            return_tensors='pt',
        )
        
        # 在这里把样本全部处理好
        input_ids = input_tokenized['input_ids']
        attention_mask = input_tokenized['attention_mask']
        labels = input_ids.clone()

        ### 结合pad token和input长度，来计算label中的非target部分
        pad_len = (attention_mask == 0).sum(-1)
        input_without_tgt_tokenized = tokenizer(
            samples_texts_without_tgt,
            padding=True,
            truncation=True,
            max_length=kwargs['max_length'],
            return_tensors='pt',
        )
        only_input_len = input_without_tgt_tokenized['attention_mask'].sum(-1)
        target_start_idx = only_input_len + pad_len
        for i, idx in enumerate(target_start_idx):
            labels[i, :idx] = -100
        
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_mask[idx], self.labels[idx]

    
    def __len__(self):
        return len(self.input_ids)

# 需要自定义checkpoint load函数，
def checkpoints_load_func(model, path):
    # 目前直接加载pytorch_model.bin
    # model=torch.load(os.path.join(path, 'pytorch_model.bin'))
    # 从training_args.bin中读取学习率
    trainer_state = json.load(open(os.path.join(path, 'trainer_state.json'), 'r', encoding='utf-8'))
    learning_rate = trainer_state['log_history'][-1]['learning_rate']
    return learning_rate
    
def TracInCPSimulator(
    orders: Optional[torch.Tensor] = None,  # tensor:(cur_bsz, finetune_gpt2_bsz)
    before_loss: Optional[torch.Tensor] = None,
    after_loss: Optional[torch.Tensor] = None,
    test_sample_ids: List = None,
    is_train=True,
    device='cuda:0',
    **kwargs
):
    # 从kwargs中读取`checkpoints_path`
    checkpoints_path: List = kwargs['path']

    # 处理TracInCP数据
    train_data = {
        'samples_texts': kwargs['samples_texts'],
        'samples_texts_without_tgt': kwargs['samples_contexts']
    }

    test_data = {
        'samples_texts': kwargs['test_sample_text'],
        'samples_texts_without_tgt': kwargs['test_sample_context']
    }

    # 设置max_length
    max_length: int = getattr(kwargs, 'max_length', 2048)

    # 设置tokenizer
    model = AutoModelForCausalLM.from_pretrained(checkpoints_path[0]).to(device)
    # debug without model #########################################
    # model = kwargs['model']
    ################################################################
    tokenizer = AutoTokenizer.from_pretrained(checkpoints_path[0])
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'
    
    step_train_dataset = EveryStepDataset(
        data=train_data,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    step_test_dataset = EveryStepDataset(
        data=test_data,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    test_dataloader = DataLoader(
        step_test_dataset,
        batch_size=1,
    )

    # 设置训练集batch size
    batch_size = getattr(kwargs, 'train_batch_size', 2)

    # 设置final_fc_layer
    final_fc_layer: str = getattr(kwargs, 'final_fc_layer', 'embed_out')

    # 设置损失函数
    loss_fn = nn.CrossEntropyLoss(reduction='mean')


    hf_tracin_cp_fast = HFTracInCPFast(
        model=model,
        final_fc_layer=final_fc_layer,
        checkpoints_load_func=checkpoints_load_func,
        train_dataset=step_train_dataset,
        checkpoints=checkpoints_path,
        loss_fn=loss_fn,
        batch_size=batch_size,
    )
    
    # for test_batch in test_dataloader:
    #     output = hf_tracin_cp_fast.influence(
    #         inputs=,
    #     )
    test_batch = list(test_dataloader)[0]
    influence = hf_tracin_cp_fast.influence(
        inputs=test_batch,
    )
    tot_influence = influence.sum()

    
    # L_t(z) - L_{t+1}(z) = tot_influence
    predict_loss = before_loss - tot_influence
    mse_loss = MSE(predict_loss, after_loss)

    return {
            "mse_loss": mse_loss, 
            "L2_loss": None,
            "predict_loss": predict_loss,
            "tot_loss": None,
        }

if __name__ == "__main__":

    from transformers import AutoModelForCausalLM, AutoTokenizer
    MODEL_NAME_OR_PATH = '/root/paddlejob/workspace/liuqingyi01/code/alpaca-lora-main/models--EleutherAI--pythia-160m-deduped/'
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME_OR_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'

    demo_train_data = {
        'samples_texts': [
            'the quick brown fox jumps over the lazy dog',
            'the quick brown fox jumps over the lazy dog',
        ],
        'samples_texts_without_tgt': [
            'the quick brown fox',
            'the quick brown fox',
        ]
    }

    demo_test_data = {
        'samples_texts': [
            'the lazy dog jumps over the quick brown fox',
            'the lazy dog jumps over the quick brown fox',
        ],
        'samples_texts_without_tgt': [
            'the lazy dog jumps over the quick brown fox',
            'the brown fox jumps over the lazy dog',
        ]
    }


    step_train_dataset = EveryStepDataset(
        data=demo_train_data,
        tokenizer=tokenizer,
        max_length=2048,
    )

    step_test_dataset = EveryStepDataset(
        data=demo_test_data,
        tokenizer=tokenizer,
        max_length=2048,
    )

    test_input_ids, test_attention_mask, test_labels = step_test_dataset[0]    
    test_dataloader = DataLoader(
        step_test_dataset,
        batch_size=4,
    )

    

    # 需要自定义checkpoint load函数，
    def checkpoints_load_func(model, path):
        model=torch.load(path)

        return 1.


    hf_tracin_cp_fast = HFTracInCPFast(
        model=model,
        final_fc_layer='embed_out',
        checkpoints_load_func=checkpoints_load_func,
        train_dataset=step_train_dataset,
        checkpoints=['/root/paddlejob/workspace/liuqingyi01/code/alpaca-lora-main/models--EleutherAI--pythia-160m-deduped/pytorch_model.bin'],
        loss_fn=nn.CrossEntropyLoss(reduction='mean'), ### ???
        batch_size=1,
    )
    print("finished")
    
    for test_batch in test_dataloader:
        hf_tracin_cp_fast.influence(
            inputs=test_batch,
        )