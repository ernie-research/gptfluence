# [EMNLP'24 (Oral) | On Training Data Influence of GPT Models](https://aclanthology.org/2024.emnlp-main.183/)

   <a href="https://huggingface.co/datasets/baidu/GPTDynamics" target="_blank">
      <img alt="Datasets" src="https://img.shields.io/badge/📚-Dataset-orange" />
   </a> 
   <a href="https://aclanthology.org/2024.emnlp-main.183/" target="_blank"><img alt="Paper" src="https://img.shields.io/badge/📜-Paper-purple" /></a>
  <a href="https://2024.emnlp.org/" target="_blank"> <img alt="EMNLP 2024" src="https://img.shields.io/badge/Proceedings-EMNLP2024-red" /> </a>


The official repository which contains the code and model checkpoints for our paper [On Training Data Influence of GPT Models (EMNLP 2024)](https://aclanthology.org/2024.emnlp-main.183.pdf).


## 🔥 News
* **21 September, 2024:** 🎉 Our work has been accepted to [EMNLP 2024 (Oral)](https://2024.emnlp.org/)! 🎉
* **1 May, 2024:** 🎉 We release the official dataset of [baidu/GPTDynamics](https://huggingface.co/datasets/baidu/GPTDynamics)!🔥

![image](https://github.com/ernie-research/gptfluence/blob/main/resources/overview.png)
Amidst the rapid advancements in generative language models, the investigation of how training data shapes the performance of GPT models is still emerging. This paper presents GPTfluence, a novel approach that leverages a featurized simulation to assess the impact of training examples on the training dynamics of GPT models. Our approach not only traces the influence of individual training instances on performance trajectories, such as loss and other key metrics, on targeted test points but also enables a comprehensive comparison with existing methods across various training scenarios in GPT models, ranging from 14 million to 2.8 billion parameters, across a range of downstream tasks. Contrary to earlier methods that struggle with generalization to new data, GPTfluence introduces a parameterized simulation of training dynamics, demonstrating robust generalization capabilities to unseen training data. This adaptability is evident across both fine-tuning and instruction-tuning scenarios, spanning tasks in natural language understanding and generation. 
## 📕 Requirements
To set up the environment and install dependencies, run:
```
bash run_requirements.sh
```
## 📚 GPTDynamics Data
We release the GPTDynamics for training and testing the featurized simulator in [baidu/GPTDynamics](https://huggingface.co/datasets/baidu/GPTDynamics). To preprocess the data, you should follow the instructions below:
### Download
First, you need to download the data in the `GPTDynamics/gptdynamics` directory from [baidu/GPTDynamics](https://huggingface.co/datasets/baidu/GPTDynamics) locally. The data contains two parts, the first part is `GPTDynamics/gptdynamics/sft_tasks` and `GPTDynamics/gptdynamics/sft_tasks` correspond to the data samples for training, evaluation, and testing used in instruction fine-tuning and fine-tuning scenarios, respectively. You need to fill the local data paths into `DATASET_ADDITIONAL_ARGS` of `train.py` according to the task names to read the samples from the paths you provided; the second part is `GPTDynamics/gptdynamics/GPTDynamics.tar`, which contains the training runs, you need to extract:
```
tar -xvf GPTDynamics.tar
```
and placed in the repository directory.
### Instruction-tuning Scenario
Preprocess loss trajectory
```
python utils/construct_runs-data-flan-multi-thread.py
```
Preprocess metric（BLEU/ROUGE score）trajectory  
```
python utils/construct_runs-data-flan-metric-multi-thread.py
```
### Fine-tuning Scenario
Preprocess loss trajectory
```
python utils/construct_runs-data.py
```
Preprocess metric（BLEU/ROUGE score）trajectory
```
python utils/construct_runs-data-metric.py
```
    
### Draw loss/BLEU/ROUGE-L trajectory (Optional)
```
python utils/draw_gt_curves.py
```
## 🚀 Featurized Simulator
### Set up training, validation, and testing data
Before training and inference, you should specify training, validation, and test data in the corresponding scripts. 
You should specify the simulator's training and validation data in `train.py` via the command line argument `--data_paths_dict`; and the simulator's test data in `test.py` via the command line argument `--data_paths_dict`. If you follow the steps above to download our open-source GPTDynamics data, you don't have to set it up additionally `-- data_paths_dict` and use the default configuration of the script species.

 ### Training
 ```
 bash run_enc_sim.sh
```
  
 ### Evaluation
 Predict loss
 ```
 bash auto_test.sh
 ```
 Predict metric(BLEU/ROUGE score)
 ```
 bash auto_test_metric.sh
 ```

## Citation
```
@inproceedings{chai-etal-2024-training,
    title = "On Training Data Influence of {GPT} Models",
    author = "Chai, Yekun  and
      Liu, Qingyi  and
      Wang, Shuohuan  and
      Sun, Yu  and
      Peng, Qiwei  and
      Wu, Hua",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.183",
    pages = "3126--3150",
    abstract = "Amidst the rapid advancements in generative language models, the investigation of how training data shapes the performance of GPT models is still emerging. This paper presents GPTfluence, a novel approach that leverages a featurized simulation to assess the impact of training examples on the training dynamics of GPT models. Our approach not only traces the influence of individual training instances on performance trajectories, such as loss and other key metrics, on targeted test points but also enables a comprehensive comparison with existing methods across various training scenarios in GPT models, ranging from 14 million to 2.8 billion parameters, across a range of downstream tasks. Contrary to earlier methods that struggle with generalization to new data, GPTfluence introduces a parameterized simulation of training dynamics, demonstrating robust generalization capabilities to unseen training data. This adaptability is evident across both fine-tuning and instruction-tuning scenarios, spanning tasks in natural language understanding and generation. We make our code and data publicly available at https://github.com/ernie-research/gptfluence.",
}
```
