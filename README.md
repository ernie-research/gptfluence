# GPTfluence: On Training Data Influence of GPT Models
![image](https://github.com/ernie-research/gptfluence/blob/main/resources/overview.png)
Amidst the rapid advancements in generative language models, the investigation of how training data shapes the performance of GPT models is still emerging. This paper presents GPTfluence, a novel approach that leverages a featurized simulation to assess the impact of training examples on the training dynamics of GPT models. Our approach not only traces the influence of individual training instances on performance trajectories, such as loss and other key metrics, on targeted test points but also enables a comprehensive comparison with existing methods across various training scenarios in GPT models, ranging from 14 million to 2.8 billion parameters, across a range of downstream tasks. Contrary to earlier methods that struggle with generalization to new data, GPTfluence introduces a parameterized simulation of training dynamics, demonstrating robust generalization capabilities to unseen training data. This adaptability is evident across both fine-tuning and instruction-tuning scenarios, spanning tasks in natural language understanding and generation. 
# Requirements
To run the code, you should install the dependency libraries.
```
bash run_requirements.sh
```
# GPTDynamics
We release the GPTDynamics for training and testing the featurized simulator in [baidu/GPTDynamics](https://huggingface.co/datasets/baidu/GPTDynamics). To preprocess the data, you should follow the instructions below:
## Download
First, you need to download the data in the `GPTDynamics/gptdynamics` directory from [baidu/GPTDynamics](https://huggingface.co/datasets/baidu/GPTDynamics) locally. The data contains two parts, the first part is `GPTDynamics/gptdynamics/sft_tasks` and `GPTDynamics/gptdynamics/sft_tasks` correspond to the data samples for training, evaluation, and testing used in instruction fine-tuning and fine-tuning scenarios, respectively. You need to fill the local data paths into `DATASET_ADDITIONAL_ARGS` of `train.py` according to the task names to read the samples from the paths you provided; the second part is `GPTDynamics/gptdynamics/GPTDynamics.tar`, which contains the training runs, you need to extract:
```
tar -xvf GPTDynamics.tar
```
and placed in the repository directory.
## Instruction-tuning Scenario
Preprocess loss trajectory
```
python utils/construct_runs-data-flan-multi-thread.py
```
Preprocess metric（BLEU/ROUGE score）trajectory  
```
python utils/construct_runs-data-flan-metric-multi-thread.py
```
## Fine-tuning Scenario
Preprocess loss trajectory
```
python utils/construct_runs-data.py
```
Preprocess metric（BLEU/ROUGE score）trajectory
```
python utils/construct_runs-data-metric.py
```
    
## Draw loss/BLEU/ROUGE-L trajectory (Optional)
```
python utils/draw_gt_curves.py
```
# Featurized Simulator
## Set up training, validation, and testing data
Before training and inference, you should specify training, validation, and test data in the corresponding scripts. 
You should specify the simulator's training and validation data in `train.py` via the command line argument `--data_paths_dict`; and the simulator's test data in `test.py` via the command line argument `--data_paths_dict`. If you follow the steps above to download our open-source GPTDynamics data, you don't have to set it up additionally `-- data_paths_dict` and use the default configuration of the script species.

 ## Training
 ```
 bash run_enc_sim.sh
```
  
 ## Evaluation
 Predict loss
 ```
 bash auto_test.sh
 ```
 Predict metric(BLEU/ROUGE score)
 ```
 bash auto_test_metric.sh
 ```
 # Citation
 For attribution in academic contexts, please cite this work as:
 ```
@misc{liu2024training,
      title={On Training Data Influence of GPT Models}, 
      author={Qingyi Liu and Yekun Chai and Shuohuan Wang and Yu Sun and Qiwei Peng and Keze Wang and Hua Wu},
      year={2024},
      eprint={2404.07840},
      archivePrefix={arXiv},
      primaryClass={id='cs.CL' full_name='Computation and Language' is_active=True alt_name='cmp-lg' in_archive='cs' is_general=False description='Covers natural language processing. Roughly includes material in ACM Subject Class I.2.7. Note that work on artificial languages (programming languages, logics, formal systems) that does not explicitly address natural-language issues broadly construed (natural-language processing, computational linguistics, speech, text retrieval, etc.) is not appropriate for this area.'}
}
 ```
