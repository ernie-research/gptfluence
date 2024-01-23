import json
import os
from collections import defaultdict
import tqdm
import time

start = time.time()

ip_seed_list = [
    {"ip": "10.127.27.144", "seed": [
        62, 63, 64, 65, 66, 67, 68, 69,
        72, 73, 74, 75, 76, 77, 78, 79,
        82, 83, 84, 85, 86, 87, 88, 89,
        92, 93, 94, 95, 96, 97, 98, 99,
    ]}, # 44: 600
]

for item in ip_seed_list:
    ip = item['ip']
    seed_list = item['seed']
    for seed in seed_list:
        print(f"process wmt18-tr-en_tgt-en ip: {ip}, seed: {seed}",)
        output_dir = f"./runs/wmt18-tr-en-few-shot_tgt-en_loss-only-tgt_bs-4_shot-200_sample-200_seed-{seed}"
        if os.path.exists(output_dir):
            print(f'{output_dir} exists')
        else:
            os.makedirs(output_dir)

        # 加载log文件，将数据集转化为list
        log_path = "/root/paddlejob/workspace/env_run/liuqingyi01/tda_output/{ip}_wmt18-tr-en-few-shot_tgt-en_loss-only-tgt_bs-4_shot-200_sample-200_seed-{seed}/wmt18-tr-en-few-shot_tgt-en_loss-only-tgt_bs-4_shot-200_sample-200_seed-{seed}.out".format(ip=ip, seed=seed)
        train_data_list = []
        with open(log_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().replace('\'', '\"')
                if "\"samples_id\"" in line and "\"loss\"" in line and "\"step\"" in line:
                    train_data_list.append(json.loads(line))

        train_samples_path = os.path.join(output_dir, 'train_samples_id.json')
        if os.path.exists(train_samples_path):
            print(f'{train_samples_path} exists')
        else:
            with open(train_samples_path, 'w', encoding='utf-8') as f:
                for i in range(len(train_data_list)):
                    o = train_data_list[i]
                    step = o['step']
                    samples_id = o['samples_id']
                    print(json.dumps({'step': step, 'samples_id':samples_id}), file=f)
            print(f"training steps: {len(train_data_list)}")

        # 加载output文件
        file_num = 600
        output_list = defaultdict(list)

        output_path = os.path.join(output_dir, 'all_loss_trajectory.out')
        if os.path.exists(output_path):
            print(f'{output_path} exists')
        else:
            for checkpoint_step in range(1, file_num+1):
                output_path = f"/root/paddlejob/workspace/env_run/liuqingyi01/tda_output/{ip}_wmt18-tr-en-few-shot_tgt-en_loss-only-tgt_bs-4_shot-200_sample-200_seed-{seed}/loss_wmt18_tr_en/loss_wmt18_tr_en_checkpoint-{checkpoint_step}.out"
                if not os.path.exists(output_path):
                    continue
                with open(output_path, "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        line = line.strip()
                        line = json.loads(line)
                        id = line['id']
                        loss = line['loss']
                        output_list[id].append({'step': checkpoint_step, 'loss': loss})
            output_path = os.path.join(output_dir, 'all_loss_trajectory.out')
            with open(output_path, 'w', encoding='utf-8') as f:
                for k, v in output_list.items():
                    print(json.dumps({'id': k, 'loss_trajectory': v}), file=f)
            print(f"num test examples: {len(output_list)}")
            print("\n")

end = time.time()
print(f"total time: {end - start}s")