import fire
import numpy as np

def main(task_name, log_file_path):
    print("任务: ", task_name)
    with open(log_file_path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
        valid_mse_loss_list = [] 
        for line in lines:
            if "valid mse loss" in line:
                valid_mse_loss_list.append(float(line.split(' ')[-1]))
        valid_mse_loss_np = np.array(valid_mse_loss_list)
        print("验证总步数: ", len(valid_mse_loss_list))
        print("验证集loss最小值: ", min(valid_mse_loss_list))
        print("对应步数: ", np.argmin(valid_mse_loss_np))
    print('done\n')

if __name__ == '__main__':
    fire.Fire(main)