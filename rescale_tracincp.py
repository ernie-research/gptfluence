import json
import numpy as np
from scipy.optimize import minimize
from scipy import stats

# 定义一元二次方程
def func(x, gt, pred):
    return np.sum((pred*x[0] + x[1] - gt)**2)

def mini_func(gt, pred):
    gt = gt[1:]
    pred = pred[1:]
    a = np.sum(pred**2)
    b = -2*np.sum(pred*gt)
    c = np.sum(gt**2)
    return (4*a*c - b**2) / (4*a), -b / (2*a)

def mini_func2(gt_list, pred_list, tot_influence_list):
    a, b, c = 0.0, 0.0, 0.0

    for gt, pred, tot_influence in zip(gt_list, pred_list, tot_influence_list):
        diff = gt[0] - gt
        diff = diff[1:]
        gt = gt[1:]
        pred = pred[1:]
        a += np.sum(tot_influence**2)
        b += np.sum(-2*tot_influence*diff)
        c += np.sum(diff**2)
    return (4*a*c - b**2) / (4*a), -b / (2*a)


if __name__ == "__main__":
    TASK = 'wmt16_de_en_grad-dot'
    path = f'output/flan/tracincp_seed-5/{TASK}/pred_and_gt_loss_trajectories.out'
    output_path = f'output/flan/tracincp_seed-5/{TASK}/rescaled_pred_and_gt_loss_trajectories.out'

    with open(output_path, 'w', encoding='utf-8') as w:
        with open(path, 'r', encoding='utf-8') as f:
            all_steps_mse = []
            all_steps_mae = []
            # last_step = {}
            gt_list, pred_list, tot_influence_list = [], [], []
            lines = f.readlines()
            for line in lines:
                line = json.loads(line)
                rescaled_line = []
                for l in line:
                    gt_loss_np = np.array(l['gt_loss'])
                    pred_loss_np = np.array(l['pred_loss'])
                    influence = pred_loss_np[:-1] - pred_loss_np[1:]

                    tot_influence = []
                    cur_influence = 0.
                    for i in range(len(influence)):
                        cur_influence += influence[i]
                        tot_influence.append(cur_influence)
                    
                    gt_list.append(gt_loss_np)
                    pred_list.append(pred_loss_np)
                    tot_influence_list.append(np.array(tot_influence))
            res = mini_func2(gt_list, pred_list, tot_influence_list)
            print("最小值：", res[0])
            print("最优解：", res[1])

            gt_list, pred_list, tot_influence_list = [], [], []
            from collections import defaultdict
            from functools import partial
            last_step = defaultdict(
                partial(defaultdict, list)
            )

            for line in lines:
                line = json.loads(line)
                rescaled_line = []
                
                for run_id, l in enumerate(line):
                    gt_loss_np = np.array(l['gt_loss'])
                    pred_loss_np = np.array(l['pred_loss'])
                    influence = pred_loss_np[:-1] - pred_loss_np[1:]

                    tot_influence = []
                    cur_influence = 0.
                    for i in range(len(influence)):
                        cur_influence += influence[i]
                        tot_influence.append(cur_influence)
                    
                    for i, inf in enumerate(influence):
                        step = i + 1
                        pred_loss_np[step] = pred_loss_np[step-1] - res[1] * inf
                    
                    # 计算all steps mse
                    all_steps_mse.append(np.mean((pred_loss_np - gt_loss_np)**2))
                    all_steps_mae.append(np.mean(np.abs(pred_loss_np - gt_loss_np)))
                    l['rescaled_pred_loss'] = list(pred_loss_np)
                    rescaled_line.append(l)
                    last_step[run_id]['gt'].append(gt_loss_np[-1])
                    last_step[run_id]['pred'].append(pred_loss_np[-1])

                print(json.dumps(rescaled_line, ensure_ascii=False), file=w)
            print('all steps mse: ', np.mean(all_steps_mse))
            print('all steps mse (std): ', np.std(all_steps_mse))

            print('all steps mae: ', np.mean(all_steps_mae))
            print('all steps mae (std): ', np.std(all_steps_mae))
            
            # last_step_spearman_np = np.array(last_step_spearman_list)
            # last_step_spearman_mean = last_step_spearman_np.mean()
            # last_step_spearman_std = last_step_spearman_np.std()
            # print(f"测试集last step spearman 均值: {last_step_spearman_mean}, 标准差: {last_step_spearman_std}")
            last_step_list = []
            for run_id, ls in last_step.items():
                sp = stats.spearmanr(ls['gt'], ls['pred'] ).statistic
                last_step_list.append(sp)
            print('last step spearman', np.mean(last_step_list))
            print('last step spearman (std)', np.std(last_step_list))
            # break



# # 定义初始值
# x0 = [1, 0]
# gt = gt_loss_np
# pred = pred_loss_np

# # 使用minimize方法求解最小值
# res = minimize(func, x0, args=(gt, pred))

# # 输出结果
# print("最小值：", res.fun)
# print("最优解：", res.x)
# print("迭代终止是否成功：", res.success)
# print("迭代终止原因：", res.message)

# res = mini_func(gt_loss_np, pred_loss_np)
# # 输出结果
# print("最小值：", res[0])
# print("最优解：", res[1])
