import json
import numpy as np
from scipy.optimize import minimize

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


if __name__ == "__main__":
    path = 'test/tracincp/pred_and_gt_loss_trajectories.out'
    output_path = 'test/tracincp/rescaled_pred_and_gt_loss_trajectories.out'
    with open(output_path, 'w', encoding='utf-8') as w:
        with open(path, 'r', encoding='utf-8') as f:
            all_steps_mse = []
            for line in f.readlines():
                line = json.loads(line)
                rescaled_line = []
                for l in line:
                    gt_loss_np = np.array(l['gt_loss'])
                    pred_loss_np = np.array(l['pred_loss'])
                    # break
                    # 最小化mse
                    ### 平移pred曲线
                    pred_loss_np[1:] = pred_loss_np[1:] - min(pred_loss_np[1:]) + 1.
                    min_mse, rescale_factor = mini_func(gt_loss_np, pred_loss_np)
                    rescale_factor = np.array([1.] + [rescale_factor] * (len(gt_loss_np) - 1))
                    pred_loss_np *= rescale_factor
                    # 计算all steps mse
                    all_steps_mse.append(np.mean((pred_loss_np - gt_loss_np)**2))
                    print(min_mse)
                    l['rescaled_pred_loss'] = list(pred_loss_np)
                    rescaled_line.append(l)
                print(json.dumps(rescaled_line, ensure_ascii=False), file=w)
            print('all steps mse: ', np.mean(all_steps_mse))
            
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

res = mini_func(gt_loss_np, pred_loss_np)
# 输出结果
print("最小值：", res[0])
print("最优解：", res[1])
