import os
import math
from tqdm import tqdm
import matplotlib.pyplot as plt

gt_root = '/home/chelx/dataset/M3_dataset/TAG/'
pred_root = '/home/chelx/ckpt/QC_code/Retinaface/base_mobile0.25/test_txt/'
save_scatter_path = './pred_offset.png'

folders = ['01', '02', '08']

def eucliDist(A,B):
    return math.sqrt(sum([(a - b)**2 for (a,b) in zip(A,B)]))

mean_euc = []
max_euc = []
for i in range(len(folders)):
    folder = folders[i]
    print(f'\n{i+1}/{len(folders)} : {folder}')
    gt_txt_path = gt_root + folder + '/point/'
    pred_txt_path = pred_root + folder + '/'

    txts = os.listdir(pred_txt_path)
    for txt in tqdm(txts):
        gt = open(gt_txt_path + txt, 'r')
        gt_info = gt.read()
        gt_info = gt_info.split('\n')
        gt_points = gt_info[1].strip().split(' ')

        gt_p1_x, gt_p1_y = float(gt_points[0]), float(gt_points[1])
        gt_p2_x, gt_p2_y = float(gt_points[2]), float(gt_points[3])
        gt_p3_x, gt_p3_y = float(gt_points[4]), float(gt_points[5])
        gt_p4_x, gt_p4_y = float(gt_points[6]), float(gt_points[7])

        pred = open(pred_txt_path + txt, 'r')
        pred_info = pred.read()
        pred_info = pred_info.split('\n')
        if len(pred_info) <=1 :
            continue
        pred_points = pred_info[1].strip().split(' ')

        pred_p1_x, pred_p1_y = float(pred_points[0]), float(pred_points[1])
        pred_p2_x, pred_p2_y = float(pred_points[2]), float(pred_points[3])
        pred_p3_x, pred_p3_y = float(pred_points[4]), float(pred_points[5])
        pred_p4_x, pred_p4_y = float(pred_points[6]), float(pred_points[7])

        euc_dis_1 = eucliDist((gt_p1_x, gt_p1_y), (pred_p1_x, pred_p1_y))
        euc_dis_2 = eucliDist((gt_p2_x, gt_p2_y), (pred_p2_x, pred_p2_y))
        euc_dis_3 = eucliDist((gt_p3_x, gt_p3_y), (pred_p3_x, pred_p3_y))
        euc_dis_4 = eucliDist((gt_p4_x, gt_p4_y), (pred_p4_x, pred_p4_y))

        mean_euc_dis = (euc_dis_1 + euc_dis_2 + euc_dis_3 + euc_dis_4) / 4
        max_euc_dis = max(euc_dis_1,  euc_dis_2, euc_dis_3, euc_dis_4)

        # mean_euc.append(mean_euc_dis)
        # max_euc.append(max_euc_dis)

        if mean_euc_dis > 100 or max_euc_dis > 100:
            continue
        else:
            mean_euc.append(mean_euc_dis)
            max_euc.append(max_euc_dis)

assert len(mean_euc) == len(max_euc), "please check euc collect process"
x = list(range(0, len(mean_euc)))

fig,axes = plt.subplots(nrows=1, ncols=2)
ax1 = axes[0]
ax2 = axes[1]

ax1.scatter(x, mean_euc, s=1)
ax1.set_ylabel('mean_euc')
ax2.scatter(x, max_euc, s=1, color='green')
ax2.set_ylabel('max_euc')

plt.savefig(save_scatter_path)

print('mean_mean-euc = ', round(sum(mean_euc) / len(mean_euc), 3))
print('mean_max-euc = ', round(sum(max_euc) / len(max_euc), 3))
