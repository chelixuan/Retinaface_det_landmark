import os
import math
from tqdm import tqdm
import matplotlib.pyplot as plt

gt_root = '/home/wybj/chelx/dataset/M3_dataset/TAG/'
pred_root = '/home/wybj/chelx/ckpt/QC_code/Retinaface/base_mobile0.25/test_txt/'

folders = ['01', '02', '08']

def eucliDist(A,B):
    return math.sqrt(sum([(a - b)**2 for (a,b) in zip(A,B)]))

mean_edc = []
max_edc = []
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
        # if len(pred_info) > 3:
        #     print(len(pred_info))
        #     exit()
        pred_points = pred_info[1].strip().split(' ')

        pred_p1_x, pred_p1_y = float(pred_points[0]), float(pred_points[1])
        pred_p2_x, pred_p2_y = float(pred_points[2]), float(pred_points[3])
        pred_p3_x, pred_p3_y = float(pred_points[4]), float(pred_points[5])
        pred_p4_x, pred_p4_y = float(pred_points[6]), float(pred_points[7])

        # print('-'*80)
        # print('gt vs pred :')
        # print(f'point1 : {(gt_p1_x, gt_p1_y)}  {(pred_p1_x, pred_p1_y)}')
        # print(f'point2 : {(gt_p2_x, gt_p2_y)}  {(pred_p2_x, pred_p2_y)}')
        # print(f'point3 : {(gt_p3_x, gt_p3_y)}  {(pred_p3_x, pred_p3_y)}')
        # print(f'point4 : {(gt_p4_x, gt_p4_y)}  {(pred_p4_x, pred_p4_y)}')
        # print()

        edc_dis_1 = eucliDist((gt_p1_x, gt_p1_y), (pred_p1_x, pred_p1_y))
        edc_dis_2 = eucliDist((gt_p2_x, gt_p2_y), (pred_p2_x, pred_p2_y))
        edc_dis_3 = eucliDist((gt_p3_x, gt_p3_y), (pred_p3_x, pred_p3_y))
        edc_dis_4 = eucliDist((gt_p4_x, gt_p4_y), (pred_p4_x, pred_p4_y))

        # print('edc_dis_1 = ', edc_dis_1)
        # print('edc_dis_2 = ', edc_dis_2)
        # print('edc_dis_3 = ', edc_dis_3)
        # print('edc_dis_4 = ', edc_dis_4)
        # print()

        mean_edc_dis = (edc_dis_1 + edc_dis_2 + edc_dis_3 + edc_dis_4) / 4
        max_edc_dis = max(edc_dis_1,  edc_dis_2, edc_dis_3, edc_dis_4)
        print('mean_edc_dis = ', round(mean_edc_dis, 3))
        print('max_edc_dis = ', round(max_edc_dis, 3))
        print()

        # mean_edc.append(mean_edc_dis)
        # max_edc.append(max_edc_dis)

        if mean_edc_dis > 100 or max_edc_dis > 100:
            continue
        else:
            mean_edc.append(mean_edc_dis)
            max_edc.append(max_edc_dis)
        # exit()

assert len(mean_edc) == len(max_edc), "please check edc collect process"
x = list(range(0, len(mean_edc)))

fig,axes = plt.subplots(nrows=1, ncols=2)
ax1 = axes[0]
ax2 = axes[1]

ax1.scatter(x, mean_edc, s=1)
ax1.set_ylabel('mean_edc')
ax2.scatter(x, max_edc, s=1, color='green')
ax2.set_ylabel('max_edc')

plt.savefig('./pred_offset.png')
