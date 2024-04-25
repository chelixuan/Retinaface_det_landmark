import os
import urllib
import traceback
import time
import sys
import numpy as np
import cv2
from rknn.api import RKNN

ONNX_MODEL = '/home/chelx/Retinaface_rknn/ckpt/mobilenet0.25_Final_sim.onnx'
RKNN_MODEL = ONNX_MODEL[:-5] + '.rknn'

DATASET = './calib_dataset.txt'
QUANTIZE_ON = True

if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN(verbose=True)

    if not os.path.exists(ONNX_MODEL):
        print('model not exist')
        exit(-1)

    # pre-process config
    print('--> Config model')
    rknn.config(
                mean_values=[[0, 0, 0]],
                std_values=[[255, 255, 255]],
                # mean_values=[[104, 117, 123]],
			    # std_values=[[1, 1, 1]],
                target_platform = 'rk3566',
                # 开启后无论模型是否量化，均对模型输入节点进行强制量化；
                # quantize_input_node=QUANTIZE_ON,
                # added by clx@2024.03.06 ----------------------------------------------------------------
                # 是否要进行水平合并，默认值 False；如果模型是 inception v1/v3/v4 则建议打开
                # need_horizontal_merge = True,
                # 量化类型，默认为 'asymmetric_quantized-u8'。高精度可选择 'dynamic_fixed_point-i16',
                # quantized_dtype = 'asymmetric_quantized-16',
                # 量化参数优化算法，默认值为 normal；mmse可以取得更好的效果，但是比较慢;The MMSE can only used with asymmetric_quantized-u8
                # quantized_algorithm = 'mmse',
                # mmse 量化算法的迭代次数，默认值为3；迭代次数越多，精度越高；
                # mmse_epoch = 3,
                # batch_size=16,
                # 模型优化等级，默认值为3，打开所有优化选项；0表示关闭所有优化选项；
                optimization_level = 0, 
                # ----------------------------------------------------------------------------------------
                )
    print('done')

    # Load ONNX model
    print('--> Loading model')
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print('Load yolov5 failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=QUANTIZE_ON, dataset=DATASET)
    if ret != 0:
        print('Build yolov5 failed!')
        exit(ret)
    print('done')

    # Export RKNN model
    print('--> Export RKNN model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export yolov5rknn failed!')
        exit(ret)
    print('done')

