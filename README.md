# RetinaFace in PyTorch

A [PyTorch](https://pytorch.org/) implementation of [RetinaFace: Single-stage Dense Face Localisation in the Wild](https://arxiv.org/abs/1905.00641). Model size only 1.7M, when Retinaface use mobilenet0.25 as backbone net. We also provide resnet50 as backbone net to get better result. The official code in Mxnet can be found [here](https://github.com/deepinsight/insightface/tree/master/RetinaFace).

## Detail Information
For more detail information, please refer to the original [Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface).

## Mobile or Edge device deploy
We also provide a set of Face Detector for edge device in [here](https://github.com/biubug6/Face-Detector-1MB-with-landmark) from python training to C++ inference.


### Contents
- [Data](#data)
- [Training](#training)
- [Evaluation](#evaluation)
- [TensorRT](#tensorrt)
- [References](#references)

## Data
Organise the dataset directory as follows:
```Shell 
  root/
    train/
      images/
      label.txt
    val/
      images/
      label.txt
```
ps: root: custom data root path.

### label format
```Shell
# image_name
bbox1_x1 bbox1_y1 bbox1_w bbox1_h point1_x point1_y point2_x point2_y point3_x point3_y point4_x point4_y
bbox2_x1 bbox2_y1 bbox2_w bbox2_h point1_x point1_y point2_x point2_y point3_x point3_y point4_x point4_y 
...
```
ps: 
  - 'bbox1_x1 bbox1_y1 bbox1_w bbox1_h' must be int format;
  - 'point1_x point1_y point2_x point2_y point3_x point3_y point4_x point4_y' can be float.

## Training
We provide restnet50 and mobilenet0.25 as backbone network to train model.
We trained Mobilenet0.25 on imagenet dataset and get 46.58%  in top 1. If you do not wish to train the model, we also provide trained model. Pretrain model  and trained model are put in [google cloud](https://drive.google.com/open?id=1oZRSG0ZegbVkVwUd8wUIQx8W7yfZ_ki1) and [baidu cloud](https://pan.baidu.com/s/12h97Fy1RYuqMMIV-RpzdPg) Password: fstq . The model could be put as follows:
```Shell
  ./weights/
      mobilenet0.25_Final.pth
      mobilenetV1X0.25_pretrain.tar
      Resnet50_Final.pth
```
1. Before training, you can check network configuration (e.g. batch_size, min_sizes and steps etc..) in ``data/config.py and train.py``.

2. Train the model using WIDER FACE:
  ```Shell
  CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --network resnet50 or
  CUDA_VISIBLE_DEVICES=0 python train.py --network mobile0.25
  ```


## Evaluation
### Evaluation custom val
1. Generate txt file
```Shell
python test_widerface.py \
       --trained_model weight_file \
       --network mobile0.25 or resnet50 \
       --save_folder your_txt_save_path \
       --dataset_folder your_val_image_path \
       --save_image (optional)
```
2. Evaluate landmark offset(use Euclidean distance ) 

```Shell
# change gt_txt_path and pred_txt_path
python landm_edc.py
```

## TensorRT
-[TensorRT](https://github.com/wang-xinyu/tensorrtx/tree/master/retinaface)

## References
- [Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface)
- [FaceBoxes](https://github.com/zisianw/FaceBoxes.PyTorch)
- [Retinaface (mxnet)](https://github.com/deepinsight/insightface/tree/master/RetinaFace)
```
@inproceedings{deng2019retinaface,
title={RetinaFace: Single-stage Dense Face Localisation in the Wild},
author={Deng, Jiankang and Guo, Jia and Yuxiang, Zhou and Jinke Yu and Irene Kotsia and Zafeiriou, Stefanos},
booktitle={arxiv},
year={2019}
```
