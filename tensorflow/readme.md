# GhostNet

GhostNet: More Features from Cheap Operations. CVPR 2020. [[arXiv]](https://arxiv.org/abs/1911.11907)

By Kai Han, Yunhe Wang, Qi Tian, Jianyuan Guo, Chunjing Xu, Chang Xu.

- **Approach**

<div align="center">
   <img src="../fig/ghost_module.png" width="720">
</div>

- **Performance**

GhostNet beats other SOTA lightweight CNNs such as **MobileNetV3** and **FBNet**.

<div align="center">
   <img src="../fig/flops_latency.png" width="720">
</div>

## Implementation

This repo provides the TensorFlow code and pretrained model of GhostNet on ImageNet. The PyTorch implementation can be found at [https://github.com/iamhankai/ghostnet.pytorch](https://github.com/iamhankai/ghostnet.pytorch).

`myconv2d.py` implemented `GhostModule` and `ghostnet.py` implemented `GhostNet`.

### Requirements
The code was verified on Python3.6, TensorFlow-1.13.1, Tensorpack-0.9.7. Not sure on other version.

### Usage
Run `python main.py --eval --data_dir=/path/to/imagenet/dir/ --load=./models/ghostnet_checkpoint` to evaluate on `val` set.

You'll get the accuracy: top-1 error=`0.26066`, top-5 error=`0.08614` with only `141M` Flops (or say MAdds).

### Data Preparation
ImageNet data dir should have the following structure, and `val` and `caffe_ilsvrc12` subdirs are essential:
```
dir/
  train/
    ...
  val/
    n01440764/
      ILSVRC2012_val_00000293.JPEG
      ...
    ...
  caffe_ilsvrc12/
    ...
```
caffe_ilsvrc12 data can be downloaded from http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz

## Citation
```
@inproceedings{ghostnet,
  title={GhostNet: More Features from Cheap Operations},
  author={Han, Kai and Wang, Yunhe and Tian, Qi and Guo, Jianyuan and Xu, Chunjing and Xu, Chang},
  booktitle={CVPR},
  year={2020}
}
```

## Other versions
This repo provides the TensorFlow code of GhostNet. Other versions can be found in the following:

0. Pytorch: [code](https://github.com/iamhankai/ghostnet.pytorch)
1. Darknet: [cfg file](https://github.com/AlexeyAB/darknet/files/3997987/ghostnet.cfg.txt), and [description](https://github.com/AlexeyAB/darknet/issues/4418)
2. Gluon/Keras/Chainer: [code](https://github.com/osmr/imgclsmob)
3. Pytorch for human pose estimation: [code](https://github.com/tensorboy/centerpose/blob/master/lib/models/backbones/ghost_net.py)
