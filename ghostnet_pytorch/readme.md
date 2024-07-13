# GhostNet

GhostNet: More Features from Cheap Operations. CVPR 2020. [[arXiv]](https://arxiv.org/abs/1911.11907) [[Most Influential CVPR 2020 Papers]](https://www.paperdigest.org/2021/08/most-influential-cvpr-papers-2021-08/)

By Kai Han, Yunhe Wang, Qi Tian, Jianyuan Guo, Chunjing Xu, Chang Xu.

- **Approach**

<div align="center">
   <img src="../fig/ghost_module.png" width="720">
</div>

- **Performance**

GhostNet beats other SOTA lightweight CNNs such as MobileNetV3 and FBNet.

<div align="center">
   <img src="../fig/flops_latency.png" width="720">
</div>

## Implementation

This folder provides the PyTorch code and pretrained model of GhostNet on ImageNet.

`ghostnet.py` implemented `GhostModule` and `GhostNet` in [CVPR paper](https://arxiv.org/abs/1911.11907).

### Requirements
The code was verified on Python3.6, PyTorch-1.0+.

### Usage
Run `python validate.py --data=/path/to/imagenet/dir/` to evaluate on `val` set.

You'll get the accuracy: top-1 acc=`0.7398` and top-5 acc=`0.9146` with only `142M` Flops (or say MAdds).

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
```

## Citation
```
@inproceedings{ghostnet2020,
  title={GhostNet: More Features from Cheap Operations},
  author={Han, Kai and Wang, Yunhe and Tian, Qi and Guo, Jianyuan and Xu, Chunjing and Xu, Chang},
  booktitle={CVPR},
  year={2020}
}
@article{ghostnet2022,
  title={GhostNets on Heterogeneous Devices via Cheap Operations},
  author={Kai Han, Yunhe Wang, Chang Xu, Jianyuan Guo, Chunjing Xu, Enhua Wu and Qi Tian},
  journal={IJCV},
  year={2022}
}
```

## Other versions of GhostNet
This repo provides the TensorFlow/PyTorch code of GhostNet. Other versions and applications can be found in the following:

0. timm: [code with pretrained model](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/ghostnet.py)
1. Darknet: [cfg file](https://github.com/AlexeyAB/darknet/files/3997987/ghostnet.cfg.txt), and [description](https://github.com/AlexeyAB/darknet/issues/4418)
2. Gluon/Keras/Chainer: [code](https://github.com/osmr/imgclsmob)
3. Paddle: [code](https://github.com/PaddlePaddle/PaddleClas/blob/master/ppcls/modeling/architectures/ghostnet.py)
4. Bolt inference framework: [benckmark](https://github.com/huawei-noah/bolt/blob/master/docs/BENCHMARK.md)
5. Human pose estimation: [code](https://github.com/tensorboy/centerpose/blob/master/lib/models/backbones/ghost_net.py)
6. YOLO with GhostNet backbone: [code](https://github.com/HaloTrouvaille/YOLO-Multi-Backbones-Attention)
7. Face recognition: [cavaface](https://github.com/cavalleria/cavaface.pytorch/blob/master/backbone/ghostnet.py), [FaceX-Zoo](https://github.com/JDAI-CV/FaceX-Zoo), [TFace](https://github.com/Tencent/TFace)
