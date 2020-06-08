# GhostNet

GhostNet: More Features from Cheap Operations. CVPR 2020. [[arXiv]](https://arxiv.org/abs/1911.11907)

By Kai Han, Yunhe Wang, Qi Tian, Jianyuan Guo, Chunjing Xu, Chang Xu.

- **Approach**

<div align="center">
   <img src="./fig/ghost_module.png" width="720">
</div>

- **Performance**

GhostNet beats other SOTA lightweight CNNs such as **MobileNetV3** and **FBNet**.

<div align="center">
   <img src="./fig/flops_latency.png" width="720">
</div>

## Code

This repo provides TensorFlow code and PyTorch code:
- Tensorflow: [./tensorflow](https://github.com/huawei-noah/ghostnet/tensorflow)
- PyTorch: [./pytorch](https://github.com/huawei-noah/ghostnet/pytorch)

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
