# CV Backbones 
including GhostNet, TinyNet, TNT (Transformer in Transformer), AugViT, WaveMLP developed by Huawei Noah's Ark Lab.
- [GhostNet Code](#ghostnet-code)
- [TinyNet Code](#tinynet-code)
- [TNT Code](#tnt-code)
- [PyramidTNT Code](#tnt-code)
- [LegoNet Code](#legonet-code)
- [Versatile Filters Code](#versatile-filters-code)
- [AugViT Code](#augvit-code)
- [ WaveMLP Code](#wavemlp-code)
- [Citation](#citation)
- [Other versions](#other-versions-of-ghostNet)

**News**

2022/06 The code of [Vision GNN (ViG)](https://arxiv.org/abs/2206.00272) will be released as soon. 

2022/02/06 Transformer in Transformer is selected as the **[Most Influential NeurIPS 2021 Papers](https://www.paperdigest.org/2022/02/most-influential-nips-papers-2022-02/)**.

2022/01/06 The extended version of [GhostNet](https://github.com/huawei-noah/CV-backbones/tree/master/ghostnet_pytorch) is accepted by IJCV.

2021/09/28 The paper of TNT (Transformer in Transformer) is accepted by NeurIPS 2021.

2021/09/18 The extended version of [Versatile Filters](https://github.com/huawei-noah/CV-backbones/tree/master/versatile_filters) is accepted by T-PAMI.

2021/08/30 GhostNet paper is selected as the **[Most Influential CVPR 2020 Papers](https://www.paperdigest.org/2021/08/most-influential-cvpr-papers-2021-08/)**.

2020/10/31 GhostNet+TinyNet achieves better performance. See details in our NeurIPS 2020 paper: [arXiv](https://arxiv.org/abs/2010.14819).

---

## GhostNet Code

This repo provides GhostNet **pretrained models** and **inference code** for TensorFlow and PyTorch:
- Tensorflow: [./ghostnet_tensorflow](https://github.com/huawei-noah/CV-backbones/tree/master/ghostnet_tensorflow) with pretrained model.
- PyTorch: [./ghostnet_pytorch](https://github.com/huawei-noah/CV-backbones/tree/master/ghostnet_pytorch) with pretrained model.
- We also opensource code on [MindSpore Hub](https://www.mindspore.cn/resources/hub) and [MindSpore Model Zoo](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/cv).

For **training**, please refer to [tinynet](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/cv/tinynet) or [timm](https://rwightman.github.io/pytorch-image-models/training_hparam_examples/#mobilenetv3-large-100-75766-top-1-92542-top-5).

## TinyNet Code

This repo provides TinyNet **pretrained models** and **inference code** for PyTorch:
- PyTorch: [./tinynet_pytorch](https://github.com/huawei-noah/CV-backbones/tree/master/tinynet_pytorch) with pretrained model.
- We also opensource training code on [MindSpore Model Zoo](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/cv).

## TNT Code

This repo provides **training code** and **pretrained models** of TNT (Transformer in Transformer) for PyTorch:
- PyTorch: [./tnt_pytorch](https://github.com/huawei-noah/CV-backbones/tree/master/tnt_pytorch).
- We also opensource code on [MindSpore Model Zoo](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/cv/TNT).

The code of PyramidTNT is also released: 
- PyTorch: [./tnt_pytorch](https://github.com/huawei-noah/CV-backbones/tree/master/tnt_pytorch).

## LegoNet Code
This repo provides the implementation of paper [LegoNet: Efficient Convolutional Neural Networks with Lego Filters (ICML 2019)](http://proceedings.mlr.press/v97/yang19c/yang19c.pdf)
- PyTorch: [./legonet_pytorch](https://github.com/huawei-noah/CV-backbones/tree/master/legonet_pytorch).

## Versatile Filters Code
This repo provides the implementation of paper [Learning Versatile Filters for Efficient Convolutional Neural Networks (NeurIPS 2018)](https://papers.nips.cc/paper/7433-learning-versatile-filters-for-efficient-convolutional-neural-networks)
- PyTorch: [./versatile_filters](https://github.com/huawei-noah/CV-backbones/tree/master/versatile_filters).

## AugViT Code

This repo provides the implementation of paper [Augmented Shortcuts for Vision Transformers (NeurIPS 2021)](https://proceedings.neurips.cc/paper/2021/file/818f4654ed39a1c147d1e51a00ffb4cb-Paper.pdf)
- PyTorch: [./augvit_pytorch](https://github.com/huawei-noah/CV-backbones/tree/master/augvit_pytorch).
- We also release the code on [MindSpore Model Zoo](https://gitee.com/mindspore/models/tree/master/research/cv/augvit).

## WaveMLP Code

This repo provides the implementation of paper [An Image Patch is a Wave: Quantum Inspired Vision MLP (CVPR 2022)](https://arxiv.org/pdf/2111.12294.pdf)
- PyTorch: [./wavemlp_pytorch](https://github.com/huawei-noah/CV-Backbones/tree/master/wavemlp_pytorch).
- We also release the code on [MindSpore Model Zoo](https://gitee.com/mindspore/models/tree/master/research/cv/wave_mlp).

## Citation
```
@inproceedings{ghostnet,
  title={GhostNet: More Features from Cheap Operations},
  author={Han, Kai and Wang, Yunhe and Tian, Qi and Guo, Jianyuan and Xu, Chunjing and Xu, Chang},
  booktitle={CVPR},
  year={2020}
}
@inproceedings{tinynet,
  title={Model Rubik’s Cube: Twisting Resolution, Depth and Width for TinyNets},
  author={Han, Kai and Wang, Yunhe and Zhang, Qiulin and Zhang, Wei and Xu, Chunjing and Zhang, Tong},
  booktitle={NeurIPS},
  year={2020}
}
@inproceedings{tnt,
  title={Transformer in transformer},
  author={Han, Kai and Xiao, An and Wu, Enhua and Guo, Jianyuan and Xu, Chunjing and Wang, Yunhe},
  booktitle={NeurIPS},
  year={2021}
}
@inproceedings{legonet,
    title={LegoNet: Efficient Convolutional Neural Networks with Lego Filters},
    author={Yang, Zhaohui and Wang, Yunhe and Liu, Chuanjian and Chen, Hanting and Xu, Chunjing and Shi, Boxin and Xu, Chao and Xu, Chang},
    booktitle={ICML},
    year={2019}
  }
@inproceedings{wang2018learning,
  title={Learning versatile filters for efficient convolutional neural networks},
  author={Wang, Yunhe and Xu, Chang and Chunjing, XU and Xu, Chao and Tao, Dacheng},
  booktitle={NeurIPS},
  year={2018}
}
@inproceedings{tang2021augmented,
      title={Augmented shortcuts for vision transformers},
      author={Tang, Yehui and Han, Kai and Xu, Chang and Xiao, An and Deng, Yiping and Xu, Chao and Wang, Yunhe},
      booktitle={NeurIPS},
      year={2021}
}
@inproceedings{tang2022image,
  title={An Image Patch is a Wave: Phase-Aware Vision MLP},
  author={Tang, Yehui and Han, Kai and Guo, Jianyuan and Xu, Chang and Li, Yanxi and Xu, Chao and Wang, Yunhe},
  booktitle={CVPR},
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
