# Efficient AI Backbones 
including GhostNet, TNT (Transformer in Transformer), AugViT, WaveMLP and ViG developed by Huawei Noah's Ark Lab.
- [Model zoo](#model-zoo)
- [Citation](#citation)
- [Other versions](#other-versions-of-ghostnet)

**News**

2022/12/01 The code of NeurIPS 2022 (Spotlight) [GhostNetV2](https://arxiv.org/abs/2211.12905) is released at [./ghostnetv2_pytorch](https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/ghostnetv2_pytorch).

2022/11/13 The code of IJCV 2022 [G-Ghost RegNet](https://arxiv.org/abs/2201.03297) is released at [./g_ghost_pytorch](https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/g_ghost_pytorch). 

2022/06/17 The code of NeurIPS 2022 [Vision GNN (ViG)](https://arxiv.org/abs/2206.00272) is released at [./vig_pytorch](https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/vig_pytorch). 

2022/02/06 Transformer in Transformer (TNT) is selected as the **[Most Influential NeurIPS 2021 Papers](https://www.paperdigest.org/2022/02/most-influential-nips-papers-2022-02/)**.

2021/09/28 The paper of TNT (Transformer in Transformer) is accepted by [NeurIPS 2021](https://arxiv.org/abs/2103.00112).

2021/09/18 The extended version of [Versatile Filters](https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/versatile_filters) is accepted by T-PAMI.

2021/08/30 GhostNet paper is selected as the **[Most Influential CVPR 2020 Papers](https://www.paperdigest.org/2021/08/most-influential-cvpr-papers-2021-08/)**.

2020/10/31 GhostNet+TinyNet achieves better performance. See details in our NeurIPS 2020 paper: [arXiv](https://arxiv.org/abs/2010.14819).

---

## Model zoo

| Model | Paper | Pytorch code | Tensorflow code | MindSpore code |
| - | - | - | - | - |
| GhostNet | GhostNet: More Features from Cheap Operations. [[CVPR 2020]](https://arxiv.org/abs/1911.11907) | [./ghostnet_pytorch](https://github.com/huawei-noah/CV-backbones/tree/master/ghostnet_pytorch) | [./ghostnet_tensorflow](https://github.com/huawei-noah/CV-backbones/tree/master/ghostnet_tensorflow) | [MindSpore Model Zoo](https://gitee.com/mindspore/models/tree/master/research/cv/ghostnet) |
| GhostNetV2 | GhostNetV2: Enhance Cheap Operation with Long-Range Attention. [[NeurIPS 2022 Spotlight]](https://arxiv.org/abs/2211.12905) | [./ghostnetv2_pytorch](https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/ghostnetv2_pytorch) | - | [MindSpore Model Zoo](https://gitee.com/mindspore/models/tree/master/research/cv/ghostnetv2) |
| TinyNet | Model Rubik’s Cube: Twisting Resolution, Depth and Width for TinyNets. [[NeurIPS 2020]](https://arxiv.org/abs/2010.14819) | [./tinynet_pytorch](https://github.com/huawei-noah/CV-backbones/tree/master/tinynet_pytorch) | - | [MindSpore Model Zoo](https://gitee.com/mindspore/models/tree/master/research/cv/tinynet) |
| TNT | Transformer in Transformer. [[NeurIPS 2021]](https://arxiv.org/abs/2103.00112) | [./tnt_pytorch](https://github.com/huawei-noah/CV-backbones/tree/master/tnt_pytorch) | - | [MindSpore Model Zoo](https://gitee.com/mindspore/models/tree/master/research/cv/TNT) |
| PyramidTNT | PyramidTNT: Improved Transformer-in-Transformer Baselines with Pyramid Architecture. [[CVPR 2022 Workshop]](https://arxiv.org/abs/2201.00978)| [./tnt_pytorch](https://github.com/huawei-noah/CV-backbones/tree/master/tnt_pytorch) | - | [MindSpore Model Zoo](https://gitee.com/mindspore/models/tree/master/research/cv/TNT) |
| LegoNet | LegoNet: Efficient Convolutional Neural Networks with Lego Filters. [[ICML 2019]](http://proceedings.mlr.press/v97/yang19c/yang19c.pdf) | [./legonet_pytorch](https://github.com/huawei-noah/CV-backbones/tree/master/legonet_pytorch) | - | - |
| Versatile Filters | Learning Versatile Filters for Efficient Convolutional Neural Networks. [[NeurIPS 2018]](https://papers.nips.cc/paper/7433-learning-versatile-filters-for-efficient-convolutional-neural-networks) | [./versatile_filters](https://github.com/huawei-noah/CV-backbones/tree/master/versatile_filters) | - | - |
| AugViT | Augmented Shortcuts for Vision Transformers. [[NeurIPS 2021]](https://proceedings.neurips.cc/paper/2021/file/818f4654ed39a1c147d1e51a00ffb4cb-Paper.pdf) | [./augvit_pytorch](https://github.com/huawei-noah/CV-backbones/tree/master/augvit_pytorch) | - | [MindSpore Model Zoo](https://gitee.com/mindspore/models/tree/master/research/cv/augvit) |
| WaveMLP | An Image Patch is a Wave: Quantum Inspired Vision MLP. [[CVPR 2022]](https://arxiv.org/pdf/2111.12294.pdf) | [./wavemlp_pytorch](https://github.com/huawei-noah/CV-Backbones/tree/master/wavemlp_pytorch) | - | [MindSpore Model Zoo](https://gitee.com/mindspore/models/tree/master/research/cv/wave_mlp) |
| ViG | Vision GNN: An Image is Worth Graph of Nodes. [[NeurIPS 2022]](https://arxiv.org/abs/2206.00272) | [./vig_pytorch](https://github.com/huawei-noah/CV-Backbones/tree/master/vig_pytorch) | - | [MindSpore Model Zoo](https://gitee.com/mindspore/models/tree/master/research/cv/ViG) |


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
@inproceedings{han2022vig,
  title={Vision GNN: An Image is Worth Graph of Nodes}, 
  author={Kai Han and Yunhe Wang and Jianyuan Guo and Yehui Tang and Enhua Wu},
  booktitle={NeurIPS},
  year={2022}
}
@article{tang2022ghostnetv2,
  title={GhostNetV2: Enhance Cheap Operation with Long-Range Attention},
  author={Tang, Yehui and Han, Kai and Guo, Jianyuan and Xu, Chang and Xu, Chao and Wang, Yunhe},
  journal={arXiv preprint arXiv:2211.12905},
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
