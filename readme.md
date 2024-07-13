# Efficient AI Backbones 
including GhostNet, TNT (Transformer in Transformer), AugViT, WaveMLP and ViG developed by Huawei Noah's Ark Lab.
- [News](#news)
- [Model zoo](#model-zoo)

## News

2024/02/27 The paper of ParameterNet is accepted by [CVPR 2024](https://arxiv.org/abs/2306.14525).

2022/12/01 The code of NeurIPS 2022 (Spotlight) [GhostNetV2](https://arxiv.org/abs/2211.12905) is released at [./ghostnetv2_pytorch](https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/ghostnetv2_pytorch).

2022/11/13 The code of IJCV 2022 [G-Ghost RegNet](https://arxiv.org/abs/2201.03297) is released at [./g_ghost_pytorch](https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/g_ghost_pytorch). 

2022/06/17 The code of NeurIPS 2022 [Vision GNN (ViG)](https://arxiv.org/abs/2206.00272) is released at [./vig_pytorch](https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/vig_pytorch). 

2022/02/06 Transformer in Transformer (TNT) is selected as the **[Most Influential NeurIPS 2021 Papers](https://www.paperdigest.org/2022/02/most-influential-nips-papers-2022-02/)**.

2021/09/18 The extended version of [Versatile Filters](https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/versatile_filters) is accepted by T-PAMI.

2021/08/30 GhostNet paper is selected as the **[Most Influential CVPR 2020 Papers](https://www.paperdigest.org/2021/08/most-influential-cvpr-papers-2021-08/)**.


## Model zoo

| Model | Paper | Pytorch code | MindSpore code |
| - | - | - | - |
| GhostNet | GhostNet: More Features from Cheap Operations. [[CVPR 2020]](https://arxiv.org/abs/1911.11907) | [./ghostnet_pytorch](https://github.com/huawei-noah/CV-backbones/tree/master/ghostnet_pytorch) | [MindSpore Model Zoo](https://gitee.com/mindspore/models/tree/master/research/cv/ghostnet) |
| GhostNetV2 | GhostNetV2: Enhance Cheap Operation with Long-Range Attention. [[NeurIPS 2022 Spotlight]](https://arxiv.org/abs/2211.12905) | [./ghostnetv2_pytorch](https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/ghostnetv2_pytorch) | [MindSpore Model Zoo](https://gitee.com/mindspore/models/tree/master/research/cv/ghostnetv2) |
| G-GhostNet | GhostNets on Heterogeneous Devices via Cheap Operations. [[IJCV 2022]](https://arxiv.org/abs/2201.03297) | [./g_ghost_pytorch](https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/g_ghost_pytorch) | [MindSpore Model Zoo](https://gitee.com/mindspore/models/tree/master/research/cv/ghostnet_d) |
| TinyNet | Model Rubik’s Cube: Twisting Resolution, Depth and Width for TinyNets. [[NeurIPS 2020]](https://arxiv.org/abs/2010.14819) | [./tinynet_pytorch](https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/tinynet_pytorch) | [MindSpore Model Zoo](https://gitee.com/mindspore/models/tree/master/research/cv/tinynet) |
| TNT | Transformer in Transformer. [[NeurIPS 2021]](https://arxiv.org/abs/2103.00112) | [./tnt_pytorch](https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/tnt_pytorch) | [MindSpore Model Zoo](https://gitee.com/mindspore/models/tree/master/research/cv/TNT) |
| PyramidTNT | PyramidTNT: Improved Transformer-in-Transformer Baselines with Pyramid Architecture. [[CVPR 2022 Workshop]](https://arxiv.org/abs/2201.00978)| [./tnt_pytorch](https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/tnt_pytorch) | [MindSpore Model Zoo](https://gitee.com/mindspore/models/tree/master/research/cv/TNT) |
| CMT | CMT: Convolutional Neural Networks Meet Vision Transformers. [[CVPR 2022]](https://arxiv.org/pdf/2107.06263.pdf) | [./cmt_pytorch](https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/cmt_pytorch) | [MindSpore Model Zoo](https://gitee.com/mindspore/models/tree/master/research/cv/CMT) |
| AugViT | Augmented Shortcuts for Vision Transformers. [[NeurIPS 2021]](https://proceedings.neurips.cc/paper/2021/file/818f4654ed39a1c147d1e51a00ffb4cb-Paper.pdf) | [./augvit_pytorch](https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/augvit_pytorch) | [MindSpore Model Zoo](https://gitee.com/mindspore/models/tree/master/research/cv/augvit) |
| SNN-MLP | Brain-inspired Multilayer Perceptron with Spiking Neurons. [[CVPR 2022]](https://arxiv.org/pdf/2203.14679.pdf) | [./snnmlp_pytorch](https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/snnmlp_pytorch) | [MindSpore Model Zoo](https://gitee.com/mindspore/models/tree/master/research/cv/snn_mlp) |
| WaveMLP | An Image Patch is a Wave: Quantum Inspired Vision MLP. [[CVPR 2022]](https://arxiv.org/pdf/2111.12294.pdf) | [./wavemlp_pytorch](https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/wavemlp_pytorch) | [MindSpore Model Zoo](https://gitee.com/mindspore/models/tree/master/research/cv/wave_mlp) |
| ViG | Vision GNN: An Image is Worth Graph of Nodes. [[NeurIPS 2022]](https://arxiv.org/abs/2206.00272) | [./vig_pytorch](https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/vig_pytorch) | - | [MindSpore Model Zoo](https://gitee.com/mindspore/models/tree/master/research/cv/ViG) |
| LegoNet | LegoNet: Efficient Convolutional Neural Networks with Lego Filters. [[ICML 2019]](http://proceedings.mlr.press/v97/yang19c/yang19c.pdf) | [./legonet_pytorch](https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/legonet_pytorch) | - |
| Versatile Filters | Learning Versatile Filters for Efficient Convolutional Neural Networks. [[NeurIPS 2018]](https://papers.nips.cc/paper/7433-learning-versatile-filters-for-efficient-convolutional-neural-networks) | [./versatile_filters](https://github.com/huawei-noah/CV-backbones/tree/master/versatile_filters) | - |
| ParameterNet | ParameterNet: Parameters Are All You Need. [[CVPR 2024]](https://arxiv.org/abs/2306.14525). | [./parameternet_pytorch](https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/parameternet_pytorch) | - |



