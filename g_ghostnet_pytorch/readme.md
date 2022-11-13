# G-GhostNet

GhostNets on Heterogeneous Devices via Cheap Operations. IJCV 2022. [[arXiv]](https://arxiv.org/abs/2201.03297)

By Kai Han, Yunhe Wang, Chang Xu, Jianyuan Guo, Chunjing Xu, Enhua Wu and Qi Tian.

- **Approach**

<div align="center">
   <img src="../fig/g-ghost.png" width="720">
</div>

## Requirements
The code was verified on Python3.6, PyTorch-1.3+.

## Data Preparation
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

## G-GhostNet Code

`g_ghost_regnet.py` implemented `G-Ghost RegNet` in the extended [IJCV paper](https://arxiv.org/abs/2201.03297).

The checkpoint is available at [BaiduDisk](https://pan.baidu.com/s/1bgdM9xWVCFGMyYMKPUn4Zg) and Password: `on8i`.

## Citation
```
@article{ghostnet2022,
  title={GhostNets on Heterogeneous Devices via Cheap Operations},
  author={Kai Han, Yunhe Wang, Chang Xu, Jianyuan Guo, Chunjing Xu, Enhua Wu and Qi Tian},
  journal={IJCV},
  year={2022}
}
```
