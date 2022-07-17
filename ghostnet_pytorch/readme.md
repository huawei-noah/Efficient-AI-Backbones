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

# G-GhostNet

`g_ghost_regnet.py` implemented `G-Ghost RegNet` in the extended [IJCV paper](https://arxiv.org/abs/2201.03297).

The checkpoint is available at [BaiduDisk](https://pan.baidu.com/s/1bgdM9xWVCFGMyYMKPUn4Zg) and Password: `on8i`.

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
