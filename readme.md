# GhostNet

GhostNet: More Features from Cheap Operations [[arXiv]](https://arxiv.org/abs/1911.11907)

By Kai Han, Yunhe Wang, Qi Tian, Jianyuan Guo, Chunjing Xu, Chang Xu.

- **Approach**

<div align="center">
   <img src="./fig/ghost_module.png" width="720">
</div>

- **Performance**

We beat other SOTA lightweight CNNs such as **MobileNetV3** and **FBNet**.

<div align="center">
   <img src="./fig/flops_latency.png" width="720">
</div>

## Implementation

The code provides the TensorFlow code and pretrained model of GhostNet on ImageNet.

`myconv2d.py` implemented `GhostModule` and `ghost_net.py` implemented `GhostNet`.

### Requirements
The code was verified on Python3.6, TensorFlow-1.13.1, Tensorpack-0.9.7. Not sure on other version.

### Usage
Run `python test-ghostnet.py --eval --data_dir=/path/to/imagenet/dir/ --load=./models/ghostnet_checkpoint` to evaluate on `val` set.

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
@article{ghostnet,
  title={GhostNet: More Features from Cheap Operations},
  author={Han, Kai and Wang, Yunhe and Tian, Qi and Guo, Jianyuan and Xu, Chunjing and Xu, Chang},
  journal={arXiv},
  year={2019}
}
```
