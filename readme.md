## GhostNet
The code provides the inference code and pretrained model of GhostNet on ImageNet.

`myconv2d.py` implemented `GhostModule` and `ghost_net.py` implemented `GhostNet`.

### Usage
Run `python test-ghostnet.py --eval --data_dir=/path/to/imagenet/dir/ --load=./models/ghostnet_checkpoint` to evaluate on `val` set.
You'll get the accuracy: top-1 error=`0.26066`, top-5 error=`0.08614`.

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

### Requirements
The code was verified on Python3.6, TF-1.13.1, Tensorpack-0.9.7. Not sure on other version.

### Citation
@article{ghostnet,
  title={GhostNet: More Features from Cheap Operations},
  author={Han, Kai and Wang, Yunhe and Tian, Qi and Guo, Jianyuan and Xu, Chunjing and Xu, Chang},
  journal={arXiv preprint},
  year={2019}
}
