# TinyNet

Model Rubik’s Cube: Twisting Resolution, Depth and Width for TinyNets. NeurIPS 2020. [[arXiv]](https://arxiv.org/abs/2010.14819)

By Kai Han, Yunhe Wang, Qiulin Zhang, Wei Zhang, Chunjing Xu, Tong Zhang.

TinyNets are a series of lightweight models obtained by twisting resolution, depth and width with a data-driven tiny formula. TinyNet outperforms EfficientNet and MobileNetV3.

|model|FLOPs|Top-1|
|-|-|-|
|TinyNet-A|339M|76.8|
|TinyNet-B|202M|75.0|
|TinyNet-C|100M|71.2|
|TinyNet-D|52M|67.0|
|TinyNet-E|24M|59.9|


## Requirements

PyTorch 1.3, timm 0.1.20

## Usage

This repo contains pytorch code of TinyNet.

`Python eval.py /path/to/imagenet/val/ --model_name=tinynet_a`

`Python eval.py /path/to/imagenet/val/ --model_name=tinynet_b`

`Python eval.py /path/to/imagenet/val/ --model_name=tinynet_c`

`Python eval.py /path/to/imagenet/val/ --model_name=tinynet_d`

`Python eval.py /path/to/imagenet/val/ --model_name=tinynet_e`

The mindspore code has been also released: [[MindSpore code]](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/cv/tinynet). 

## Citation
```
@inproceedings{tinynet,
  title={Model Rubik’s Cube: Twisting Resolution, Depth and Width for TinyNets},
  author={Han, Kai and Wang, Yunhe and Zhang, Qiulin and Zhang, Wei and Xu, Chunjing and Zhang, Tong},
  booktitle={NeurIPS},
  year={2020}
}
```
