# Transformer in Transformer (TNT) and PyramidTNT
By Kai Han, An Xiao, Enhua Wu, Jianyuan Guo, Chunjing Xu, Yunhe Wang. NeurIPS 2021. [[arXiv link]](https://arxiv.org/abs/2103.00112)

![image](https://user-images.githubusercontent.com/9500784/122160150-ff1bca80-cea1-11eb-9329-be5031bad78e.png)

## Requirements
Pytorch 1.7.0,
timm 0.3.2,
apex

## TNT Code
[Paper: Transformer in Transformer (TNT)](https://arxiv.org/abs/2103.00112)

- Training example for 8 GPUs:
```
python -m torch.distributed.launch --nproc_per_node=8 train.py /path/to/imagenet/ --model tnt_s_patch16_224 --sched cosine --epochs 300 --opt adamw -j 8 --warmup-lr 1e-6 --mixup .8 --cutmix 1.0 --model-ema --model-ema-decay 0.99996 --aa rand-m9-mstd0.5-inc1 --color-jitter 0.4 --warmup-epochs 5 --opt-eps 1e-8 --repeated-aug --remode pixel --reprob 0.25 --amp --lr 1e-3 --weight-decay .05 --drop 0 --drop-path .1 -b 128 --output /path/to/save/models/
```

- Pretrained models

|Model|Params (M)|FLOPs (B)|Top-1|Top-5|URL|
|-|-|-|-|-|-|
|TNT-S|23.8|5.2|81.5|95.7|[[BaiduDisk]](https://pan.baidu.com/s/1AwJDWEPl-hqLHfUvqmlqxQ), Password: 7ndi|
|TNT-B|65.6|14.1|82.9|96.3|[[BaiduDisk]](https://pan.baidu.com/s/1_TemN7kvWuYeZohisObQ1w), Password: 2gb7|

- Evaluate example:
```
python train.py /path/to/imagenet/ --model tnt_s_patch16_224 -b 256 --pretrain_path /path/to/pretrained/model/ --evaluate
```

## PyramidTNT Code
[Paper: PyramidTNT](https://arxiv.org/abs/2201.00978)

- Training example for 8 GPUs:
```
python -m torch.distributed.launch --nproc_per_node=8 train.py /path/to/imagenet/ --model ptnt_s_patch16_256 --sched cosine --epochs 300 --opt adamw -j 8 --warmup-lr 1e-6 --mixup .8 --cutmix 1.0 --model-ema --model-ema-decay 0.99996 --aa rand-m9-mstd0.5-inc1 --color-jitter 0.4 --warmup-epochs 20 --opt-eps 1e-8 --repeated-aug --remode pixel --reprob 0.25 --amp --lr 1e-3 --weight-decay .05 --drop 0 --drop-path .1 -b 128 --output /path/to/save/models/
```

- Pretrained models

|Model|Params (M)|FLOPs (B)|Top-1|URL|
|-|-|-|-|-|
|PyramidTNT-Ti|10.6|0.6|75.2|[[BaiduDisk]](https://pan.baidu.com/s/1xm3DSGcAJbvFQm4jmDAlnQ), Password: 0r5t|
|PyramidTNT-S|32.0|3.3|82.0|[[BaiduDisk]](https://pan.baidu.com/s/1xha8x3DTlPq9-KeC6EPPow), Password: v5w5|
|PyramidTNT-M|85.0|8.2|83.5|[[BaiduDisk]](https://pan.baidu.com/s/1B9zkWkrUAETuiyr08ClY-w), Password: jqm3|
|PyramidTNT-B|157.0|16.0|84.1|[[BaiduDisk]](https://pan.baidu.com/s/1tT0mxRrZQx6facYsdPBt0g), Password: ns4t|

- Evaluate example:
```
python train.py /path/to/imagenet/ --model ptnt_s_patch16_256 -b 256 --pretrain_path /path/to/pretrained/model/ --evaluate
```


## Citation
```
@inproceedings{tnt,
  title={Transformer in transformer},
  author={Han, Kai and Xiao, An and Wu, Enhua and Guo, Jianyuan and Xu, Chunjing and Wang, Yunhe},
  booktitle={NeurIPS},
  year={2021}
}
@misc{pyramidtnt,
  title={PyramidTNT: Improved Transformer-in-Transformer Baselines with Pyramid Architecture}, 
  author={Kai Han and Jianyuan Guo and Yehui Tang and Yunhe Wang},
  year={2022},
  eprint={2201.00978},
  archivePrefix={arXiv}
}
```

## Third-party implementations
1. Pytorch (timm) with ImageNet pretrained models: https://www.github.com/rwightman/pytorch-image-models/tree/master/timm/models/tnt.py
2. Pytorch (mmclassification) with ImageNet pretrained models: https://github.com/open-mmlab/mmclassification/blob/master/docs/model_zoo.md
3. JAX/FLAX: https://github.com/NZ99/transformer_in_transformer_flax
4. MindSpore Code: https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/cv/TNT and pretrained weights on Oxford-IIIT Pets dataset: https://www.mindspore.cn/resources/hub/details?noah-cvlab/gpu/1.1/tnt_v1.0_oxford_pets
