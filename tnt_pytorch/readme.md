# Transformer in Transformer (TNT)
By Kai Han, An Xiao, Enhua Wu, Jianyuan Guo, Chunjing Xu, Yunhe Wang. [[arXiv]](https://arxiv.org/abs/2103.00112)

## Requirments
Pytorch 1.7+
timm 0.3.2

## Code
Train:
```
python train.py
```

The pretrained models will be released as soon.

## Citation
```
@misc{han2021transformer,
      title={Transformer in Transformer}, 
      author={Kai Han and An Xiao and Enhua Wu and Jianyuan Guo and Chunjing Xu and Yunhe Wang},
      year={2021},
      eprint={2103.00112},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Third-party implementations
1. Pytorch with **ImageNet pretrained models**: https://www.github.com/rwightman/pytorch-image-models/tree/master/timm/models/tnt.py
2. JAX/FLAX: https://github.com/NZ99/transformer_in_transformer_flax
3. MindSpore Code: https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/cv/TNT and pretrained weights on Oxford-IIIT Pets dataset: https://www.mindspore.cn/resources/hub/details?noah-cvlab/gpu/1.1/tnt_v1.0_oxford_pets