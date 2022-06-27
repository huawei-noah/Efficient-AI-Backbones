# CMT.pytorch


## Implementation of [CMT: Convolutional Neural Networks Meet Vision Transformers](https://arxiv.org/pdf/2107.06263.pdf)

### Set up
```
- python==3.6
- cuda==10.0

# other pytorch/timm version can also work

pip install torch==1.7.0 torchvision==0.8.1;
pip install timm==0.3.2;
pip install torchprofile;

# build apex

cd /your_path_to/apex-master/;
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

### Data preparation

Download and extract ImageNet train and val images from http://image-net.org/.
The directory structure is:

```
│path/to/imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```

#### Training

To train CMT-Tiny on ImageNet-1K on a single node with 8 gpus:

```
python -m torch.distributed.launch --nproc_per_node=8 train.py --data-path /your_path_to/imagenet/ --output_dir /your_path_to/output/ --model cmt_ti --batch-size 256 --apex-amp --input-size 160 --weight-decay 0.05 --drop-path 0.1 --epochs 800 --test_freq 100 --test_epoch 760 --warmup-lr 1e-7 --warmup-epochs 20 --lr 8e-4 --min-lr 1e-5 --no-model-ema
```

To train CMT-XS on ImageNet-1K on a single node with 8 gpus:
```
python -m torch.distributed.launch --nproc_per_node=8 train.py --data-path /your_path_to/imagenet/ --output_dir /your_path_to/output/ --model cmt_xs --batch-size 256 --apex-amp --input-size 192 --weight-decay 0.04 --drop-path 0.08 --epochs 400 --test_freq 100 --test_epoch 360 --warmup-lr 1e-6 --warmup-epochs 20 --lr 7e-4 --min-lr 2e-5 --model-ema-decay 0.9998
```

To train CMT-Small on ImageNet-1K on a single node with 8 gpus:

```
python -m torch.distributed.launch --nproc_per_node=8 train.py --data-path /your_path_to/imagenet/ --output_dir /your_path_to/output/ --model cmt_s --batch-size 128 --apex-amp --input-size 224 --weight-decay 0.05 --drop-path 0.1 --epochs 300 --test_freq 100 --test_epoch 260 --warmup-lr 1e-7 --warmup-epochs 20
```

To train CMT-Base on ImageNet-1K on a single node with 8 gpus:

```
python -m torch.distributed.launch --nproc_per_node=8 train.py --data-path /your_path_to/imagenet/ --output_dir /your_path_to/output/ --model cmt_b --batch-size 64 --apex-amp --input-size 256 --weight-decay 0.05 --drop-path 0.25 --epochs 300 --test_freq 100 --test_epoch 260 --warmup-lr 1e-6 --min-lr 2e-5 --warmup-epochs 20
```

## Citation

If you find this project useful in your research, please consider cite:

```bibtex
@article{guo2021cmt,
  title={Cmt: Convolutional neural networks meet vision transformers},
  author={Guo, Jianyuan and Han, Kai and Wu, Han and Xu, Chang and Tang, Yehui and Xu, Chunjing and Wang, Yunhe},
  journal={arXiv preprint arXiv:2107.06263},
  year={2021}
}
```

## Acknowledgment

This repo is based on [DeiT](https://github.com/facebookresearch/deit) and [pytorch-image-models](https://github.com/rwightman/pytorch-image-models).
