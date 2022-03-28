# 2022.02.14-Changed for main script for augvit model
#            Huawei Technologies Co., Ltd. <foss@huawei.com>

""" Vision Transformer (ViT) in PyTorch
A PyTorch implement of Vision Transformers as described in
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale' - https://arxiv.org/abs/2010.11929
The official jax code is released and available at https://github.com/google-research/vision_transformer
Status/TODO:
* Models updated to be compatible with official impl. Args added to support backward compat for old PyTorch weights.
* Weights ported from official jax impl for 384x384 base and small models, 16x16 and 32x32 patches.
* Trained (supervised on ImageNet-1k) my custom 'small' patch model to 77.9, 'base' to 79.4 top-1 with this code.
* Hopefully find time and GPUs for SSL or unsupervised pretraining on OpenImages w/ ImageNet fine-tune in future.
Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert
Hacked together by / Copyright 2020 Ross Wightman
"""

import time
import torch
import torch.nn as nn
from torch.nn import init
from functools import partial
import numpy as np
from math import sqrt

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.resnet import resnet26d, resnet50d
from timm.models.registry import register_model

import math

from scipy import linalg
import torch.nn.functional as F



def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    # patch models
    'vit_small_patch16_224': _cfg(),
    'vit_base_patch16_224': _cfg(
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'vit_base_patch16_384': _cfg(
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_base_patch32_384': _cfg(
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_large_patch16_224': _cfg(
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_large_patch16_384': _cfg(
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_large_patch32_384': _cfg(
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_huge_patch16_224': _cfg(),
    'vit_huge_patch32_384': _cfg(input_size=(3, 384, 384)),
}


class Circulant_Linear(nn.Module): 
    def __init__(self,in_features, out_features, bias=True,num_cir=4):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.num_cir=num_cir
       
        self.min_features=min(in_features,out_features)
        self.num_cir_in=self.num_cir*in_features//self.min_features 
        self.num_cir_out=self.num_cir*out_features//self.min_features
        #print('mode:',mode,'self.num_cir_out',self.num_cir_out,'self.num_cir_in',self.num_cir_in)
        
        assert self.min_features%self.num_cir==0
        self.fea_block=self.min_features//self.num_cir
        
        self.weight_list=nn.Parameter(torch.Tensor(self.num_cir_out*self.num_cir_in*self.fea_block))
        
        index_matrix=torch.from_numpy(linalg.circulant(range(self.fea_block))).long()
        
        index_list= [[] for iblock in range(self.num_cir_out)]   
        for iblock in range(self.num_cir_out): 
            for jblock in range(self.num_cir_in):
                 index_list[iblock].append(index_matrix+(iblock*self.num_cir_in+jblock)*self.fea_block)
        for iblock in range(self.num_cir_out):
             index_list[iblock]=torch.cat(index_list[iblock],dim=1)
        #print('weight_list',weight_list)
        self.register_buffer('index_matrix',torch.cat(index_list,dim=0))        

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight_list.view(self.num_cir_out,self.num_cir_in,self.fea_block), a=math.sqrt(5))
        if self.bias is not None:
            fan_in=self.in_features
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
            
    def forward(self, input):
        weight=self.weight_list[self.index_matrix]        
        return F.linear(input, weight, self.bias)

  
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


   
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,num_cir=4,mode='linear'):
        super().__init__()
        
        self.mode=mode
        
        self.norm1 = norm_layer(dim)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
        if self.mode in ['linear']: #### work well 
            self.augs_attn=nn.Linear(dim,dim, bias=True)
            self.augs=nn.Linear(dim,dim, bias=True)
        elif self.mode in ['circulant']:
            self.augs_attn=Circulant_Linear(in_features=dim, out_features=dim, bias=True,num_cir=num_cir)
            self.augs=Circulant_Linear(in_features=dim, out_features=dim, bias=True,num_cir=num_cir) 
        elif self.mode in ['linear2']:
            self.augs_attn=nn.Sequential(nn.Linear(dim,dim, bias=True),act_layer()) 
            self.augs=nn.Sequential(nn.Linear(dim,dim, bias=True),act_layer())  
            
            self.augs_attn2=nn.Sequential(nn.Linear(dim,dim, bias=True),act_layer()) 
            self.augs2=nn.Sequential(nn.Linear(dim,dim, bias=True),act_layer()) 
            
        elif self.mode in ['circulant2']:
            self.augs_attn=nn.Sequential(Circulant_Linear(in_features=dim, out_features=dim, 
                       bias=True,num_cir=num_cir),act_layer()) 
            self.augs=nn.Sequential(Circulant_Linear(in_features=dim, out_features=dim,
                       bias=True,num_cir=num_cir),act_layer())  
            
            self.augs_attn2=nn.Sequential(Circulant_Linear(in_features=dim, out_features=dim, 
                       bias=True,num_cir=num_cir),act_layer()) 
            self.augs2=nn.Sequential(Circulant_Linear(in_features=dim, out_features=dim,
                       bias=True,num_cir=num_cir),act_layer()) 
           
    def forward(self, x):

        if self.mode in['base']:
            x = x + self.drop_path(self.attn(self.norm1(x))) 
            x_norm2=self.norm2(x)
            x = x + self.drop_path(self.mlp(x_norm2)) 
        elif self.mode in['linear','circulant']: 
            x_norm1=self.norm1(x)
            x = x + self.drop_path(self.attn(x_norm1))+ self.augs_attn(x_norm1) 
            x_norm2=self.norm2(x)
            x = x + self.drop_path(self.mlp(x_norm2)) + self.augs(x_norm2) 
        elif self.mode in['linear2','circulant2']: 
            x_norm1=self.norm1(x)
            x = (x + self.drop_path(self.attn(x_norm1))+ self.augs_attn(x_norm1) +self.augs_attn2(x_norm1))*3.0/4 
            x_norm2=self.norm2(x)
            x = (x + self.drop_path(self.mlp(x_norm2)) + self.augs(x_norm2) + self.augs2(x_norm2))*3.0/4     
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """
    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
                # map for all networks, the feature metadata has reliable channel and stride info, but using
                # stride to calc feature dim requires info about padding of each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Linear(feature_dim, embed_dim)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm,num_cir=4,mode='linear'):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,num_cir=num_cir,
                mode=mode)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # NOTE as per official impl, we could have a pre-logits representation dense layer + tanh here
        #self.repr = nn.Linear(embed_dim, representation_size)
        #self.repr_act = nn.Tanh()

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict


@register_model 
def augvit_small(pretrained=False, **kwargs): 
    
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),mode='base',num_cir=4, **kwargs) #mode,num_cir
    model.default_cfg = default_cfgs['vit_base_patch16_224']
    return model

@register_model 
def augvit_small_linear(pretrained=False, **kwargs): 
    
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),mode='linear',num_cir=4, **kwargs) #mode,num_cir
    model.default_cfg = default_cfgs['vit_base_patch16_224']
    return model

@register_model 
def augvit_small_circulant2(pretrained=False, **kwargs): 
    
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),mode='circulant2',num_cir=4, **kwargs) #mode,num_cir
    model.default_cfg = default_cfgs['vit_base_patch16_224']
    return model

@register_model 
def augvit_small_linear2(pretrained=False, **kwargs): 
    
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),mode='linear2',num_cir=4, **kwargs) #mode,num_cir
    model.default_cfg = default_cfgs['vit_base_patch16_224']
    return model

@register_model 
def augvit_base_circulant2(pretrained=False, **kwargs): 
    
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),mode='circulant2',num_cir=4, **kwargs) #mode,num_cir
    model.default_cfg = default_cfgs['vit_base_patch16_224']
    return model
