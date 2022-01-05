# 2022.01.04-Changed for PyramidTNT model
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
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.resnet import resnet26d, resnet50d
from timm.models.registry import register_model


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
    'tnt_ti_patch16_192': _cfg(
        input_size=(3, 192, 192), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'tnt_s_patch16_256': _cfg(
        input_size=(3, 256, 256), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'tnt_m_patch16_256': _cfg(
        input_size=(3, 256, 256), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'tnt_b_patch16_256': _cfg(
        input_size=(3, 256, 256), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
}


def make_divisible(v, divisor=8, min_value=None):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


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


class SE(nn.Module):
    def __init__(self, dim, hidden_ratio=None):
        super().__init__()
        hidden_ratio = hidden_ratio or 1
        self.dim = dim
        hidden_dim = int(dim * hidden_ratio)
        self.fc = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim),
            nn.Tanh()
        )

    def forward(self, x):
        a = x.mean(dim=1, keepdim=True) # B, 1, C
        a = self.fc(a)
        x = a * x
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.pool = nn.AvgPool2d(sr_ratio, stride=sr_ratio)
            self.linear = nn.Linear(dim, dim)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W, relative_pos=None):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.pool(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(self.linear(x_))
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if relative_pos is not None:
            attn += relative_pos
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """ TNT Block
    """
    def __init__(self, outer_dim, inner_dim, outer_head, inner_head, num_words, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, se=0, sr_ratio=1):
        super().__init__()
        self.has_inner = inner_dim > 0
        if self.has_inner:
            # Inner
            self.inner_norm1 = norm_layer(num_words * inner_dim)
            self.inner_attn = Attention(
                inner_dim, num_heads=inner_head, qkv_bias=qkv_bias,
                qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
            self.inner_norm2 = norm_layer(num_words * inner_dim)
            self.inner_mlp = Mlp(in_features=inner_dim, hidden_features=int(inner_dim * mlp_ratio),
                                 out_features=inner_dim, act_layer=act_layer, drop=drop)

            self.proj_norm1 = norm_layer(num_words * inner_dim)
            self.proj = nn.Linear(num_words * inner_dim, outer_dim, bias=False)
            self.proj_norm2 = norm_layer(outer_dim)
        # Outer
        self.outer_norm1 = norm_layer(outer_dim)
        self.outer_attn = Attention(
            outer_dim, num_heads=outer_head, qkv_bias=qkv_bias,
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.outer_norm2 = norm_layer(outer_dim)
        self.outer_mlp = Mlp(in_features=outer_dim, hidden_features=int(outer_dim * mlp_ratio),
                             out_features=outer_dim, act_layer=act_layer, drop=drop)
        # SE
        self.se = se
        self.se_layer = None
        if self.se > 0:
            self.se_layer = SE(outer_dim, 0.25)

    def forward(self, x, outer_tokens, H_out, W_out, H_in, W_in, relative_pos):
        B, N, C = outer_tokens.size()
        if self.has_inner:
            x = x + self.drop_path(self.inner_attn(self.inner_norm1(x.reshape(B, N, -1)).reshape(B*N, H_in*W_in, -1), H_in, W_in)) # B*N, k*k, c
            x = x + self.drop_path(self.inner_mlp(self.inner_norm2(x.reshape(B, N, -1)).reshape(B*N, H_in*W_in, -1))) # B*N, k*k, c
            outer_tokens = outer_tokens + self.proj_norm2(self.proj(self.proj_norm1(x.reshape(B, N, -1)))) # B, N, C
        if self.se > 0:
            outer_tokens = outer_tokens + self.drop_path(self.outer_attn(self.outer_norm1(outer_tokens), H_out, W_out, relative_pos))
            tmp_ = self.outer_mlp(self.outer_norm2(outer_tokens))
            outer_tokens = outer_tokens + self.drop_path(tmp_ + self.se_layer(tmp_))
        else:
            outer_tokens = outer_tokens + self.drop_path(self.outer_attn(self.outer_norm1(outer_tokens), H_out, W_out, relative_pos))
            outer_tokens = outer_tokens + self.drop_path(self.outer_mlp(self.outer_norm2(outer_tokens)))
        return x, outer_tokens


class SentenceAggregation(nn.Module):
    """ Sentence Aggregation
    """
    def __init__(self, dim_in, dim_out, stride=2, act_layer=nn.GELU):
        super().__init__()
        self.stride = stride
        self.norm = nn.LayerNorm(dim_in)
        self.conv = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=2*stride-1, padding=stride-1, stride=stride),
        )
        
    def forward(self, x, H, W):
        B, N, C = x.shape # B, N, C
        x = self.norm(x)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.conv(x)
        H, W = math.ceil(H / self.stride), math.ceil(W / self.stride)
        x = x.reshape(B, -1, H * W).transpose(1, 2)
        return x, H, W


class WordAggregation(nn.Module):
    """ Word Aggregation
    """
    def __init__(self, dim_in, dim_out, stride=2, act_layer=nn.GELU):
        super().__init__()
        self.stride = stride
        self.dim_out = dim_out
        self.norm = nn.LayerNorm(dim_in)
        self.conv = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=2*stride-1, padding=stride-1, stride=stride),
        )

    def forward(self, x, H_out, W_out, H_in, W_in):
        B_N, M, C = x.shape # B*N, M, C
        x = self.norm(x)
        x = x.reshape(-1, H_out, W_out, H_in, W_in, C)
        
        # padding to fit (1333, 800) in detection.
        pad_input = (H_out % 2 == 1) or (W_out % 2 == 1)
        if pad_input:
            x = F.pad(x.permute(0, 3, 4, 5, 1, 2), (0, W_out % 2, 0, H_out % 2))
            x = x.permute(0, 4, 5, 1, 2, 3)            
        # patch merge
        x1 = x[:, 0::2, 0::2, :, :, :]  # B, H/2, W/2, H_in, W_in, C
        x2 = x[:, 1::2, 0::2, :, :, :]
        x3 = x[:, 0::2, 1::2, :, :, :]
        x4 = x[:, 1::2, 1::2, :, :, :]
        x = torch.cat([torch.cat([x1, x2], 3), torch.cat([x3, x4], 3)], 4) # B, H/2, W/2, 2*H_in, 2*W_in, C
        x = x.reshape(-1, 2*H_in, 2*W_in, C).permute(0, 3, 1, 2) # B_N/4, C, 2*H_in, 2*W_in
        x = self.conv(x)  # B_N/4, C, H_in, W_in
        x = x.reshape(-1, self.dim_out, M).transpose(1, 2)
        return x


class Stem(nn.Module):
    """ Image to Visual Word Embedding
    Overlap: https://arxiv.org/pdf/2106.13797.pdf
    """
    def __init__(self, img_size=224, in_chans=3, outer_dim=768, inner_dim=24):
        super().__init__()
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.inner_dim = inner_dim
        self.num_patches = img_size[0] // 8 * img_size[1] // 8
        self.num_words = 16
        
        self.common_conv = nn.Sequential(
            nn.Conv2d(in_chans, inner_dim*2, 3, stride=2, padding=1),
            nn.BatchNorm2d(inner_dim*2),
            nn.ReLU(inplace=True),
        )
        self.inner_convs = nn.Sequential(
            nn.Conv2d(inner_dim*2, inner_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(inner_dim),
            nn.ReLU(inplace=False),
        )
        self.outer_convs = nn.Sequential(
            nn.Conv2d(inner_dim*2, inner_dim*4, 3, stride=2, padding=1),
            nn.BatchNorm2d(inner_dim*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_dim*4, inner_dim*8, 3, stride=2, padding=1),
            nn.BatchNorm2d(inner_dim*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_dim*8, outer_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(outer_dim),
            nn.ReLU(inplace=False),
        )
        
        self.unfold = nn.Unfold(kernel_size=4, padding=0, stride=4)

    def forward(self, x):
        B, C, H, W = x.shape
        H_out, W_out = H // 8, W // 8
        H_in, W_in = 4, 4
        x = self.common_conv(x)
        # inner_tokens
        inner_tokens = self.inner_convs(x) # B, C, H, W
        inner_tokens = self.unfold(inner_tokens).transpose(1, 2) # B, N, Ck2
        inner_tokens = inner_tokens.reshape(B * H_out * W_out, self.inner_dim, H_in*W_in).transpose(1, 2) # B*N, C, 4*4
        # outer_tokens
        outer_tokens = self.outer_convs(x) # B, C, H_out, W_out
        outer_tokens = outer_tokens.permute(0, 2, 3, 1).reshape(B, H_out * W_out, -1)
        return inner_tokens, outer_tokens, (H_out, W_out), (H_in, W_in)


class Stage(nn.Module):
    """ PyramidTNT stage
    """
    def __init__(self, num_blocks, outer_dim, inner_dim, outer_head, inner_head, num_patches, num_words, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, se=0, sr_ratio=1):
        super().__init__()
        blocks = []
        drop_path = drop_path if isinstance(drop_path, list) else [drop_path] * num_blocks
        
        for j in range(num_blocks):
            if j == 0:
                _inner_dim = inner_dim
            elif j == 1 and num_blocks > 6:
                _inner_dim = inner_dim
            else:
                _inner_dim = -1
            blocks.append(Block(
                outer_dim, _inner_dim, outer_head=outer_head, inner_head=inner_head,
                num_words=num_words, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop,
                attn_drop=attn_drop, drop_path=drop_path[j], act_layer=act_layer, norm_layer=norm_layer,
                se=se, sr_ratio=sr_ratio))

        self.blocks = nn.ModuleList(blocks)
        self.relative_pos = nn.Parameter(torch.randn(
                        1, outer_head, num_patches, num_patches // sr_ratio // sr_ratio))

    def forward(self, inner_tokens, outer_tokens, H_out, W_out, H_in, W_in):
        for blk in self.blocks:
            inner_tokens, outer_tokens = blk(inner_tokens, outer_tokens, H_out, W_out, H_in, W_in, self.relative_pos)
        return inner_tokens, outer_tokens


class PyramidTNT(nn.Module):
    """ PyramidTNT (Transformer in Transformer) for computer vision
    """
    def __init__(self, configs=None, img_size=224, in_chans=3, num_classes=1000, mlp_ratio=4., qkv_bias=False,
                qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, se=0):
        super().__init__()
        self.num_classes = num_classes
        depths = configs['depths']
        outer_dims = configs['outer_dims']
        inner_dims = configs['inner_dims']
        outer_heads = configs['outer_heads']
        inner_heads = configs['inner_heads']
        sr_ratios = [4, 2, 1, 1]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule 
        self.num_features = outer_dims[-1]  # num_features for consistency with other models       

        self.patch_embed = Stem(
            img_size=img_size, in_chans=in_chans, outer_dim=outer_dims[0], inner_dim=inner_dims[0])
        num_patches = self.patch_embed.num_patches
        num_words = self.patch_embed.num_words
        
        self.outer_pos = nn.Parameter(torch.zeros(1, num_patches, outer_dims[0]))
        self.inner_pos = nn.Parameter(torch.zeros(1, num_words, inner_dims[0]))
        self.pos_drop = nn.Dropout(p=drop_rate)

        depth = 0
        self.word_merges = nn.ModuleList([])
        self.sentence_merges = nn.ModuleList([])
        self.stages = nn.ModuleList([])
        for i in range(4):
            if i > 0:
                self.word_merges.append(WordAggregation(inner_dims[i-1], inner_dims[i], stride=2))
                self.sentence_merges.append(SentenceAggregation(outer_dims[i-1], outer_dims[i], stride=2))
            self.stages.append(Stage(depths[i], outer_dim=outer_dims[i], inner_dim=inner_dims[i],
                        outer_head=outer_heads[i], inner_head=inner_heads[i],
                        num_patches=num_patches // (2 ** i) // (2 ** i), num_words=num_words, mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[depth:depth+depths[i]], norm_layer=norm_layer, se=se, sr_ratio=sr_ratios[i])
            )
            depth += depths[i]
        
        self.norm = norm_layer(outer_dims[-1])

        # NOTE as per official impl, we could have a pre-logits representation dense layer + tanh here
        # self.repr = nn.Linear(outer_dim, representation_size)
        # self.repr_act = nn.Tanh()

        # Classifier head
        self.head = nn.Linear(outer_dims[-1], num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.outer_pos, std=.02)
        trunc_normal_(self.inner_pos, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'outer_pos', 'inner_pos'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        inner_tokens, outer_tokens, (H_out, W_out), (H_in, W_in) = self.patch_embed(x)
        inner_tokens = inner_tokens + self.inner_pos # B*N, 8*8, C
        outer_tokens = outer_tokens + self.pos_drop(self.outer_pos)  # B, N, D
        
        for i in range(4):
            if i > 0:
                inner_tokens = self.word_merges[i-1](inner_tokens, H_out, W_out, H_in, W_in)
                outer_tokens, H_out, W_out = self.sentence_merges[i-1](outer_tokens, H_out, W_out)
            inner_tokens, outer_tokens = self.stages[i](inner_tokens, outer_tokens, H_out, W_out, H_in, W_in)
        
        outer_tokens = self.norm(outer_tokens)
        return outer_tokens.mean(dim=1)

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
def ptnt_ti_patch16_192(pretrained=False, **kwargs):
    outer_dim = 80
    inner_dim = 5
    outer_head = 2
    inner_head = 1
    configs = {
        'depths': [2, 6, 3, 2],
        'outer_dims': [outer_dim, outer_dim*2, outer_dim*4, outer_dim*4],
        'inner_dims': [inner_dim, inner_dim*2, inner_dim*4, inner_dim*4],
        'outer_heads': [outer_head, outer_head*2, outer_head*4, outer_head*4],
        'inner_heads': [inner_head, inner_head*2, inner_head*4, inner_head*4],
    }
    
    model = PyramidTNT(configs=configs, img_size=192, qkv_bias=False, **kwargs)
    model.default_cfg = default_cfgs['tnt_ti_patch16_192']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter)
    return model


@register_model
def ptnt_s_patch16_256(pretrained=False, **kwargs):
    outer_dim = 128
    inner_dim = 8
    outer_head = 4
    inner_head = 2    
    configs = {
        'depths': [2, 8, 4, 2],
        'outer_dims': [outer_dim, outer_dim*2, outer_dim*4, outer_dim*4],
        'inner_dims': [inner_dim, inner_dim*2, inner_dim*4, inner_dim*4],
        'outer_heads': [outer_head, outer_head*2, outer_head*4, outer_head*4],
        'inner_heads': [inner_head, inner_head*2, inner_head*4, inner_head*4],
    }
    
    model = PyramidTNT(configs=configs, img_size=256, qkv_bias=False, **kwargs)
    model.default_cfg = default_cfgs['tnt_s_patch16_256']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter)
    return model


@register_model
def ptnt_m_patch16_256(pretrained=False, **kwargs):
    outer_dim = 192
    inner_dim = 12
    outer_head = 4
    inner_head = 2    
    configs = {
        'depths': [2, 8, 6, 2],
        'outer_dims': [outer_dim, outer_dim*2, outer_dim*4, outer_dim*4],
        'inner_dims': [inner_dim, inner_dim*2, inner_dim*4, inner_dim*4],
        'outer_heads': [outer_head, outer_head*2, outer_head*4, outer_head*4],
        'inner_heads': [inner_head, inner_head*2, inner_head*4, inner_head*4],
    }
    
    model = PyramidTNT(configs=configs, img_size=256, qkv_bias=False, **kwargs)
    model.default_cfg = default_cfgs['tnt_m_patch16_256']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter)
    return model


@register_model
def ptnt_b_patch16_256(pretrained=False, **kwargs):
    outer_dim = 256
    inner_dim = 16
    outer_head = 4
    inner_head = 2    
    configs = {
        'depths': [2, 10, 6, 2],
        'outer_dims': [outer_dim, outer_dim*2, outer_dim*4, outer_dim*4],
        'inner_dims': [inner_dim, inner_dim*2, inner_dim*4, inner_dim*4],
        'outer_heads': [outer_head, outer_head*2, outer_head*4, outer_head*4],
        'inner_heads': [inner_head, inner_head*2, inner_head*4, inner_head*4],
    }
    
    model = PyramidTNT(configs=configs, img_size=256, qkv_bias=False, **kwargs)
    model.default_cfg = default_cfgs['tnt_b_patch16_256']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter)
    return model
