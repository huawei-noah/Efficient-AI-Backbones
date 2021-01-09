# 2021.01.09-Changed for main script for testing TinyNet on ImageNet
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
""" 
An implementation of TinyNet

Requirements: timm==0.1.20
"""
from timm.models.efficientnet_builder import *
from timm.models.efficientnet import EfficientNet, EfficientNetFeatures, _cfg
from timm.models.registry import register_model


def _gen_tinynet(variant_cfg, channel_multiplier=1.0, depth_multiplier=1.0, depth_trunc='round', **kwargs):
    """Creates a TinyNet model.
    """
    arch_def = [
        ['ds_r1_k3_s1_e1_c16_se0.25'], ['ir_r2_k3_s2_e6_c24_se0.25'],
        ['ir_r2_k5_s2_e6_c40_se0.25'], ['ir_r3_k3_s2_e6_c80_se0.25'],
        ['ir_r3_k5_s1_e6_c112_se0.25'], ['ir_r4_k5_s2_e6_c192_se0.25'],
        ['ir_r1_k3_s1_e6_c320_se0.25'],
    ]
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier, depth_trunc=depth_trunc),
        num_features=max(1280, round_channels(1280, channel_multiplier, 8, None)),
        stem_size=32,
        fix_stem=True,
        channel_multiplier=channel_multiplier,
        act_layer=Swish,
        norm_kwargs=resolve_bn_args(kwargs),
        **kwargs,
    ) 
    model = EfficientNet(**model_kwargs)
    model.default_cfg = variant_cfg
    return model


@register_model
def tinynet(r=1.0, w=1.0, d=1.0, **kwargs):
    """ TinyNet """
    hw = int(224 * r)
    model = _gen_tinynet(
        _cfg(input_size=(3, hw, hw)), channel_multiplier=w, depth_multiplier=d, **kwargs)
    return model
