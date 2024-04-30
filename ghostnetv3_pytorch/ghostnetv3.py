# 2020.06.09-Changed for building GhostNet
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
"""
Creates a GhostNet Model as defined in:
GhostNet: More Features from Cheap Operations By Kai Han, Yunhe Wang, Qi Tian, Jianyuan Guo, Chunjing Xu, Chang Xu.
https://arxiv.org/abs/1911.11907
Modified from https://github.com/d-li14/mobilenetv3.pytorch and https://github.com/rwightman/pytorch-image-models
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from typing import Optional, List, Tuple
from timm.models.registry import register_model

#__all__ = ['ghost_net']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.


class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x    

    
class ConvBnAct(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size,
                 stride=1, act_layer=nn.ReLU):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size, stride, kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_chs)
        self.act1 = act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x
    
    
def gcd(a,b):
    if a<b:
        a,b=b,a
    while(a%b != 0):
        c = a%b
        a=b
        b=c
    return b
    
def MyNorm(dim):
    return nn.GroupNorm(1, dim)  

class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True,mode=None,args=None):
        super(GhostModule, self).__init__()
        #self.args=args
        self.mode = mode
        self.gate_loc = 'before'
        
        self.inter_mode = 'nearest'
        self.scale = 1.0
        
        self.infer_mode = False
        self.num_conv_branches = 3
        self.dconv_scale = True
        self.gate_fn = nn.Sigmoid()

        # if args.gate_fn=='hard_sigmoid':
        #     self.gate_fn=hard_sigmoid
        # elif args.gate_fn=='sigmoid': 
        #     self.gate_fn=nn.Sigmoid()
        # elif args.gate_fn=='relu': 
        #     self.gate_fn=nn.ReLU()
        # elif args.gate_fn=='clip': 
        #     self.gate_fn=myclip 
        # elif args.gate_fn=='tanh': 
        #     self.gate_fn=nn.Tanh()

        if self.mode in ['ori']:
            self.oup = oup
            init_channels = math.ceil(oup / ratio) 
            new_channels = init_channels*(ratio-1)
            if self.infer_mode:  
                self.primary_conv = nn.Sequential(  
                    nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
                    nn.BatchNorm2d(init_channels),
                    nn.ReLU(inplace=True) if relu else nn.Sequential(),
                )
                self.cheap_operation = nn.Sequential(
                    nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
                    nn.BatchNorm2d(new_channels),
                    nn.ReLU(inplace=True) if relu else nn.Sequential(),
                )
            else:
                self.primary_rpr_skip = nn.BatchNorm2d(inp) \
                    if inp == init_channels and stride == 1 else None
                primary_rpr_conv = list()
                for _ in range(self.num_conv_branches):
                    primary_rpr_conv.append(self._conv_bn(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False))
                self.primary_rpr_conv = nn.ModuleList(primary_rpr_conv)
                # Re-parameterizable scale branch
                self.primary_rpr_scale = None
                if kernel_size > 1:
                    self.primary_rpr_scale = self._conv_bn(inp, init_channels, 1, 1, 0, bias=False)
                self.primary_activation = nn.ReLU(inplace=True) if relu else None


                self.cheap_rpr_skip = nn.BatchNorm2d(init_channels) \
                    if init_channels == new_channels else None
                cheap_rpr_conv = list()
                for _ in range(self.num_conv_branches):
                    cheap_rpr_conv.append(self._conv_bn(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False))
                self.cheap_rpr_conv = nn.ModuleList(cheap_rpr_conv)
                # Re-parameterizable scale branch
                self.cheap_rpr_scale = None
                if dw_size > 1:
                    self.cheap_rpr_scale = self._conv_bn(init_channels, new_channels, 1, 1, 0, groups=init_channels, bias=False)
                self.cheap_activation = nn.ReLU(inplace=True) if relu else None
                self.in_channels = init_channels
                self.groups = init_channels
                self.kernel_size = dw_size
     
        elif self.mode in ['ori_shortcut_mul_conv15']: 
            self.oup = oup
            init_channels = math.ceil(oup / ratio) 
            new_channels = init_channels*(ratio-1)
            self.short_conv = nn.Sequential( 
                nn.Conv2d(inp, oup, kernel_size, stride, kernel_size//2, bias=False),
                nn.BatchNorm2d(oup),
                nn.Conv2d(oup, oup, kernel_size=(1,5), stride=1, padding=(0,2), groups=oup,bias=False),
                nn.BatchNorm2d(oup),
                nn.Conv2d(oup, oup, kernel_size=(5,1), stride=1, padding=(2,0), groups=oup,bias=False),
                nn.BatchNorm2d(oup),
            )
            if self.infer_mode:
                self.primary_conv = nn.Sequential(  
                    nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
                    nn.BatchNorm2d(init_channels),
                    nn.ReLU(inplace=True) if relu else nn.Sequential(),
                )
                self.cheap_operation = nn.Sequential(
                    nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
                    nn.BatchNorm2d(new_channels),
                    nn.ReLU(inplace=True) if relu else nn.Sequential(),
                ) 
            else:
                self.primary_rpr_skip = nn.BatchNorm2d(inp) \
                    if inp == init_channels and stride == 1 else None
                primary_rpr_conv = list()
                for _ in range(self.num_conv_branches):
                    primary_rpr_conv.append(self._conv_bn(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False))
                self.primary_rpr_conv = nn.ModuleList(primary_rpr_conv)
                # Re-parameterizable scale branch
                self.primary_rpr_scale = None
                if kernel_size > 1:
                    self.primary_rpr_scale = self._conv_bn(inp, init_channels, 1, 1, 0, bias=False)
                self.primary_activation = nn.ReLU(inplace=True) if relu else None


                self.cheap_rpr_skip = nn.BatchNorm2d(init_channels) \
                    if init_channels == new_channels else None
                cheap_rpr_conv = list()
                for _ in range(self.num_conv_branches):
                    cheap_rpr_conv.append(self._conv_bn(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False))
                self.cheap_rpr_conv = nn.ModuleList(cheap_rpr_conv)
                # Re-parameterizable scale branch
                self.cheap_rpr_scale = None
                if dw_size > 1:
                    self.cheap_rpr_scale = self._conv_bn(init_channels, new_channels, 1, 1, 0, groups=init_channels, bias=False)
                self.cheap_activation = nn.ReLU(inplace=True) if relu else None
                self.in_channels = init_channels
                self.groups = init_channels
                self.kernel_size = dw_size

      
    def forward(self, x):
        if self.mode in ['ori']:
            if self.infer_mode:
                x1 = self.primary_conv(x)
                x2 = self.cheap_operation(x1)
            else:
                identity_out = 0
                if self.primary_rpr_skip is not None:
                    identity_out = self.primary_rpr_skip(x)
                scale_out = 0
                if self.primary_rpr_scale is not None and self.dconv_scale:
                    scale_out = self.primary_rpr_scale(x)
                x1 = scale_out + identity_out
                for ix in range(self.num_conv_branches):
                    x1 += self.primary_rpr_conv[ix](x)
                if self.primary_activation is not None:
                    x1 = self.primary_activation(x1)

                cheap_identity_out = 0
                if self.cheap_rpr_skip is not None:
                    cheap_identity_out = self.cheap_rpr_skip(x1)
                cheap_scale_out = 0
                if self.cheap_rpr_scale is not None and self.dconv_scale:
                    cheap_scale_out = self.cheap_rpr_scale(x1)
                x2 = cheap_scale_out + cheap_identity_out
                for ix in range(self.num_conv_branches):
                    x2 += self.cheap_rpr_conv[ix](x1)
                if self.cheap_activation is not None:
                    x2 = self.cheap_activation(x2)

            out = torch.cat([x1,x2], dim=1)
            return out

        elif self.mode in ['ori_shortcut_mul_conv15']:  
            res=self.short_conv(F.avg_pool2d(x,kernel_size=2,stride=2))
            
            if self.infer_mode:
                x1 = self.primary_conv(x)
                x2 = self.cheap_operation(x1)
            else:
                identity_out = 0
                if self.primary_rpr_skip is not None:
                    identity_out = self.primary_rpr_skip(x)
                scale_out = 0
                if self.primary_rpr_scale is not None and self.dconv_scale:
                    scale_out = self.primary_rpr_scale(x)
                x1 = scale_out + identity_out
                for ix in range(self.num_conv_branches):
                    x1 += self.primary_rpr_conv[ix](x)
                if self.primary_activation is not None:
                    x1 = self.primary_activation(x1)

                cheap_identity_out = 0
                if self.cheap_rpr_skip is not None:
                    cheap_identity_out = self.cheap_rpr_skip(x1)
                cheap_scale_out = 0
                if self.cheap_rpr_scale is not None and self.dconv_scale:
                    cheap_scale_out = self.cheap_rpr_scale(x1)
                x2 = cheap_scale_out + cheap_identity_out
                for ix in range(self.num_conv_branches):
                    x2 += self.cheap_rpr_conv[ix](x1)
                if self.cheap_activation is not None:
                    x2 = self.cheap_activation(x2)

            out = torch.cat([x1,x2], dim=1)

            if self.gate_loc=='before':
                return out[:,:self.oup,:,:]*F.interpolate(self.gate_fn(res/self.scale),size=out.shape[-2:],mode=self.inter_mode) # 'nearest'
#                 return out*F.interpolate(self.gate_fn(res/self.scale),size=out.shape[-1].item(),mode=self.inter_mode) # 'nearest'
            else:
                return out[:,:self.oup,:,:]*self.gate_fn(F.interpolate(res,size=out.shape[-2:],mode=self.inter_mode))  
#                 return out*self.gate_fn(F.interpolate(res,size=out.shape[-1],mode=self.inter_mode))  


    def reparameterize(self):
        """ Following works like `RepVGG: Making VGG-style ConvNets Great Again` -
        https://arxiv.org/pdf/2101.03697.pdf. We re-parameterize multi-branched
        architecture used at training time to obtain a plain CNN-like structure
        for inference.
        """
        if self.infer_mode:
            return
        primary_kernel, primary_bias = self._get_kernel_bias_primary()
        self.primary_conv = nn.Conv2d(in_channels=self.primary_rpr_conv[0].conv.in_channels,
                                      out_channels=self.primary_rpr_conv[0].conv.out_channels,
                                      kernel_size=self.primary_rpr_conv[0].conv.kernel_size,
                                      stride=self.primary_rpr_conv[0].conv.stride,
                                      padding=self.primary_rpr_conv[0].conv.padding,
                                      dilation=self.primary_rpr_conv[0].conv.dilation,
                                      groups=self.primary_rpr_conv[0].conv.groups,
                                      bias=True)
        self.primary_conv.weight.data = primary_kernel
        self.primary_conv.bias.data = primary_bias
        self.primary_conv = nn.Sequential(
            self.primary_conv, 
            self.primary_activation if self.primary_activation is not None else nn.Sequential()
        )

        cheap_kernel, cheap_bias = self._get_kernel_bias_cheap()
        self.cheap_operation = nn.Conv2d(in_channels=self.cheap_rpr_conv[0].conv.in_channels,
                                      out_channels=self.cheap_rpr_conv[0].conv.out_channels,
                                      kernel_size=self.cheap_rpr_conv[0].conv.kernel_size,
                                      stride=self.cheap_rpr_conv[0].conv.stride,
                                      padding=self.cheap_rpr_conv[0].conv.padding,
                                      dilation=self.cheap_rpr_conv[0].conv.dilation,
                                      groups=self.cheap_rpr_conv[0].conv.groups,
                                      bias=True)
        self.cheap_operation.weight.data = cheap_kernel
        self.cheap_operation.bias.data = cheap_bias

        self.cheap_operation = nn.Sequential(
            self.cheap_operation, 
            self.cheap_activation if self.cheap_activation is not None else nn.Sequential()
        )

        # Delete un-used branches
        for para in self.parameters():
            para.detach_()
        if hasattr(self, 'primary_rpr_conv'):
            self.__delattr__('primary_rpr_conv')
        if hasattr(self, 'primary_rpr_scale'):
            self.__delattr__('primary_rpr_scale')
        if hasattr(self, 'primary_rpr_skip'):
            self.__delattr__('primary_rpr_skip')

        if hasattr(self, 'cheap_rpr_conv'):
            self.__delattr__('cheap_rpr_conv')
        if hasattr(self, 'cheap_rpr_scale'):
            self.__delattr__('cheap_rpr_scale')
        if hasattr(self, 'cheap_rpr_skip'):
            self.__delattr__('cheap_rpr_skip')

        self.infer_mode = True

    def _get_kernel_bias_primary(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Method to obtain re-parameterized kernel and bias.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L83

        :return: Tuple of (kernel, bias) after fusing branches.
        """
        # get weights and bias of scale branch
        kernel_scale = 0
        bias_scale = 0
        if self.primary_rpr_scale is not None:
            kernel_scale, bias_scale = self._fuse_bn_tensor(self.primary_rpr_scale)
            # Pad scale branch kernel to match conv branch kernel size.
            pad = self.kernel_size // 2
            kernel_scale = torch.nn.functional.pad(kernel_scale,
                                                   [pad, pad, pad, pad])

        # get weights and bias of skip branch
        kernel_identity = 0
        bias_identity = 0
        if self.primary_rpr_skip is not None:
            kernel_identity, bias_identity = self._fuse_bn_tensor(self.primary_rpr_skip)

        # get weights and bias of conv branches
        kernel_conv = 0
        bias_conv = 0
        for ix in range(self.num_conv_branches):
            _kernel, _bias = self._fuse_bn_tensor(self.primary_rpr_conv[ix])
            kernel_conv += _kernel
            bias_conv += _bias

        kernel_final = kernel_conv + kernel_scale + kernel_identity
        bias_final = bias_conv + bias_scale + bias_identity
        return kernel_final, bias_final
    
    def _get_kernel_bias_cheap(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Method to obtain re-parameterized kernel and bias.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L83

        :return: Tuple of (kernel, bias) after fusing branches.
        """
        # get weights and bias of scale branch
        kernel_scale = 0
        bias_scale = 0
        if self.cheap_rpr_scale is not None:
            kernel_scale, bias_scale = self._fuse_bn_tensor(self.cheap_rpr_scale)
            # Pad scale branch kernel to match conv branch kernel size.
            pad = self.kernel_size // 2
            kernel_scale = torch.nn.functional.pad(kernel_scale,
                                                   [pad, pad, pad, pad])

        # get weights and bias of skip branch
        kernel_identity = 0
        bias_identity = 0
        if self.cheap_rpr_skip is not None:
            kernel_identity, bias_identity = self._fuse_bn_tensor(self.cheap_rpr_skip)

        # get weights and bias of conv branches
        kernel_conv = 0
        bias_conv = 0
        for ix in range(self.num_conv_branches):
            _kernel, _bias = self._fuse_bn_tensor(self.cheap_rpr_conv[ix])
            kernel_conv += _kernel
            bias_conv += _bias

        kernel_final = kernel_conv + kernel_scale + kernel_identity
        bias_final = bias_conv + bias_scale + bias_identity
        return kernel_final, bias_final

    def _fuse_bn_tensor(self, branch) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Method to fuse batchnorm layer with preceeding conv layer.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L95

        :param branch:
        :return: Tuple of (kernel, bias) after fusing batchnorm.
        """
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = torch.zeros((self.in_channels,
                                            input_dim,
                                            self.kernel_size,
                                            self.kernel_size),
                                           dtype=branch.weight.dtype,
                                           device=branch.weight.device)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim,
                                 self.kernel_size // 2,
                                 self.kernel_size // 2] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def _conv_bn(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, bias=False):
        """ Helper method to construct conv-batchnorm layers.

        :param kernel_size: Size of the convolution kernel.
        :param padding: Zero-padding size.
        :return: Conv-BN module.
        """
        mod_list = nn.Sequential()
        mod_list.add_module('conv', nn.Conv2d(in_channels=in_channels,
                                              out_channels=out_channels,
                                              kernel_size=kernel_size,
                                              stride=stride,
                                              padding=padding,
                                              groups=groups,
                                              bias=bias))
        mod_list.add_module('bn', nn.BatchNorm2d(out_channels))
        return mod_list


class GhostBottleneck(nn.Module): 
    """ Ghost bottleneck w/ optional SE"""

    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3,
                 stride=1, act_layer=nn.ReLU, se_ratio=0.,layer_id=None,args=None):
        super(GhostBottleneck, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.
        self.stride = stride

        self.num_conv_branches = 3
        self.infer_mode = False
        self.dconv_scale = True

        # Point-wise expansion
        if layer_id<=1:
            self.ghost1 = GhostModule(in_chs, mid_chs, relu=True,mode='ori',args=args)
        else:
            self.ghost1 = GhostModule(in_chs, mid_chs, relu=True,mode='ori_shortcut_mul_conv15',args=args) ####这里是扩张 mid_chs远大于in_chs

        # Depth-wise convolution
        if self.stride > 1:
            if self.infer_mode:
                self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride,
                                 padding=(dw_kernel_size-1)//2,
                                 groups=mid_chs, bias=False)
                self.bn_dw = nn.BatchNorm2d(mid_chs)
            else:
                self.dw_rpr_skip = nn.BatchNorm2d(mid_chs) if stride == 1 else None
                dw_rpr_conv = list()
                for _ in range(self.num_conv_branches):
                    dw_rpr_conv.append(self._conv_bn(mid_chs, mid_chs, dw_kernel_size, stride, (dw_kernel_size-1)//2, groups=mid_chs, bias=False))
                self.dw_rpr_conv = nn.ModuleList(dw_rpr_conv)
                # Re-parameterizable scale branch
                self.dw_rpr_scale = None
                if dw_kernel_size > 1:
                    self.dw_rpr_scale = self._conv_bn(mid_chs, mid_chs, 1, 2, 0, groups=mid_chs, bias=False)
                self.kernel_size = dw_kernel_size
                self.in_channels = mid_chs

        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None

        # Point-wise linear projection
        if layer_id<=1:
            self.ghost2 = GhostModule(mid_chs, out_chs, relu=False,mode='ori',args=args)
        else:
            self.ghost2 = GhostModule(mid_chs, out_chs, relu=False,mode='ori',args=args)
        
        # shortcut
        if (in_chs == out_chs and self.stride == 1):
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                       padding=(dw_kernel_size-1)//2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )

    def forward(self, x):
        residual = x

        # 1st ghost bottleneck
        x = self.ghost1(x)

        # Depth-wise convolution
        if self.stride > 1:
            if self.infer_mode:
                x = self.conv_dw(x)
                x = self.bn_dw(x)
            else:
                dw_identity_out = 0
                if self.dw_rpr_skip is not None:
                    dw_identity_out = self.dw_rpr_skip(x)
                dw_scale_out = 0
                if self.dw_rpr_scale is not None and self.dconv_scale:
                    dw_scale_out = self.dw_rpr_scale(x)
                x1 = dw_scale_out + dw_identity_out
                for ix in range(self.num_conv_branches):
                    x1 += self.dw_rpr_conv[ix](x)
                x = x1

        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)

        # 2nd ghost bottleneck
        x = self.ghost2(x)
        
        x += self.shortcut(residual)
        return x

    def _conv_bn(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, bias=False):
        """ Helper method to construct conv-batchnorm layers.

        :param kernel_size: Size of the convolution kernel.
        :param padding: Zero-padding size.
        :return: Conv-BN module.
        """
        mod_list = nn.Sequential()
        mod_list.add_module('conv', nn.Conv2d(in_channels=in_channels,
                                              out_channels=out_channels,
                                              kernel_size=kernel_size,
                                              stride=stride,
                                              padding=padding,
                                              groups=groups,
                                              bias=bias))
        mod_list.add_module('bn', nn.BatchNorm2d(out_channels))
        return mod_list


    def reparameterize(self):
        """ Following works like `RepVGG: Making VGG-style ConvNets Great Again` -
        https://arxiv.org/pdf/2101.03697.pdf. We re-parameterize multi-branched
        architecture used at training time to obtain a plain CNN-like structure
        for inference.
        """
        if self.infer_mode or self.stride == 1:
            return
        dw_kernel, dw_bias = self._get_kernel_bias_dw()
        self.conv_dw = nn.Conv2d(in_channels=self.dw_rpr_conv[0].conv.in_channels,
                                      out_channels=self.dw_rpr_conv[0].conv.out_channels,
                                      kernel_size=self.dw_rpr_conv[0].conv.kernel_size,
                                      stride=self.dw_rpr_conv[0].conv.stride,
                                      padding=self.dw_rpr_conv[0].conv.padding,
                                      dilation=self.dw_rpr_conv[0].conv.dilation,
                                      groups=self.dw_rpr_conv[0].conv.groups,
                                      bias=True)
        self.conv_dw.weight.data = dw_kernel
        self.conv_dw.bias.data = dw_bias
        self.bn_dw = nn.Identity()

        # Delete un-used branches
        for para in self.parameters():
            para.detach_()
        if hasattr(self, 'dw_rpr_conv'):
            self.__delattr__('dw_rpr_conv')
        if hasattr(self, 'dw_rpr_scale'):
            self.__delattr__('dw_rpr_scale')
        if hasattr(self, 'dw_rpr_skip'):
            self.__delattr__('dw_rpr_skip')

        self.infer_mode = True

    def _get_kernel_bias_dw(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Method to obtain re-parameterized kernel and bias.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L83

        :return: Tuple of (kernel, bias) after fusing branches.
        """
        # get weights and bias of scale branch
        kernel_scale = 0
        bias_scale = 0
        if self.dw_rpr_scale is not None:
            kernel_scale, bias_scale = self._fuse_bn_tensor(self.dw_rpr_scale)
            # Pad scale branch kernel to match conv branch kernel size.
            pad = self.kernel_size // 2
            kernel_scale = torch.nn.functional.pad(kernel_scale,
                                                   [pad, pad, pad, pad])

        # get weights and bias of skip branch
        kernel_identity = 0
        bias_identity = 0
        if self.dw_rpr_skip is not None:
            kernel_identity, bias_identity = self._fuse_bn_tensor(self.dw_rpr_skip)

        # get weights and bias of conv branches
        kernel_conv = 0
        bias_conv = 0
        for ix in range(self.num_conv_branches):
            _kernel, _bias = self._fuse_bn_tensor(self.dw_rpr_conv[ix])
            kernel_conv += _kernel
            bias_conv += _bias

        kernel_final = kernel_conv + kernel_scale + kernel_identity
        bias_final = bias_conv + bias_scale + bias_identity
        return kernel_final, bias_final


    def _fuse_bn_tensor(self, branch) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Method to fuse batchnorm layer with preceeding conv layer.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L95

        :param branch:
        :return: Tuple of (kernel, bias) after fusing batchnorm.
        """
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = torch.zeros((self.in_channels,
                                            input_dim,
                                            self.kernel_size,
                                            self.kernel_size),
                                           dtype=branch.weight.dtype,
                                           device=branch.weight.device)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim,
                                 self.kernel_size // 2,
                                 self.kernel_size // 2] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

class GhostNet(nn.Module):
    def __init__(self, cfgs, num_classes=1000, width=1.0, dropout=0.2, block=GhostBottleneck, args=None):
        super(GhostNet, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        self.dropout = dropout

        # building first layer
        output_channel = _make_divisible(16 * width, 4)
        self.conv_stem = nn.Conv2d(3, output_channel, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.act1 = nn.ReLU(inplace=True)
        input_channel = output_channel

        # building inverted residual blocks
        stages = []
        #block = block
        layer_id=0
        for cfg in self.cfgs:
            layers = []
            for k, exp_size, c, se_ratio, s in cfg:
                
                output_channel = _make_divisible(c * width, 4)
                hidden_channel = _make_divisible(exp_size * width, 4)
                if block==GhostBottleneck:
                    layers.append(block(input_channel, hidden_channel, output_channel, k, s,
                                  se_ratio=se_ratio,layer_id=layer_id,args=args))
                input_channel = output_channel
                layer_id+=1
            stages.append(nn.Sequential(*layers))

        output_channel = _make_divisible(exp_size * width, 4)
        stages.append(nn.Sequential(ConvBnAct(input_channel, output_channel, 1)))
        input_channel = output_channel
        
        self.blocks = nn.Sequential(*stages)        

        # building last several layers
        output_channel = 1280
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_head = nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=True)
        self.act2 = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(output_channel, num_classes)

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act2(x)
        x = x.view(x.size(0), -1)
        # if self.dropout > 0.:
            # x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier(x)
        x = x.squeeze()
        return x

    def reparameterize(self):
        for _, module in self.named_modules():
            if isinstance(module, GhostModule):
                module.reparameterize()
            if isinstance(module, GhostBottleneck):
                module.reparameterize()

@register_model
def ghostnetv3(**kwargs):
    """
    Constructs a GhostNet model
    """
    cfgs = [
        # k, t, c, SE, s 
        # stage1
        [[3,  16,  16, 0, 1]],
        # stage2
        [[3,  48,  24, 0, 2]],
        [[3,  72,  24, 0, 1]],
        # stage3
        [[5,  72,  40, 0.25, 2]],
        [[5, 120,  40, 0.25, 1]],
        # stage4
        [[3, 240,  80, 0, 2]],
        [[3, 200,  80, 0, 1],
         [3, 184,  80, 0, 1],
         [3, 184,  80, 0, 1],
         [3, 480, 112, 0.25, 1],
         [3, 672, 112, 0.25, 1]
        ],
        # stage5
        [[5, 672, 160, 0.25, 2]],
        [[5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1],
         [5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1]
        ]
    ]
    return GhostNet(cfgs, num_classes=1000, width=kwargs['width'], dropout=0.2)

if __name__=='__main__':
    model = ghostnetv3(width=1.0)
    model.eval()
    print(model)
    input1 = torch.randn(32,3,320,256)
    input2 = torch.randn(32,3,256,320)
    input3 = torch.randn(32,3,224,224)

    with torch.inference_mode():
        y11 = model(input1)
        y12 = model(input2)
        y13 = model(input3)
    model.reparameterize()
    print(model)
    with torch.inference_mode():
        y21 = model(input1)
        y22 = model(input2)
        y23 = model(input3)
    print(torch.allclose(y11, y21), torch.norm(y11 - y21))
    print(torch.allclose(y12, y22), torch.norm(y12 - y22))
    print(torch.allclose(y13, y23), torch.norm(y13 - y23))
