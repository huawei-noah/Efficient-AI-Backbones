import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['g_ghost_regnetx_002', 'g_ghost_regnetx_004', 'g_ghost_regnetx_006', 'g_ghost_regnetx_008', 'g_ghost_regnetx_016', 'g_ghost_regnetx_032',
           'g_ghost_regnetx_040', 'g_ghost_regnetx_064', 'g_ghost_regnetx_080', 'g_ghost_regnetx_120', 'g_ghost_regnetx_160', 'g_ghost_regnetx_320']

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, group_width=1,
                 dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = planes * self.expansion
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, width // min(width, group_width), dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes)
        self.bn3 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class Stage(nn.Module):
    def __init__(self, block, inplanes, planes, group_width, blocks, stride=1, dilate=False, cheap_ratio=0.5):
        super(Stage, self).__init__()
        norm_layer = nn.BatchNorm2d
        downsample = None
        self.dilation = 1
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                norm_layer(planes),
            )
            
        self.base = block(inplanes, planes, stride, downsample, group_width,
                            self.dilation, norm_layer)
        self.end = block(planes, planes, group_width=group_width,
                                dilation=self.dilation,
                                norm_layer=norm_layer)
        
        group_width = int(group_width * 0.75)
        raw_planes = int(planes * (1 - cheap_ratio) / group_width) * group_width
        cheap_planes = planes - raw_planes
        self.cheap_planes = cheap_planes
        self.raw_planes = raw_planes
        
        self.merge = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(planes+raw_planes*(blocks-2), cheap_planes,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(cheap_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(cheap_planes, cheap_planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(cheap_planes),
        )
        self.cheap = nn.Sequential(
            nn.Conv2d(cheap_planes, cheap_planes,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(cheap_planes),
        )
        self.cheap_relu = nn.ReLU(inplace=True)
        
        layers = []
        downsample = nn.Sequential(
            LambdaLayer(lambda x: x[:, :raw_planes])
        )

        layers.append(block(raw_planes, raw_planes, 1, downsample, group_width,
                            self.dilation, norm_layer))
        inplanes = raw_planes
        for _ in range(2, blocks-1):
            layers.append(block(inplanes, raw_planes, group_width=group_width,
                                dilation=self.dilation,
                                norm_layer=norm_layer))

        self.layers = nn.Sequential(*layers)
        
    def forward(self, input):
        x0 = self.base(input)
        
        m_list = [x0]
        e = x0[:, :self.raw_planes]        
        for l in self.layers:
            e = l(e)
            m_list.append(e)
        m = torch.cat(m_list, 1)
        m = self.merge(m)
        
        c = x0[:, self.raw_planes:]
        c = self.cheap_relu(self.cheap(c) + m)
        
        x = torch.cat((e, c), 1)
        x = self.end(x)
        return x

class GGhostRegNet(nn.Module):
    def __init__(self, block, layers, widths, num_classes=1000, zero_init_residual=True,
                 group_width=1, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(GGhostRegNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 32
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False, False]
        if len(replace_stride_with_dilation) != 4:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 4-element tuple, got {}".format(replace_stride_with_dilation))
        self.group_width = group_width
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(block, widths[0], layers[0], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        
        self.inplanes = widths[0]
        if layers[1] > 2:
            self.layer2 = Stage(block, self.inplanes, widths[1], group_width, layers[1], stride=2,
                          dilate=replace_stride_with_dilation[1], cheap_ratio=0.5) 
        else:      
            self.layer2 = self._make_layer(block, widths[1], layers[1], stride=2,
                                           dilate=replace_stride_with_dilation[1])
        
        self.inplanes = widths[1]
        self.layer3 = Stage(block, self.inplanes, widths[2], group_width, layers[2], stride=2,
                      dilate=replace_stride_with_dilation[2], cheap_ratio=0.5)
        
        self.inplanes = widths[2]
        if layers[3] > 2:
            self.layer4 = Stage(block, self.inplanes, widths[3], group_width, layers[3], stride=2,
                          dilate=replace_stride_with_dilation[3], cheap_ratio=0.5) 
        else:
            self.layer4 = self._make_layer(block, widths[3], layers[3], stride=2,
                                           dilate=replace_stride_with_dilation[3])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(self.inplanes * block.expansion, num_classes)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.group_width,
                            self.dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, group_width=self.group_width,
                                dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x

def g_ghost_regnetx_002(pretrained=False, num_classes=1000, **kwargs):
    model = GGhostRegNet(Bottleneck, [1, 1, 1, 1], [32, 64, 128, 256], num_classes=num_classes, **kwargs)
    return model

def g_ghost_regnetx_004(pretrained=False, num_classes=1000, **kwargs):
    model = GGhostRegNet(Bottleneck, [1, 1, 2, 2], [32, 64, 128, 256], num_classes=num_classes, **kwargs)
    return model

def g_ghost_regnetx_006(pretrained=False, num_classes=1000, **kwargs):
    model = GGhostRegNet(Bottleneck, [1, 1, 3, 3], [32, 64, 128, 256], num_classes=num_classes, **kwargs)
    return model

def g_ghost_regnetx_008(pretrained=False, num_classes=1000, **kwargs):
    model = GGhostRegNet(Bottleneck, [1, 1, 4, 4], [32, 64, 128, 256], num_classes=num_classes, **kwargs)
    return model

def g_ghost_regnetx_016(pretrained=False, num_classes=1000, **kwargs):
    model = GGhostRegNet(Bottleneck, [2, 2, 4, 4], [64, 128, 256, 512], num_classes=num_classes, **kwargs)
    return model

def g_ghost_regnetx_032(pretrained=False, num_classes=1000, **kwargs):
    model = GGhostRegNet(Bottleneck, [2, 2, 6, 6], [64, 128, 256, 512], num_classes=num_classes, **kwargs)
    return model

def g_ghost_regnetx_040(pretrained=False, num_classes=1000, **kwargs):
    model = GGhostRegNet(Bottleneck, [3, 4, 6, 6], [64, 128, 256, 512], num_classes=num_classes, **kwargs)
    return model

def g_ghost_regnetx_064(pretrained=False, num_classes=1000, **kwargs):
    model = GGhostRegNet(Bottleneck, [3, 4, 8, 8], [64, 128, 256, 512], num_classes=num_classes, **kwargs)
    return model

def g_ghost_regnetx_080(pretrained=False, num_classes=1000, **kwargs):
    model = GGhostRegNet(Bottleneck, [3, 4, 12, 12], [64, 128, 256, 512], num_classes=num_classes, **kwargs)
    return model

def g_ghost_regnetx_120(pretrained=False, num_classes=1000, **kwargs):
    model = GGhostRegNet(Bottleneck, [4, 6, 12, 12], [128, 256, 512, 1024], num_classes=num_classes, **kwargs)
    return model

def g_ghost_regnetx_160(pretrained=False, num_classes=1000, **kwargs):
    model = GGhostRegNet(Bottleneck, [4, 6, 16, 16], [128, 256, 512, 1024], num_classes=num_classes, **kwargs)
    return model

def g_ghost_regnetx_320(pretrained=False, num_classes=1000, **kwargs):
    model = GGhostRegNet(Bottleneck, [4, 6, 24, 24], [128, 256, 512, 1024], num_classes=num_classes, **kwargs)
    return model



# Add functions for other configurations like g_ghost_regnetx_004, g_ghost_regnetx_006, etc.

