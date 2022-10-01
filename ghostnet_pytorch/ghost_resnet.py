# 2022.09.30-Changed for building Ghost-ResNet
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
"""
Creates a Ghost-ResNet Model as defined in:
GhostNet: More Features from Cheap Operations By Kai Han, Yunhe Wang, Qi Tian, Jianyuan Guo, Chunjing Xu, Chang Xu.
https://arxiv.org/abs/1911.11907
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


class GhostModule(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, dw_size=3, ratio=2, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(GhostModule, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.weight = None
        self.ratio = ratio
        self.dw_size = dw_size
        self.dw_dilation = (dw_size - 1) // 2
        self.init_channels = math.ceil(out_channels / ratio)
        self.new_channels = self.init_channels * (ratio - 1)
        
        self.conv1 = nn.Conv2d(self.in_channels, self.init_channels, kernel_size, self.stride, padding=self.padding)
        self.conv2 = nn.Conv2d(self.init_channels, self.new_channels, self.dw_size, 1, padding=int(self.dw_size/2), groups=self.init_channels)
        
        
        self.weight1 = nn.Parameter(torch.Tensor(self.init_channels, self.in_channels, kernel_size, kernel_size))
        self.bn1 = nn.BatchNorm2d(self.init_channels)
        if self.new_channels > 0:
            self.weight2 = nn.Parameter(torch.Tensor(self.new_channels, 1, self.dw_size, self.dw_size))
            self.bn2 = nn.BatchNorm2d(self.out_channels - self.init_channels)
        
        if bias:
            self.bias =nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_custome_parameters()
    
    def reset_custome_parameters(self):
        nn.init.kaiming_uniform_(self.weight1, a=math.sqrt(5))
        if self.new_channels > 0:
            nn.init.kaiming_uniform_(self.weight2, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
    
    def forward(self, input): 
        x1 = self.conv1(input)
        if self.new_channels == 0:
            return x1
        x2 = self.conv2(x1)
        x2 = x2[:, :self.out_channels - self.init_channels, :, :]
        x = torch.cat([x1, x2], 1)
        return x


def conv3x3(in_planes, out_planes, stride=1, s=4, d=3):
    "3x3 convolution with padding"
    return GhostModule(in_planes, out_planes, kernel_size=3, dw_size=d, ratio=s,
                     stride=stride, padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, s=4, d=3):
        super(Bottleneck, self).__init__()
        self.conv1 = GhostModule(inplanes, planes, kernel_size=1, dw_size=d, ratio=s, bias=False)
        self.conv2 = GhostModule(planes, planes, kernel_size=3, dw_size=d, ratio=s,
                     stride=stride, padding=1, bias=False)
        self.conv3 = GhostModule(planes, planes * 4, kernel_size=1, dw_size=d, ratio=s, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, s=4, d=3):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, s=s, d=d)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, s=s, d=d)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, s=s, d=d)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, s=s, d=d)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) and not isinstance(m, GhostModule):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, s=4, d=3):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                GhostModule(self.inplanes, planes * block.expansion, ratio=s, dw_size=d,
                          kernel_size=1, stride=stride, bias=False),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, s, d))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, s=s, d=d))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model
 