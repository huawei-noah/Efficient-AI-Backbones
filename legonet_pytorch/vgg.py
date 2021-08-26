'''
2019.07.24 Changed details for LegoNet
           Huawei Technologies Co., Ltd. <foss@huawei.com>
'''

import torch
import torch.nn as nn

from module import *

cfg = {
    'lego_vgg16': [64, 64, 64, 'M', 128, 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
}


class lego_vgg16(nn.Module):
    def __init__(self, vgg_name, n_split, n_lego, n_classes):
        super(lego_vgg16, self).__init__()
        self.n_split, self.n_lego, self.n_classes = n_split, n_lego, n_classes
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, n_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for i, x in enumerate(cfg):
            if i == 0:
                layers += [nn.Conv2d(in_channels, x, 3, padding = 1),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True)]
                in_channels = x
                continue
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [LegoConv2d(in_channels, x, 3, self.n_split, self.n_lego),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def copy_grad(self, balance_weight):
        for layer in self.features.children():
            if isinstance(layer, LegoConv2d):
                layer.copy_grad(balance_weight)
