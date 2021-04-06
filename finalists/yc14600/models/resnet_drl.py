from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

########## functions are based on tensorflow keras implementation ##########

import os
import torch
from torchvision.models.resnet import ResNet, Bottleneck, BasicBlock
from torchvision.models.utils import load_state_dict_from_url
from ResNeSt.resnest.torch.resnet import ResNet as ResNetSt
from ResNeSt.resnest.torch.resnet import Bottleneck as Bottleneck_st
from ResNeSt.resnest.torch.resnest import short_hash


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


class ResNet_DRL(ResNet):

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        self.H = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        self.H.append(x)
        x = self.layer1(x)
        self.H.append(x)
        x = self.layer2(x)
        self.H.append(x)
        x = self.layer3(x)
        self.H.append(x)
        x = self.layer4(x)
        self.H.append(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        if self.incl_fc:
            self.H.append(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)

class ResNeSt_DRL(ResNetSt):

    def forward(self, x):
        self.H = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        self.H.append(x)
        x = self.layer2(x)
        self.H.append(x)
        x = self.layer3(x)
        self.H.append(x)
        x = self.layer4(x)
        self.H.append(x)

        x = self.avgpool(x)
        #x = x.view(x.size(0), -1)
        x = torch.flatten(x, 1)
        if self.drop:
            x = self.drop(x)
        x = self.fc(x)
        self.H.append(x)
        return x

def _resnet(arch, block, layers, pretrained, progress,incl_fc=True, **kwargs):
    model = ResNet_DRL(block, layers, **kwargs)
    model.incl_fc = incl_fc
        
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet50(pretrained=False, progress=True,incl_fc=True, **kwargs):

    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,incl_fc,
                   **kwargs)

def resnet18(pretrained=False, progress=True,incl_fc=True, **kwargs):

    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,incl_fc,
                   **kwargs)

def resnest50(pretrained=False, root='~/.encoding/models', **kwargs):

    _url_format = 'https://hangzh.s3.amazonaws.com/encoding/models/{}-{}.pth'

    _model_sha256 = {name: checksum for checksum, name in [
        ('528c19ca', 'resnest50'),
        ('22405ba7', 'resnest101'),
        ('75117900', 'resnest200'),
        ('0cc87c48', 'resnest269'),
        ]}

    resnest_model_urls = {name: _url_format.format(name, short_hash(name)) for
    name in _model_sha256.keys()}

    model = ResNeSt_DRL(Bottleneck_st, [3, 4, 6, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=32, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(
            resnest_model_urls['resnest50'], progress=True, check_hash=True))
    return model
