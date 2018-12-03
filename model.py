import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from torchvision.models.resnet import ResNet, BasicBlock, model_urls, model_zoo


from data import (
    n_class
)


def make_backbone_resnet34(pretrained=True, **kwargs):
    backbone = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    del backbone.fc
    del backbone.avgpool
    print('Removed fc and avgpool')

    if pretrained:
        backbone.load_state_dict(model_zoo.load_url(model_urls['resnet34']),
                                 strict=False)
        print('ImageNet pretrained weights were loaded')

    conv1_weight = backbone.conv1.weight
    del backbone.conv1
    backbone.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7),
                               stride=(2, 2), padding=(3, 3), bias=False)
    backbone.conv1.weight = nn.Parameter(torch.cat((conv1_weight, torch.zeros(64, 1, 7, 7)), dim=1))
    torch.nn.init.kaiming_normal_(backbone.conv1.weight[:, 3])
    print('Set 4 channel input conv1')

    return backbone


class GAP(nn.Module):
    def __init__(self, flatten=False):
        super().__init__()
        self.enable_flatten = flatten

    def forward(self, x):
        x = F.avg_pool2d(x, (x.shape[-2], x.shape[-1]))
        if self.enable_flatten:
            x = x.view(x.size(0), -1)
        return x


class GAMP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        return torch.cat([self.gap(x), self.gmp(x)], 1).view(x.size(0), -1)


class ResNet34(nn.Module):
    def __init__(self, pretrained=True, **kwargs):
        super().__init__()
        self.backbone = make_backbone_resnet34(pretrained=pretrained,
                                               **kwargs)
        self.gap = GAP(flatten=True)
        self.fc = nn.Linear(512, n_class)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.gap(x)
        logit = self.fc(x)
        return logit


class ResNet34v2(nn.Module):
    def __init__(self, pretrained=True, **kwargs):
        super().__init__()
        self.backbone = make_backbone_resnet34(pretrained=pretrained,
                                               **kwargs)
        self.gamp = GAMP()
        self.bn = nn.BatchNorm1d(1024)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(1024, n_class)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.gamp(x)
        x = self.bn(x)
        x = self.dropout(x)

        logit = self.fc(x)
        return logit