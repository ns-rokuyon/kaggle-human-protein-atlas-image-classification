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


def make_resnet34_binary_classifier(model):
    assert isinstance(model, ResNet34)

    del model.fc
    print('Removed fc')

    model.fc = nn.Linear(512, 1)
    print('Append new fc for binary output')
    return model


def freeze_backbone(model):
    for name, param in model.named_parameters():
        if name.startswith('backbone'):
            param.requires_grad = False
        else:
            param.requires_grad = True
        print(f'{name}: {param.requires_grad}')


def unfreeze(model):
    for name, param in model.named_parameters():
        param.requires_grad = True
        print(f'{name}: {param.requires_grad}')


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


class ConvBn2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size,
                      stride=1, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch)
        )
        torch.nn.init.kaiming_normal_(self.layer[0].weight)

    def forward(self, x):
        return self.layer(x)


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


class ResNet34v3(nn.Module):
    def __init__(self, pretrained=True, **kwargs):
        super().__init__()
        self.backbone = make_backbone_resnet34(pretrained=pretrained,
                                               **kwargs)
        self.gamp = GAMP()
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(1024, 512, bias=True)
        self.fc2 = nn.Linear(512, n_class, bias=True)

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

        x = self.bn1(x)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.relu1(x)

        x = self.bn2(x)
        x = self.dropout2(x)
        logit = self.fc2(x)

        return logit


class ABNResNet34(nn.Module):
    def __init__(self, pretrained=True, **kwargs):
        super().__init__()
        self.backbone = make_backbone_resnet34(pretrained=pretrained,
                                               **kwargs)
        self.gap = GAP(flatten=True)
        self.fc = nn.Linear(512, n_class)

        self.att_block = self.backbone._make_layer(BasicBlock, 512, 3, stride=1)
        self.att_bn = nn.BatchNorm2d(512)
        self.att_conv = ConvBn2d(512, n_class, kernel_size=1, padding=0)
        self.att_relu = nn.ReLU(inplace=True)
        self.att_map_conv = ConvBn2d(n_class, 1, kernel_size=3, padding=1)
        self.att_map_sigmoid = nn.Sigmoid()
        self.att_out_conv = nn.Conv2d(n_class, n_class, kernel_size=1, padding=0, bias=False)
        self.att_out_gap = GAP(flatten=True)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)

        a = self.att_block(x)
        a = self.att_bn(a)
        a = self.att_conv(a)
        a = self.att_relu(a)

        b = self.att_out_conv(a)
        attention_branch_logit = self.gap(b)

        c = self.att_map_conv(a)
        attention_map = self.att_map_sigmoid(c)

        x = x + attention_map * x
        x = self.backbone.layer4(x)

        x = self.gap(x)
        logit = self.fc(x)

        return logit, attention_branch_logit, attention_map