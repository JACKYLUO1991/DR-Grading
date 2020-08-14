#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/6/5 13:18
# @Author  : JackyLUO
# @E-mail  : lingluo@stumail.neu.edu.cn
# @Site    : 
# @File    : resnet.py
# @Software: PyCharm

import torch.nn as nn

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """Self-distillation based on resnet"""

    def __init__(self, block, layers, branch_layers, num_classes=2):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        inplanes_head1 = self.inplanes
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        inplanes_head2 = self.inplanes
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        inplanes_head3 = self.inplanes
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.fc_main = nn.Linear(512 * block.expansion, num_classes)

        # side branch 1
        self.inplanes = inplanes_head1
        self.sb11 = self._make_layer(block, 128, branch_layers[0][0], stride=2)
        self.sb12 = self._make_layer(block, 256, branch_layers[0][1], stride=2)
        self.sb13 = self._make_layer(block, 512, branch_layers[0][2], stride=2)
        self.fc_head1 = nn.Linear(512 * block.expansion, num_classes)

        # side branch 2
        self.inplanes = inplanes_head2
        self.sb21 = self._make_layer(block, 256, branch_layers[1][0], stride=2)
        self.sb22 = self._make_layer(block, 512, branch_layers[1][1], stride=2)
        self.fc_head2 = nn.Linear(512 * block.expansion, num_classes)

        # side branch 3
        self.inplanes = inplanes_head3
        self.sb31 = self._make_layer(block, 512, branch_layers[2][0], stride=2)
        self.fc_head3 = nn.Linear(512 * block.expansion, num_classes)

        # CAM-attention
        self.cam_conv1 = nn.Conv2d(512 * block.expansion, num_classes, kernel_size=1, padding=0,
                                   bias=False)
        self.cam_bn1 = nn.BatchNorm2d(num_classes)
        self.cam_out = nn.Conv2d(num_classes, num_classes, kernel_size=1, padding=0,
                                 bias=False)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.cam_conv2 = nn.Conv2d(num_classes, 1, kernel_size=3, padding=1,
                                   bias=False)
        self.cam_bn2 = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = node1 = self.layer1(x)
        x = node2 = self.layer2(x)
        x = node3 = self.layer3(x)
        x = self.layer4(x)

        # CAM branch
        cam1 = self.relu(self.cam_bn1(self.cam_conv1(x)))
        out_cam = self.gap(self.cam_out(cam1))
        out_cam = out_cam.view(out_cam.size(0), -1)

        # main branch
        cam2 = self.sigmoid(self.cam_bn2(self.cam_conv2(cam1)))
        main_feature = x * cam2 + x
        m = self.gap(main_feature)
        m = m.view(m.size(0), -1)
        out_main = self.fc_main(m)

        # side branch 1
        hide_feature1 = self.sb13(self.sb12(self.sb11(node1)))
        h1 = self.gap(hide_feature1)
        h1 = h1.view(h1.size(0), -1)
        side_out1 = self.fc_head1(h1)

        # side branch 2
        hide_feature2 = self.sb22(self.sb21(node2))
        h2 = self.gap(hide_feature2)
        h2 = h2.view(h2.size(0), -1)
        side_out2 = self.fc_head2(h2)

        # side branch 3
        hide_feature3 = self.sb31(node3)
        h3 = self.gap(hide_feature3)
        h3 = h3.view(h3.size(0), -1)
        side_out3 = self.fc_head3(h3)

        return [out_cam, main_feature, out_main], [hide_feature1, side_out1], [hide_feature2, side_out2], [
            hide_feature3, side_out3]


        # Main  Branch1   Branch2  Branch3
        # 1.84GFlops  1.36GFlops  1.59GFlops 1.82GFlops   


def resnet18(num_classes, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], [[1, 1, 2], [1, 2], [2]], num_classes, **kwargs)
    return model


def resnet34(num_classes, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], [[2, 3, 3], [3, 3], [3]], num_classes, **kwargs)
    return model


def resnet50(num_classes, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], [[2, 3, 3], [3, 3], [3]], num_classes, **kwargs)
    return model


def resnet101(num_classes, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], [[2, 6, 3], [6, 3], [3]], num_classes, **kwargs)
    return model


def resnet152(num_classes, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], [[4, 12, 3], [12, 3], [3]], num_classes, **kwargs)
    return model


if __name__ == '__main__':
    import torch
    import time
    # from torchstat import stat

    # net = resnet50(num_classes=4)
    # # stat(net, (3, 224, 224))

    # # Freeze some layers
    # ct = 0
    # for name, child in net.named_children():
    #     ct += 1
    #     if ct < 6:
    #         for names, params in child.named_children():
    #             params.requires_grad = False
    N = 500
    input = torch.randn(N, 3, 224 ,224).cuda()
    net = resnet18(num_classes=2).cuda()
    # stat(net, (3, 224, 224))

    torch.cuda.synchronize()
    start = time.time()

    with torch.no_grad():
        net(input)
    
    torch.cuda.synchronize()
    dur = time.time() - start

    print(dur / N)
