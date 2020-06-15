import torch
import torch.nn as nn

from .cbam import CBAM

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
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


class ResNet(nn.Module):

    def __init__(self, block, layers, branch_layers, num_classes=2, zero_init_residual=False, norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.num_classes = num_classes

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer)
        inplanes_head1 = self.inplanes
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        inplanes_head2 = self.inplanes
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_layer=norm_layer)
        inplanes_head3 = self.inplanes
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

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

        # CBAM module
        self.dropout = nn.Dropout(0.3)
        self.branch_bam1 = CBAM(512 * block.expansion)
        self.branch_bam2 = CBAM(512 * block.expansion)
        self.classifier_dep1 = nn.Linear(512 * block.expansion, 1024)
        self.classifier_dep2 = nn.Linear(512 * block.expansion, 1024)
        self.branch_bam3 = CBAM(1024, no_spatial=True)
        self.branch_bam4 = CBAM(1024, no_spatial=True)
        self.classifier1 = nn.Linear(1024, self.num_classes)
        self.classifier2 = nn.Linear(1024, 3)
        self.classifier_specific_1 = nn.Linear(1024, self.num_classes)
        self.classifier_specific_2 = nn.Linear(1024, 3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))

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

        x = self.dropout(x)
        main_feature = self.branch_bam1(x)
        x1 = self.avgpool(main_feature)
        x2 = self.avgpool(self.branch_bam2(x))

        # task specific feature
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        x1 = self.classifier_dep1(x1)
        x2 = self.classifier_dep2(x2)

        out1 = self.classifier_specific_1(x1)
        out2 = self.classifier_specific_2(x2)
        #
        # # learn task correlation
        x1_att = self.branch_bam3(x1.view(x1.size(0), -1, 1, 1))
        x2_att = self.branch_bam4(x2.view(x2.size(0), -1, 1, 1))

        x1_att = x1_att.view(x1_att.size(0), -1)
        x2_att = x2_att.view(x2_att.size(0), -1)

        x1 = torch.stack([x1, x2_att], dim=0).sum(dim=0)
        x2 = torch.stack([x2, x1_att], dim=0).sum(dim=0)

        x1 = self.classifier1(x1)
        x2 = self.classifier2(x2)

        # side branch 1
        hide_feature1 = self.sb13(self.sb12(self.sb11(node1)))
        h1 = self.avgpool(hide_feature1)
        h1 = h1.view(h1.size(0), -1)
        side_out1 = self.fc_head1(h1)

        # side branch 2
        hide_feature2 = self.sb22(self.sb21(node2))
        h2 = self.avgpool(hide_feature2)
        h2 = h2.view(h2.size(0), -1)
        side_out2 = self.fc_head2(h2)

        # side branch 3
        hide_feature3 = self.sb31(node3)
        h3 = self.avgpool(hide_feature3)
        h3 = h3.view(h3.size(0), -1)
        side_out3 = self.fc_head3(h3)

        # out1: dr; out2: dme; x1: main dr; x2: main dme
        return [main_feature, out1, out2, x1, x2], [hide_feature1, side_out1], [hide_feature2, side_out2], [
            hide_feature3, side_out3]


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], [[1, 1, 2], [1, 1], [2]], **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], [[2, 3, 3], [3, 3], [3]], **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], [[2, 3, 3], [3, 3], [3]], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], [[2, 6, 3], [6, 3], [3]], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], [[4, 12, 3], [12, 3], [3]], **kwargs)
    return model


if __name__ == '__main__':
    from torchstat import stat

    net = resnet50()
    stat(net, (3, 224, 224))
