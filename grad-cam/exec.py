#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/7/6 12:46
# @Author  : JackyLUO
# @E-mail  : lingluo@stumail.neu.edu.cn
# @Site    : 
# @File    : exec.py.py
# @Software: PyCharm

import argparse
import os
import re

import cv2
import numpy as np
import torch
from skimage import io
from torch import nn
from torchvision import models

from gram_cam import GradCAM
from guided_back_propagation import GuidedBackPropagation

import sys
sys.path.append("..")
from models.messidor import resnet

from PIL import Image
import torchvision.transforms as transforms


def get_net(net_name, weight_path=None):
    # pretrain = weight_path is None
    # if net_name in ['vgg', 'vgg16']:
    #     net = models.vgg16(pretrained=pretrain)
    # elif net_name == 'vgg19':
    #     net = models.vgg19(pretrained=pretrain)
    # elif net_name in ['resnet', 'resnet50']:
    #     net = models.resnet50(pretrained=pretrain)
    # elif net_name == 'resnet101':
    #     net = models.resnet101(pretrained=pretrain)
    # elif net_name in ['densenet', 'densenet121']:
    #     net = models.densenet121(pretrained=pretrain)
    # elif net_name in ['inception']:
    #     net = models.inception_v3(pretrained=pretrain)
    # elif net_name in ['mobilenet_v2']:
    #     net = models.mobilenet_v2(pretrained=pretrain)
    # elif net_name in ['shufflenet_v2']:
    #     net = models.shufflenet_v2_x1_0(pretrained=pretrain)
    # else:
    #     raise ValueError('invalid network name:{}'.format(net_name))
    # if weight_path is not None and net_name.startswith('densenet'):
    #     pattern = re.compile(
    #         r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
    #     state_dict = torch.load(weight_path)
    #     for key in list(state_dict.keys()):
    #         res = pattern.match(key)
    #         if res:
    #             new_key = res.group(1) + res.group(2)
    #             state_dict[new_key] = state_dict[key]
    #             del state_dict[key]
    #     net.load_state_dict(state_dict)
    net = resnet.resnet18(num_classes=2)
    if weight_path is not None:
        # net.load_state_dict(torch.load(weight_path))
        checkpoint = torch.load(weight_path, map_location="cuda:0") 
        net.load_state_dict(checkpoint['state_dict'])

    return net


def get_last_conv_name(net):
    """
    :param net:
    :return:
    """
    layer_name = None
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            layer_name = name
    return layer_name


def prepare_input(image):
    image = image.copy()
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    image -= means
    image /= stds

    image = np.ascontiguousarray(np.transpose(image, (2, 0, 1)))
    image = image[np.newaxis, ...]

    return torch.tensor(image, requires_grad=True)


def gen_cam(image, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1]  # gbr to rgb

    cam = heatmap + np.float32(image)
    return norm_image(cam), (heatmap * 255).astype(np.uint8)


def norm_image(image):
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)


def gen_gb(grad):
    grad = grad.data.numpy()
    gb = np.transpose(grad, (1, 2, 0))
    return gb


def save_image(image_dicts, input_image_name, network, output_dir):
    prefix = os.path.splitext(input_image_name)[0]
    for key, image in image_dicts.items():
        io.imsave(os.path.join(output_dir, '{}-{}-{}.jpg'.format(prefix, network, key)), image)


def main(args):
    # img = io.imread(args.image_path)
    img = Image.open(args.image_path).convert("RGB")
    trans = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224)]
    )
    img = trans(img)
    img.save("results/rgb.jpg")
    img = np.float32(cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)) / 255
    # img = np.float32(cv2.resize(img, (224, 224))) / 255
    inputs = prepare_input(img)
    image_dict = {}
    net = get_net(args.network, args.weight_path)
    # Grad-CAM
    layer_name = get_last_conv_name(net) if args.layer_name is None else args.layer_name
    grad_cam = GradCAM(net, layer_name)
    mask = grad_cam(inputs, args.class_id)  # cam mask
    image_dict['cam'], image_dict['heatmap'] = gen_cam(img, mask)
    grad_cam.remove_handlers()
    # # Grad-CAM++
    # grad_cam_plus_plus = GradCamPlusPlus(net, layer_name)
    # mask_plus_plus = grad_cam_plus_plus(inputs, args.class_id)  # cam mask
    # image_dict['cam++'], image_dict['heatmap++'] = gen_cam(img, mask_plus_plus)
    # grad_cam_plus_plus.remove_handlers()

    # GuidedBackPropagation
    gbp = GuidedBackPropagation(net)
    inputs.grad.zero_()
    grad = gbp(inputs)

    gb = gen_gb(grad)
    image_dict['gb'] = norm_image(gb)
    # cam_gb = gb * mask[..., np.newaxis]
    # image_dict['cam_gb'] = norm_image(cam_gb)

    save_image(image_dict, os.path.basename(args.image_path), args.network, args.output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='resnet18',
                        help='ImageNet classification network')
    parser.add_argument('--image-path', type=str, default='./examples/pic1.jpg',
                        help='input image path')
    parser.add_argument('--weight-path', type=str, default=None,
                        help='weight path of the model')
    parser.add_argument('--layer-name', type=str, default=None,
                        help='last convolutional layer name')
    parser.add_argument('--class-id', type=int, default=None,
                        help='class id')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='output directory to save results')
    arguments = parser.parse_args()

    main(arguments)
