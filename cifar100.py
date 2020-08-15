#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/8/12 15:24
# @Author  : JackyLUO
# @E-mail  : lingluo@stumail.neu.edu.cn
# @Site    : 
# @File    : cifar100.py
# @Software: PyCharm

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torch.utils.model_zoo as model_zoo
from torch.utils.tensorboard import SummaryWriter

import torchvision as tv

import time
import random
import numpy as np
import argparse

from autoaugment import CIFAR10Policy
from cutout import Cutout
from utils import *
import models.cifar as models

from tqdm import tqdm
from sklearn.metrics import classification_report, precision_score, recall_score, accuracy_score

my_whole_seed = 2020
torch.manual_seed(my_whole_seed)
torch.cuda.manual_seed_all(my_whole_seed)
torch.cuda.manual_seed(my_whole_seed)
np.random.seed(my_whole_seed)
random.seed(my_whole_seed)
cudnn.deterministic = True
cudnn.benchmark = False

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
}

parser = argparse.ArgumentParser(description='PyTorch CIFAR-100 Training')
parser.add_argument('-d', '--data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-c', '--checkpoint', default='cifar100', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoints)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--num-classes', default=100, type=int,
                    help='classes number')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lr-decay', type=str, default='schedule',
                    help='mode for learning rate decay')
parser.add_argument('--warmup', action='store_true',
                    help='set lower initial learning rate to warm up the training')
parser.add_argument('--schedule', type=int, nargs='+', default=[200//3, 200 * 2//3, 200-10],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--save', dest='save', action='store_true',
                    help='save model or not')
parser.add_argument('--gpu', default='0', type=int,
                    help='GPU id to use.')
parser.add_argument('--temperature', default=3.0, type=float,
                    help='temperature for smoothing the soft target')
parser.add_argument('--theta', default=0.4, type=float,
                    help='weight of side loss')
parser.add_argument('--alpha', default=0.3, type=float,
                    help='weight of kd loss')
parser.add_argument('--beta', default=1e-5, type=float,
                    help='weight of feature loss')

best_acc = 0


def main():
    args = parser.parse_args()
    main_worker(args)


def main_worker(args):
    global best_acc
    global logger

    start_epoch = args.start_epoch

    if not os.path.exists(args.checkpoint) and not args.resume:
        mkdir_p(args.checkpoint)

    if args.gpu is not None:
        print("Use GPU: {} for experiments".format(args.gpu))

    model = models.__dict__[args.arch](args.num_classes)
    if args.pretrained:
        print("==> Load pretrained model")
        model_dict = model.state_dict()
        pretrained_dict = model_zoo.load_url(model_urls[args.arch])
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    torch.cuda.set_device(args.gpu)
    model = model.cuda()

    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    title = 'CIFAR100-' + args.arch
    if args.resume:
        if os.path.isfile(args.resume):
            print('==> Resuming from checkpoint..')
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            args.checkpoint = os.path.dirname(args.resume)
            logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            exit(-1)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Val Loss', 'Val Acc', 'Val Side1 Acc', 'Val Side2 Acc',
                          'Val Side3 Acc'])

    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])
    tra_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        # CIFAR10Policy(),
        transforms.ToTensor(),
        Cutout(n_holes=1, length=16),
        normalize,
    ])
    tra_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    # Set download=True to download data
    train_dataset = tv.datasets.CIFAR100(root=args.data, train=True, transform=tra_train, download=False)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers
    )

    val_dataset = tv.datasets.CIFAR100(root=args.data, train=False, transform=tra_test, download=False)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )

    if args.evaluate:
        m, s1, s2, s3 = test(val_loader, model)
        print(m)
        print("\n")
        print(s1)
        print("\n")
        print(s2)
        print("\n")
        print(s3)

        return

    writer = SummaryWriter(args.checkpoint)

    for epoch in range(start_epoch, args.epochs):
        total_loss = train(train_loader, model, criterion, epoch, optimizer, args)
        writer.add_scalar('Train loss', total_loss, epoch + 1)

        # evaluate on validation set
        val_total_loss, val_main_acc, val_side1_acc, val_side2_acc, val_side3_acc = validate(
            val_loader, model,
            criterion, args)
        lr = optimizer.param_groups[0]['lr']
        logger.append(
            [lr, total_loss, val_total_loss, val_main_acc, val_side1_acc, val_side2_acc, val_side3_acc])

        writer.add_scalar('Learning rate', lr, epoch + 1)
        writer.add_scalar('Val Loss', val_total_loss, epoch + 1)
        writer.add_scalar('Main Acc', val_main_acc, epoch + 1)
        writer.add_scalar('Side1 Acc', val_side1_acc, epoch + 1)
        writer.add_scalar('Side2 Acc', val_side2_acc, epoch + 1)
        writer.add_scalar('Side3 Acc', val_side3_acc, epoch + 1)

        is_best = best_acc < val_main_acc
        best_acc = max(best_acc, val_main_acc)

        if args.save:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, is_best, checkpoint=args.checkpoint)

    logger.close()
    writer.close()

    print('Best accuracy:')
    print(best_acc)

    torch.cuda.empty_cache()


def train(train_loader, model, criterion, epoch, optimizer, args):
    bar = Bar('Train Processing', max=len(train_loader))

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        adjust_learning_rate(optimizer, epoch, i, len(train_loader), args)

        input = input.cuda()
        target = target.cuda()

        optimizer.zero_grad()

        # x1 is the main output
        [cam, main_feature, x1], [hide_feature1, side_out1], [hide_feature2, side_out2], [
            hide_feature3, side_out3] = model(input)

        # cross-entropy loss
        cam_loss = criterion(cam, target)
        out_main_loss = criterion(x1, target)
        side1_loss = criterion(side_out1, target)
        side2_loss = criterion(side_out2, target)
        side3_loss = criterion(side_out3, target)

        # K-L divergence loss
        side1_kl_loss = kl_loss(side_out1, x1.detach(), args)
        side2_kl_loss = kl_loss(side_out2, x1.detach(), args)
        side3_kl_loss = kl_loss(side_out3, x1.detach(), args)

        # L2 loss
        side1_l2_loss = torch.dist(hide_feature1, main_feature.detach())
        side2_l2_loss = torch.dist(hide_feature2, main_feature.detach())
        side3_l2_loss = torch.dist(hide_feature3, main_feature.detach())

        total_loss = (1 - args.alpha) * (out_main_loss + side1_loss + side2_loss + side3_loss) + args.theta * cam_loss + args.alpha * (
                             side1_kl_loss + side2_kl_loss + side3_kl_loss) + args.beta * (
                             side1_l2_loss + side2_l2_loss + side3_l2_loss)
        losses.update(total_loss.item(), input.size(0))

        total_loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.3f}'.format(
            batch=i + 1,
            size=len(train_loader),
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
        )
        bar.next()
    bar.finish()
    return losses.avg


def validate(val_loader, model, criterion, args):
    bar = Bar('Validation Processing', max=len(val_loader))

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    middle1_top1 = AverageMeter()
    middle2_top1 = AverageMeter()
    middle3_top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()

            [cam, main_feature, x1], [hide_feature1, side_out1], [hide_feature2, side_out2], [
                hide_feature3, side_out3] = model(input)

            # cross-entropy loss
            cam_loss = criterion(cam, target)
            out_main_loss = criterion(x1, target)
            side1_loss = criterion(side_out1, target)
            side2_loss = criterion(side_out2, target)
            side3_loss = criterion(side_out3, target)

            # K-L divergence loss
            side1_kl_loss = kl_loss(side_out1, x1.detach(), args)
            side2_kl_loss = kl_loss(side_out2, x1.detach(), args)
            side3_kl_loss = kl_loss(side_out3, x1.detach(), args)

            # L2 loss
            side1_l2_loss = torch.dist(hide_feature1, main_feature.detach())
            side2_l2_loss = torch.dist(hide_feature2, main_feature.detach())
            side3_l2_loss = torch.dist(hide_feature3, main_feature.detach())

            total_loss = (1 - args.alpha) * (out_main_loss + side1_loss + side2_loss + side3_loss) + args.theta * cam_loss + args.alpha * (
                                 side1_kl_loss + side2_kl_loss + side3_kl_loss) + args.beta * (
                                 side1_l2_loss + side2_l2_loss + side3_l2_loss)
            losses.update(total_loss.item(), input.size(0))

            # Accuracy
            prec1 = accuracy(x1, target, topk=(1,))
            top1.update(prec1[0], input.size(0))
            middle1_prec1 = accuracy(side_out1, target, topk=(1,))
            middle1_top1.update(middle1_prec1[0], input.size(0))
            middle2_prec1 = accuracy(side_out2, target, topk=(1,))
            middle2_top1.update(middle2_prec1[0], input.size(0))
            middle3_prec1 = accuracy(side_out3, target, topk=(1,))
            middle3_top1.update(middle3_prec1[0], input.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.3f} | top1: {top1: .3f}, {middle1_top1: .3f}, {middle2_top1: .3f}, {middle3_top1: .3f}'.format(
                batch=i + 1,
                size=len(val_loader),
                data=data_time.avg,
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                top1=top1.avg,
                middle1_top1=middle1_top1.avg,
                middle2_top1=middle2_top1.avg,
                middle3_top1=middle3_top1.avg,
            )
            bar.next()
        bar.finish()

    return losses.avg, top1.avg, middle1_top1.avg, middle2_top1.avg, middle3_top1.avg


def test(test_loader, model):
    model.eval()

    # main_target = []
    # main_out = []
    # side1_out = []
    # side2_out = []
    # side3_out = []
    y_target_list = []
    y_main_list = []
    y_side1_list = []
    y_side2_list = []
    y_side3_list = []


    with torch.no_grad():
        for i, (input, target) in tqdm(enumerate(test_loader)):
            input = input.cuda()
            target = target.cuda()

            [_, _, m_out], [_, side_out1], [_, side_out2], [_, side_out3] = model(input)
            # m_out = torch.softmax(m_out, dim=1)
            # side_out1 = torch.softmax(side_out1, dim=1)
            # side_out2 = torch.softmax(side_out2, dim=1)
            # side_out3 = torch.softmax(side_out3, dim=1)

            # main_target.append(target.cpu().data.numpy())
            # main_out.append(m_out.cpu().data.numpy())
            # side1_out.append(side_out1.cpu().data.numpy())
            # side2_out.append(side_out2.cpu().data.numpy())
            # side3_out.append(side_out3.cpu().data.numpy())
            y_main_softmax = torch.log_softmax(m_out, dim=1)
            _, y_main_tags = torch.max(y_main_softmax, dim=1)
            y_main_list.append(y_main_tags.cpu().numpy())

            y_side1_softmax = torch.log_softmax(side_out1, dim=1)
            _, y_side1_tags = torch.max(y_side1_softmax, dim=1)
            y_side1_list.append(y_side1_tags.cpu().numpy())

            y_side2_softmax = torch.log_softmax(side_out2, dim=1)
            _, y_side2_tags = torch.max(y_side2_softmax, dim=1)
            y_side2_list.append(y_side2_tags.cpu().numpy())

            y_side3_softmax = torch.log_softmax(side_out3, dim=1)
            _, y_side3_tags = torch.max(y_side3_softmax, dim=1)
            y_side3_list.append(y_side3_tags.cpu().numpy())

            y_target_list.append(target.cpu().numpy())

    print("\nCalculating...")

    # all_target = [item for sublist in main_target for item in sublist]
    # m_output = [item for sublist in main_out for item in sublist]
    # s1_output = [item for sublist in side1_out for item in sublist]
    # s2_output = [item for sublist in side2_out for item in sublist]
    # s3_output = [item for sublist in side3_out for item in sublist]

    # # precision and recall
    # m_dr_p = precision_score(all_target, np.argmax(m_output, axis=1), average='macro')
    # m_dr_r = recall_score(all_target, np.argmax(m_output, axis=1), average='macro')
    # s1_dr_p = precision_score(all_target, np.argmax(s1_output, axis=1), average='macro')
    # s1_dr_r = recall_score(all_target, np.argmax(s1_output, axis=1), average='macro')
    # s2_dr_p = precision_score(all_target, np.argmax(s2_output, axis=1), average='macro')
    # s2_dr_r = recall_score(all_target, np.argmax(s2_output, axis=1), average='macro')
    # s3_dr_p = precision_score(all_target, np.argmax(s3_output, axis=1), average='macro')
    # s3_dr_r = recall_score(all_target, np.argmax(s3_output, axis=1), average='macro')
    y_target_list = [a.squeeze().tolist() for a in y_target_list]
    y_main_list = [a.squeeze().tolist() for a in y_main_list]
    y_side1_list = [a.squeeze().tolist() for a in y_side1_list]
    y_side2_list = [a.squeeze().tolist() for a in y_side2_list]
    y_side3_list = [a.squeeze().tolist() for a in y_side3_list]

    m_dr_a = accuracy_score(y_target_list, y_main_list)
    m_dr_p = precision_score(y_target_list, y_main_list, average='macro')
    m_dr_r = recall_score(y_target_list, y_main_list, average='macro')

    s1_dr_a = accuracy_score(y_target_list, y_side1_list)
    s1_dr_p = precision_score(y_target_list, y_side1_list, average='macro')
    s1_dr_r = recall_score(y_target_list, y_side1_list, average='macro')

    s2_dr_a = accuracy_score(y_target_list, y_side2_list)
    s2_dr_p = precision_score(y_target_list, y_side2_list, average='macro')
    s2_dr_r = recall_score(y_target_list, y_side2_list, average='macro')

    s3_dr_a = accuracy_score(y_target_list, y_side3_list)
    s3_dr_p = precision_score(y_target_list, y_side3_list, average='macro')
    s3_dr_r = recall_score(y_target_list, y_side3_list, average='macro')

    return [m_dr_a, m_dr_p, m_dr_r], [s1_dr_a, s1_dr_p, s1_dr_r], [s2_dr_a, s2_dr_p, s2_dr_r], [s3_dr_a, s3_dr_p, s3_dr_r]


if __name__ == '__main__':
    main()
