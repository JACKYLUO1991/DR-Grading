#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/6/9 13:51
# @Author  : JackyLUO
# @E-mail  : lingluo@stumail.neu.edu.cn
# @Site    : 
# @File    : messidor.py
# @Software: PyCharm

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torch.utils.model_zoo as model_zoo
from torch.utils.tensorboard import SummaryWriter

import os
import time
import random
import numpy as np
import argparse

from utils import *
from datasets.messidor import Messidor
import models.messidor as models

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import label_binarize

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
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

parser = argparse.ArgumentParser(description='PyTorch MESSIDOR Training')
parser.add_argument('-d', '--data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-c', '--checkpoint', default='checkpoints', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoints)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--num-classes', default=2, type=int,
                    help='classes number')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=40, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lr-decay', type=str, default='cos',
                    help='mode for learning rate decay')
parser.add_argument('--warmup', action='store_true',
                    help='set lower initial learning rate to warm up the training')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
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
parser.add_argument("--fold_name", default="fold1", type=str)
parser.add_argument('--temperature', default=1, type=float,
                    help='temperature for smoothing the soft target')
parser.add_argument('--lambd', default=1, type=float,
                    help='weight of loss')
parser.add_argument('--theta', default=0.4, type=float,
                    help='weight of side loss')
parser.add_argument('--alpha', default=1, type=float,
                    help='weight of kd loss')
parser.add_argument('--beta', default=1e-7, type=float,
                    help='weight of feature loss')

# # lr
# parser.add_argument("--lr_mode", default="cosine", type=str)
# parser.add_argument("--base_lr", default=0.0003, type=float)
# parser.add_argument("--warmup_epochs", default=0, type=int)
# parser.add_argument("--warmup_lr", default=0.0, type=float)
# parser.add_argument("--targetlr", default=0.0, type=float)
# parser.add_argument("--lambda_value", default=0.25, type=float)

best_acc1 = 0


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    args = parser.parse_args()
    main_worker(args)


def main_worker(args):
    global best_acc1

    start_epoch = args.start_epoch

    if not os.path.exists(args.checkpoint):
        mkdir_p(args.checkpoint)

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    model = models.__dict__[args.arch](args.num_classes)
    if args.pretrained:
        print("==> Load pretrained model")
        model_dict = model.state_dict()
        pretrained_dict = model_zoo.load_url(model_urls[args.arch])
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    ###################################  Freeze some layers  ###################################
    # ct = 0
    # for name, child in model.named_children():
    #     ct += 1
    #     if ct < 6:
    #         for names, params in child.named_children():
    #             params.requires_grad = False
    ###################################  Freeze some layers  ###################################

    torch.cuda.set_device(args.gpu)
    model = model.cuda()

    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    # optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), args.lr,
                                 weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    title = 'Messidor-' + args.arch
    if args.resume:
        # Load checkpoint
        if os.path.isfile(args.resume):
            print('==> Resuming from checkpoint..')
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc1']
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
        logger.set_names(['Learning Rate', 'Train Loss', 'Val Loss', 'Val Acc.', 'Val Side1 Acc.', 'Val Side2 Acc.',
                          'Val Side3 Acc.', 'Val Auc.', 'Side1 Auc.', 'Side2 Auc.', 'Side3 Auc.'])

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    tra_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    tra_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize])

    val_dataset = Messidor(root=args.data, mode='val',
                           transform=tra_test, args=args)

    train_dataset = Messidor(root=args.data, mode='train', transform=tra_train, args=args)

    # For unbalanced dataset we create a weighted sampler, binary classification
    weights = make_weights_for_balanced_classes(train_dataset.train_label, args.num_classes)
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False, sampler=sampler,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        pass
        # a = time.time()
        # # savedir = args.resume.replace("model_converge.pth.tar","")
        # savedir = args.resume.replace(args.resume.split("/")[-1], "")
        # # savedir = "./"
        # if not args.multitask:
        #     acc, auc, precision_dr, recall_dr, f1score_dr = validate(val_loader, model, args)
        #     result_list = [acc, auc, precision_dr, recall_dr, f1score_dr]
        #     print("acc, auc, precision, recall, f1", acc, auc, precision_dr, recall_dr, f1score_dr)
        #     save_result_txt(savedir, result_list)
        #     print("time", time.time() - a)
        #     return
        # else:
        #     acc_dr, acc_dme, acc_joint, other_results, se, sp = validate(val_loader, model, args)
        #     print("acc_dr, acc_dme, acc_joint", acc_dr, acc_dme, acc_joint)
        #     print("auc_dr, auc_dme, precision_dr, precision_dme, recall_dr, recall_dme, f1score_dr, f1score_dme",
        #           other_results)
        #     print("se, sp", se, sp)
        #     result_list = [acc_dr, acc_dme, acc_joint]
        #     result_list += other_results
        #     result_list += [se, sp]
        #     save_result_txt(savedir, result_list)
        #
        #     print("time", time.time() - a)
        #     return

    writer = SummaryWriter(args.checkpoint)
    # lr_scheduler = LRScheduler(optimizer, len(train_loader), args)

    for epoch in range(start_epoch, args.epochs):
        total_loss = train(train_loader, model, criterion, writer, epoch, optimizer, args)
        writer.add_scalar('Train loss', total_loss, epoch + 1)

        # evaluate on validation set
        val_total_loss, val_main_acc, val_side1_acc, val_side2_acc, val_side3_acc, m_auc, s1_auc, s2_auc, s3_auc = validate(
            val_loader, model,
            criterion, args)
        lr = optimizer.param_groups[0]['lr']
        logger.append(
            [lr, total_loss, val_total_loss, val_main_acc, val_side1_acc, val_side2_acc, val_side3_acc, m_auc, s1_auc,
             s2_auc, s3_auc])

        writer.add_scalar('Learning rate', lr, epoch + 1)
        writer.add_scalar('Val Loss', val_total_loss, epoch + 1)
        writer.add_scalar('Main Acc', val_main_acc, epoch + 1)
        writer.add_scalar('Side1 Acc', val_side1_acc, epoch + 1)
        writer.add_scalar('Side2 Acc', val_side2_acc, epoch + 1)
        writer.add_scalar('Side3 Acc', val_side3_acc, epoch + 1)
        writer.add_scalar('Main Auc', m_auc, epoch + 1)
        writer.add_scalar('Side1 Auc', s1_auc, epoch + 1)
        writer.add_scalar('Side2 Auc', s2_auc, epoch + 1)
        writer.add_scalar('Side3 Auc', s3_auc, epoch + 1)

        is_best = val_main_acc > best_acc1
        best_acc1 = max(val_main_acc, best_acc1)

        if args.save:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, is_best, checkpoint=args.checkpoint)

    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.eps'))
    writer.close()

    print('Best accuracy:')
    print(best_acc1)

    torch.cuda.empty_cache()


def train(train_loader, model, criterion, writer, epoch, optimizer, args):
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
        # lr = lr_scheduler.update(i, epoch + 1)
        # writer.add_scalar("lr", lr, epoch + 1)

        input = input.cuda()
        target = [item.cuda() for item in target]

        # x1 is the main output
        [cam, main_feature, x1], [hide_feature1, side_out1], [hide_feature2, side_out2], [
            hide_feature3, side_out3] = model(input)

        # cross-entropy loss
        cam_loss = criterion(cam, target[0])
        out_main_loss = criterion(x1, target[0])
        side1_loss = criterion(side_out1, target[0])
        side2_loss = criterion(side_out2, target[0])
        side3_loss = criterion(side_out3, target[0])

        # K-L divergence loss
        side1_kl_loss = kl_loss(side_out1, x1.detach(), args.temperature)
        side2_kl_loss = kl_loss(side_out2, x1.detach(), args.temperature)
        side3_kl_loss = kl_loss(side_out3, x1.detach(), args.temperature)

        # L2 loss
        side1_l2_loss = feature_loss(hide_feature1, main_feature.detach())
        side2_l2_loss = feature_loss(hide_feature2, main_feature.detach())
        side3_l2_loss = feature_loss(hide_feature3, main_feature.detach())

        total_loss = (out_main_loss + args.lambd * (
                side1_loss + side2_loss + side3_loss)) + args.theta * cam_loss + args.alpha * (
                             side1_kl_loss + side2_kl_loss + side3_kl_loss) + args.beta * (
                             side1_l2_loss + side2_l2_loss + side3_l2_loss)
        losses.update(total_loss.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
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

    all_target = []
    main_output = []
    side1_output = []
    side2_output = []
    side3_output = []

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = [item.cuda() for item in target]

            [cam, main_feature, x1], [hide_feature1, side_out1], [hide_feature2, side_out2], [
                hide_feature3, side_out3] = model(input)

            # Auc calculation
            main_o = torch.softmax(x1, dim=1)
            side1_o = torch.softmax(side_out1, dim=1)
            side2_o = torch.softmax(side_out2, dim=1)
            side3_o = torch.softmax(side_out3, dim=1)

            all_target.append(target[0].cpu().data.numpy())
            main_output.append(main_o.cpu().data.numpy())
            side1_output.append(side1_o.cpu().data.numpy())
            side2_output.append(side2_o.cpu().data.numpy())
            side3_output.append(side3_o.cpu().data.numpy())

            # cross-entropy loss
            cam_loss = criterion(cam, target[0])
            out_main_loss = criterion(x1, target[0])
            side1_loss = criterion(side_out1, target[0])
            side2_loss = criterion(side_out2, target[0])
            side3_loss = criterion(side_out3, target[0])

            # K-L divergence loss
            side1_kl_loss = kl_loss(side_out1, x1.detach(), args.temperature)
            side2_kl_loss = kl_loss(side_out2, x1.detach(), args.temperature)
            side3_kl_loss = kl_loss(side_out3, x1.detach(), args.temperature)

            # L2 loss
            side1_l2_loss = feature_loss(hide_feature1, main_feature.detach())
            side2_l2_loss = feature_loss(hide_feature2, main_feature.detach())
            side3_l2_loss = feature_loss(hide_feature3, main_feature.detach())

            total_loss = (out_main_loss + args.lambd * (
                    side1_loss + side2_loss + side3_loss)) + args.theta * cam_loss + args.alpha * (
                                 side1_kl_loss + side2_kl_loss + side3_kl_loss) + args.beta * (
                                 side1_l2_loss + side2_l2_loss + side3_l2_loss)
            losses.update(total_loss.item(), input.size(0))

            # Accuracy
            prec1 = accuracy(x1, target[0], topk=(1,))
            top1.update(prec1[0], input.size(0))
            middle1_prec1 = accuracy(side_out1, target[0], topk=(1,))
            middle1_top1.update(middle1_prec1[0], input.size(0))
            middle2_prec1 = accuracy(side_out2, target[0], topk=(1,))
            middle2_top1.update(middle2_prec1[0], input.size(0))
            middle3_prec1 = accuracy(side_out3, target[0], topk=(1,))
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

    # Borrowed from CANet
    all_target = [item for sublist in all_target for item in sublist]
    m_output = [item for sublist in main_output for item in sublist]
    s1_output = [item for sublist in side1_output for item in sublist]
    s2_output = [item for sublist in side2_output for item in sublist]
    s3_output = [item for sublist in side3_output for item in sublist]

    m_auc = roc_auc_score(all_target, [item[1] for item in m_output])
    s1_auc = roc_auc_score(all_target, [item[1] for item in s1_output])
    s2_auc = roc_auc_score(all_target, [item[1] for item in s2_output])
    s3_auc = roc_auc_score(all_target, [item[1] for item in s3_output])

    return losses.avg, top1.avg, middle1_top1.avg, middle2_top1.avg, middle3_top1.avg, m_auc, s1_auc, s2_auc, s3_auc

    #         output = torch.softmax(x1, dim=1)
    #
    #         all_target.append(target[0].cpu().data.numpy())
    #         all_output.append(output.cpu().data.numpy())
    #
    # all_target = [item for sublist in all_target for item in sublist]
    # all_output = [item for sublist in all_output for item in sublist]
    #
    # acc_dr = accuracy_score(all_target, np.argmax(all_output, axis=1))
    # auc_dr = roc_auc_score(all_target, [item[1] for item in all_output])
    #
    # precision_dr = precision_score(all_target, np.argmax(all_output, axis=1))
    # recall_dr = recall_score(all_target, np.argmax(all_output, axis=1))
    # f1score_dr = f1_score(all_target, np.argmax(all_output, axis=1))
    #
    # cm1 = confusion_matrix(all_target, np.argmax(all_output, axis=1))
    # sensitivity1 = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
    # specificity1 = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])


def multi_class_auc(all_target, all_output, num_c=None):
    all_output = np.stack(all_output)
    all_target = label_binarize(all_target, classes=list(range(0, num_c)))
    auc_sum = []

    for num_class in range(0, num_c):
        try:
            auc = roc_auc_score(all_target[:, num_class], all_output[:, num_class])
            auc_sum.append(auc)
        except ValueError:
            pass

    auc = sum(auc_sum) / float(len(auc_sum))

    return auc


def save_result_txt(savedir, result):
    with open(savedir + '/result.txt', 'w') as f:
        for item in result:
            f.write("%.8f\n" % item)
        f.close()


if __name__ == '__main__':
    main()
