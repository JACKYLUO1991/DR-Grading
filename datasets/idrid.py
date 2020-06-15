#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/6/12 9:17
# @Author  : JackyLUO
# @E-mail  : lingluo@stumail.neu.edu.cn
# @Site    : 
# @File    : idrid.py
# @Software: PyCharm

import torch.utils.data as data

import csv
import glob
import numpy as np
from PIL import Image

from tqdm import tqdm


class Idrid(data.Dataset):
    """Loading IDRID dataset"""

    def __init__(self, root, mode, transform=None):
        self.root = root
        self.mode = mode
        self.transform = transform
        self.train_data = []
        self.train_label = []
        self.test_data = []
        self.test_label = []

        self.train_root = glob.glob(self.root + '/images/trainset/*.jpg')
        self.test_root = glob.glob(self.root + '/images/testset/*.jpg')

        if self.mode == 'train':
            dictLabels_DR, dictLabels_DME = self.load_csv(self.root + '/labels/trainset.csv')
            for each_one in tqdm(self.train_root):
                each_one = each_one.replace("\\", "/")  # windows and linex path difference
                img = Image.open(each_one).convert("RGB")
                img = np.array(img, dtype=np.uint8)[:, 300: -600]
                img = Image.fromarray(img)
                self.train_data.append(img)

                label_DR = [k for k, v in dictLabels_DR.items() if each_one.split("/")[-1][:-4] in v]
                label_DME = [k for k, v in dictLabels_DME.items() if each_one.split("/")[-1][:-4] in v]
                self.train_label.append([int(label_DR[0]), int(label_DME[0])])
            assert len(self.train_label) == len(self.train_label)
            print('=> Total Train: ', len(self.train_root), " Multi-Task images ")

        elif self.mode == 'test':
            dictLabels_DR, dictLabels_DME = self.load_csv(self.root + '/labels/testset.csv')
            for each_one in tqdm(self.test_root):
                each_one = each_one.replace("\\", "/")
                img = Image.open(each_one).convert("RGB")
                img = np.array(img, dtype=np.uint8)[:, 300: -600]
                img = Image.fromarray(img)
                self.test_data.append(img)

                label_DR = [k for k, v in dictLabels_DR.items() if each_one.split("/")[-1][:-4] in v]
                label_DME = [k for k, v in dictLabels_DME.items() if each_one.split("/")[-1][:-4] in v]

                self.test_label.append([int(label_DR[0]), int(label_DME[0])])
            assert len(self.test_root) == len(self.test_label)
            print('=> Total Test: ', len(self.test_root), " Multi-Task images ")

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_root)
        elif self.mode == 'test':
            return len(self.test_root)

    def __getitem__(self, item):
        if self.mode == 'train':
            img, label = self.train_data[item], self.train_label[item]
        elif self.mode == 'test':
            img, label = self.test_data[item], self.test_label[item]
        img = self.transform(img)

        return img, label

    def load_csv(self, path):
        dictLabels_DR = {}
        dictLabels_DME = {}
        with open(path) as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            next(csvreader, None)  # skip (filename, label)
            for i, row in enumerate(csvreader):
                filename = row[0]
                label1 = row[1]
                label2 = row[2]

                if label1 in dictLabels_DR.keys():
                    dictLabels_DR[label1].append(filename)
                else:
                    dictLabels_DR[label1] = [filename]

                if label2 in dictLabels_DME.keys():
                    dictLabels_DME[label2].append(filename)
                else:
                    dictLabels_DME[label2] = [filename]

            return dictLabels_DR, dictLabels_DME


if __name__ == '__main__':
    Idrid(root='../idrid_data', mode='test')
