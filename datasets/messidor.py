import os
import os.path

import torch.utils.data as data
from PIL import Image
import glob
import xlrd
import numpy as np

from tqdm import tqdm


class Messidor(data.Dataset):
    """Loading MESSIDOR dataset"""

    def __init__(self, root, mode, transform=None, args=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.mode = mode
        self.args = args

        self.train_data = []
        self.train_label = []
        self.test_data = []
        self.test_label = []
        self.name = []

        xls_files = glob.glob(self.root + '/*/*.xls')
        dictLabels_DR, dictLabels_DME = self.load_csv(xls_files)

        files = np.loadtxt(self.root + "file_list.txt", dtype=str)
        idx = np.loadtxt(self.root + "/10fold/" + str(args.fold_name) + ".txt", dtype=int)
        print("foldname:", args.fold_name)

        self.test_root = [files[idx_item] for idx_item in idx]
        self.train_root = list(set(files) - set(self.test_root))
        self.train_root = [self.root + item for item in self.train_root]
        self.test_root = [self.root + item for item in self.test_root]

        if self.mode == 'train':
            for each_one in tqdm(self.train_root):
                img = Image.open(each_one)
                img = img.convert('RGB')
                label_DR = [k for k, v in dictLabels_DR.items() if each_one.split("/")[-1] in v]
                label_DME = [k for k, v in dictLabels_DME.items() if each_one.split("/")[-1] in v]
                self.train_label.append([int(label_DR[0]), int(label_DME[0])])

                assert len(label_DR) == 1
                self.train_data.append(img)
            assert len(self.train_label) == len(self.train_data)

            print('=> Total Train: ', len(self.train_data), " Multi-Task images ")

        elif self.mode == 'val':
            for item in tqdm(self.test_root):
                img = Image.open(item)
                img = img.convert('RGB')
                label_DR = [k for k, v in dictLabels_DR.items() if item.split("/")[-1] in v]
                label_DME = [k for k, v in dictLabels_DME.items() if item.split("/")[-1] in v]
                self.test_label.append([int(label_DR[0]), int(label_DME[0])])

                assert len(label_DR) == 1
                self.test_data.append(img)
            assert len(self.test_data) == len(self.test_label)
            print('=> Total Test: ', len(self.test_data), " Multi-Task images ")

    def load_csv(self, path):
        dictLabels_DR = {}
        dictLabels_DME = {}
        for per_path in path:
            # open xlsx
            xl_workbook = xlrd.open_workbook(per_path)
            xl_sheet = xl_workbook.sheet_by_index(0)
            for rowx in range(1, xl_sheet.nrows):
                cols = xl_sheet.row_values(rowx)
                filename = cols[0]
                label1 = int(cols[2])
                label2 = int(cols[3])

                if self.args.num_classes == 2:
                    if label1 < 2:
                        label1 = 0
                    else:
                        label1 = 1

                if label1 in dictLabels_DR.keys():
                    dictLabels_DR[label1].append(filename)
                else:
                    dictLabels_DR[label1] = [filename]

                if label2 in dictLabels_DME.keys():
                    dictLabels_DME[label2].append(filename)
                else:
                    dictLabels_DME[label2] = [filename]

        return dictLabels_DR, dictLabels_DME

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        if self.mode == 'train':
            img, label = self.train_data[index], self.train_label[index]
        elif self.mode == 'val':
            img, label = self.test_data[index], self.test_label[index]

        img = self.transform(img)

        return img, label

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_data)
        elif self.mode == 'val':
            return len(self.test_data)
