from __future__ import print_function

import os
import os.path
import numpy as np
import random
import pickle
import json
import math

import torch
import torch.utils.data as data
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
# import torchnet as tnt

# import h5py

from PIL import Image
from PIL import ImageEnhance

from pdb import set_trace as breakpoint


# Set the appropriate paths of the datasets here.
_MINI_IMAGENET_DATASET_DIR = '/home/eegrad/snag/Desktop/fs_ood/src/datasets/db/miniimagenet2/' ## your miniimagenet folder


def buildLabelIndex(labels):
    label2inds = {}
    for idx, label in enumerate(labels):
        if label not in label2inds:
            label2inds[label] = []
        label2inds[label].append(idx)

    return label2inds


def load_data(file):
    try:
        with open(file, 'rb') as fo:
            data = pickle.load(fo)
        return data
    except:
        with open(file, 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            data = u.load()
        return data

class MiniImageNet(data.Dataset):
    def __init__(self, phase='train', do_not_use_random_transf=False):

        self.base_folder = 'miniImagenet'
        #assert(phase=='train' or phase=='val' or phase=='test' or ph)
        self.phase = phase
        self.name = 'MiniImageNet_' + phase

        print('Loading mini ImageNet dataset - phase {0}'.format(phase))
        file_train_categories_train_phase = os.path.join(
            _MINI_IMAGENET_DATASET_DIR,
            'miniImageNet_category_split_train_phase_train.pickle')
        file_train_categories_val_phase = os.path.join(
            _MINI_IMAGENET_DATASET_DIR,
            'miniImageNet_category_split_train_phase_val.pickle')
        file_train_categories_test_phase = os.path.join(
            _MINI_IMAGENET_DATASET_DIR,
            'miniImageNet_category_split_train_phase_test.pickle')
        file_val_categories_val_phase = os.path.join(
            _MINI_IMAGENET_DATASET_DIR,
            'miniImageNet_category_split_val.pickle')
        file_test_categories_test_phase = os.path.join(
            _MINI_IMAGENET_DATASET_DIR,
            'miniImageNet_category_split_test.pickle')

        if self.phase=='train':
            # During training phase we only load the training phase images
            # of the training categories (aka base categories).
            data_train = load_data(file_train_categories_train_phase)
            self.data = data_train['data']
            self.labels = data_train['labels']

            self.label2ind = buildLabelIndex(self.labels)
            self.labelIds = sorted(self.label2ind.keys())
            self.num_cats = len(self.labelIds)
            self.labelIds_base = self.labelIds
            self.num_cats_base = len(self.labelIds_base)
        elif self.phase == 'trainval':
            # During training phase we only load the training phase images
            # of the training categories (aka base categories).
            data_train = load_data(file_train_categories_train_phase)
            self.data = data_train['data']
            self.labels = data_train['labels']
            data_base = load_data(file_train_categories_val_phase)
            data_novel = load_data(file_val_categories_val_phase)
            self.data = np.concatenate(
                [self.data, data_novel['data']], axis=0)
            self.data = np.concatenate(
                [self.data, data_base['data']], axis=0)
            self.labels = np.concatenate(
                [self.labels, data_novel['labels']], axis=0)
            self.labels = np.concatenate(
                [self.labels, data_base['labels']], axis=0)

            self.label2ind = buildLabelIndex(self.labels)
            self.labelIds = sorted(self.label2ind.keys())
            self.num_cats = len(self.labelIds)
            self.labelIds_base = self.labelIds
            self.num_cats_base = len(self.labelIds_base)
        elif self.phase=='val' or self.phase=='test':
            if self.phase=='test':
                # load data that will be used for evaluating the recognition
                # accuracy of the base categories.
                data_base = load_data(file_train_categories_test_phase)
                # load data that will be use for evaluating the few-shot recogniton
                # accuracy on the novel categories.
                data_novel = load_data(file_test_categories_test_phase)
            else: # phase=='val'
                # load data that will be used for evaluating the recognition
                # accuracy of the base categories.
                data_base = load_data(file_train_categories_val_phase)
                # load data that will be use for evaluating the few-shot recogniton
                # accuracy on the novel categories.
                data_novel = load_data(file_val_categories_val_phase)

            # self.data = np.concatenate(
            #     [data_base['data'], data_novel['data']], axis=0)
            # self.labels = data_base['labels'] + data_novel['labels']
            self.data = data_novel['data']
            self.labels = data_novel['labels']

            self.label2ind = buildLabelIndex(self.labels)
            self.labelIds = sorted(self.label2ind.keys())
            self.num_cats = len(self.labelIds)

            self.labelIds_base = buildLabelIndex(data_base['labels']).keys()
            self.labelIds_novel = buildLabelIndex(data_novel['labels']).keys()
            self.num_cats_base = len(self.labelIds_base)
          
            self.num_cats_novel = len(self.labelIds_novel)
            intersection = set(self.labelIds_base) & set(self.labelIds_novel)
            assert(len(intersection) == 0)
        else:
            raise ValueError('Not valid phase {0}'.format(self.phase))
        self.targets = self.labels
        mean_pix = [x/255.0 for x in [120.39586422,  115.59361427, 104.54012653]]
        std_pix = [x/255.0 for x in [70.68188272,  68.27635443,  72.54505529]]
        normalize = transforms.Normalize(mean=mean_pix, std=std_pix)

        if (self.phase=='test' or self.phase=='val') or (do_not_use_random_transf==True):
            self.transform = transforms.Compose([
                transforms.RandomCrop(84, padding=8),
                transforms.ToTensor(),
                normalize
            ])
            # self.transform = transforms.Compose([
            #     transforms.Resize((256 , 256)),
            #     transforms.ToTensor(),
            #     # normalize
            # ])
        else:
            self.transform = transforms.Compose([
                transforms.RandomCrop(84, padding=8),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
            
    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index]
        template = Image.open(_MINI_IMAGENET_DATASET_DIR+ 'templates/'+self.phase+'/class_tensor('+str(label)+')')
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # print(label)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
            template = self.transform(template)
        label = torch.tensor(label)
        return img, label, template

    def __len__(self):
        return len(self.data)

    def collate_fn(self,batch):
        x = []
        y = []

        for b in batch:
            x.append(b[0])
            y.append(b[1])
        x = torch.stack(x,dim=0)
        y = torch.stack(y,dim=0)
        return x,y

