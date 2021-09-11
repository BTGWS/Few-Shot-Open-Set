from __future__ import print_function
import torch.utils.data as data
import numpy as np
import errno
import os
from PIL import Image
import torch
import shutil
import pickle
from torchvision import datasets
from torchvision.transforms import transforms as T
from datasets.augmentations import *
'''
Inspired by https://github.com/pytorch/vision/pull/46
'''


class MiniImagenetDataset(data.Dataset):
    def __init__(self, resize, mode='train', root='../dataset/mini-imagenet', augment=1):
        '''
        The items are (filename,category). The index of all the categories can be found in self.idx_classes
        Args:
        - root: the directory where the dataset will be stored
        - transform: how to transform the input
        - target_transform: how to transform the target
        - download: need to download the dataset
        '''
        # self.mean_pix = np.array([x/255.0 for x in [120.39586422, 115.59361427, 104.54012653]])        
        # self.std_pix = np.array([x/255.0 for x in [70.68188272, 68.27635443, 72.54505529]])
        self.mean_pix = np.array([0.485, 0.456, 0.406])
        self.std_pix = np.array([0.229, 0.224, 0.225])
        self.resize = resize
        self.augment = augment
        padding = 8  # im_size = 84
        if augment == 0:
            self.transform = T.Compose([
                T.Resize((self.resize[0] , self.resize[1] )),
                T.CenterCrop(self.resize),
                T.ToTensor()
                # T.Normalize(mean=mean_pix, std=std_pix)
            ])
        elif augment == 1:
            self.transform = T.Compose([
                T.Resize((self.resize[0] ,self.resize[1])),
                # T.RandomCrop(self.resize),
                T.CenterCrop(self.resize),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=.1, contrast=.1, saturation=.1, hue=.1),                
                T.ToTensor(),
                # T.Normalize(mean=self.mean_pix, std=self.std_pix)
            ])

        elif augment == 2:
            self.transform = T.Compose([
                T.Resize((self.resize[0], self.resize[1])),
                T.CenterCrop(self.resize),
                # T.RandomCrop(self.resize),
                # T.ColorJitter(brightness=.1, contrast=.1, saturation=.1, hue=.1), 
                T.ToTensor(),
                # T.Normalize(mean=self.mean_pix, std=self.std_pix)
                # T.Lambda(lambda crops: [T.ToTensor()(crop) for crop in crops]),
                # T.Lambda(lambda crops: torch.stack([T.Normalize(mean=mean_pix, std=std_pix)(crop) for crop in crops]))
            ])

        elif augment == 3:
            self.augmentations = Compose([Scale(self.resize[1]), # resize longer side of an image to the defined size
                          CenterPadding([self.resize[0], self.resize[1]]), # zero pad remaining regions
                          RandomHorizontallyFlip(), # random horizontal flip
                          RandomRotate(180)])  # ramdom rotation
            self.transform = T.Compose([T.Resize((self.resize[0],self.resize[1])),
                            T.ToTensor(),
                            T.Normalize(mean=self.mean_pix, std=self.std_pix)
                            ])
        elif augment == 4:
            self.augmentations = Compose([Scale(self.resize[1]), # resize longer side of an image to the defined size
                          CenterPadding([self.resize[0], self.resize[1]]) # zero pad remaining regions
                          ])
            self.transform = T.Compose([T.Resize((self.resize[0], self.resize[1])),
                            T.ToTensor(),
                            T.Normalize(mean=self.mean_pix, std=self.std_pix)
                            ])
        else:
            raise NameError('Augment mode {} not implemented.'.format(augment))
        self.mode = mode
        super(MiniImagenetDataset, self).__init__()
        self.root = root
        # self.transform = transform
        # self.target_transform = target_transform

        if not self._check_exists():
            raise RuntimeError('Dataset not found. Follow instructions to download mini-imagenet.')

        pickle_file = os.path.join(self.root, 'mini-imagenet-cache-' + mode + '.pkl')
        f = open(pickle_file, 'rb')
        self.data = pickle.load(f,encoding='bytes')
        # self.data = pil_loader(pickle_file)
        self.x = [Image.fromarray(x)  for x in self.data['image_data']]
        # self.x = [np.transpose(x, (2, 0, 1)) for x in self.data['image_data']]
        # self.x = [torch.FloatTensor(x) for x in self.x]
        self.y = [-1 for _ in range(len(self.x))]
        class_idx = index_classes(mode,self.data['class_dict'].keys())
        for class_name, idxs in self.data['class_dict'].items():
            for idx in idxs:
                self.y[idx] = class_idx[class_name]
        self.targets = self.y
        # self.all_items

    def __getitem__(self, idx):
        template = Image.open(self.root + 'random_templates/'+self.mode+'/class_tensor('+str(self.y[idx])+')')
        
        x = self.x[idx]
        if self.augment == 3 or self.augment == 4 :
            x = np.array(x, dtype=np.uint8)
            template = np.array(template, dtype=np.uint8)
            x,template = self.augmentations(x,template)
            # x = (x - self.mean_pix)/self.std_pix
            # template = (x - self.mean_pix)/self.std_pix
            x = Image.fromarray(x.astype(np.uint8))
            template = Image.fromarray(template.astype(np.uint8))
        x = self.transform(x)
        template = self.transform(template)
        return x, self.y[idx], template

    def __len__(self):
        return len(self.x)

    def _check_exists(self):
        return os.path.exists(self.root)

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def index_classes(mode,items):
    idx = {}
    for i in items:
        if (not i in idx):
            idx[i] = len(idx)
    print("== Dataset:{} Found %d classes".format(mode) % len(idx))
    return idx



