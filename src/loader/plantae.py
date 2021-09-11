import torch
from PIL import Image
import json
import numpy as np
import torch.utils.data as data
from torchvision import datasets
from torchvision.transforms import transforms as T
import os
import random

def default_loader(path):
    return Image.open(path).convert('RGB')

class plantae_Loader(data.Dataset):

    def __init__(self, root,resize,mode= 'train', transform=1, target_transform=None,
                 loader=default_loader):

        # assumes classes and im_ids are in same order

        # load annotations
        # print('Loading annotations from: ' + os.path.basename(ann_file))
        
        if mode == 'train':
          ann_file = root + '/base.json'
        elif mode == 'test':
          ann_file = root + '/novel.json'
        else:
          ann_file = root + '/val.json'

        self.mean_pix = np.array([0.485, 0.456, 0.406])
        self.std_pix = np.array([0.229, 0.224, 0.225])
        self.resize = resize

        if transform == 1:
            self.transform = T.Compose([
                T.Resize((self.resize[0] ,self.resize[1])),
                # T.RandomCrop(self.resize),
                T.CenterCrop(self.resize),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=.1, contrast=.1, saturation=.1, hue=.1),                
                T.ToTensor(),
                T.Normalize(mean=self.mean_pix, std=self.std_pix)
            ])

        elif transform == 2:
            self.transform = T.Compose([
                T.Resize((self.resize[0], self.resize[1])),
                T.CenterCrop(self.resize),
                # T.RandomCrop(self.resize),
                # T.ColorJitter(brightness=.1, contrast=.1, saturation=.1, hue=.1), 
                T.ToTensor(),
                T.Normalize(mean=self.mean_pix, std=self.std_pix)
                # T.Lambda(lambda crops: [T.ToTensor()(crop) for crop in crops]),
                # T.Lambda(lambda crops: torch.stack([T.Normalize(mean=mean_pix, std=std_pix)(crop) for crop in crops]))
            ])

        with open(ann_file) as data_file:
            ann_data = json.load(data_file)

        # set up the filenames and annotations
        imgs = [aa for aa in ann_data['image_names']]
        im_ids = [aa for aa in ann_data['label_names']]

        if 'image_labels' in ann_data.keys():
            # if we have class labels
            classes = [aa for aa in ann_data['image_labels']]
        else:
            # otherwise dont have class info so set to 0
            classes = [0]*len(im_ids)

        idx_to_class = [ cc for cc in ann_data['label_names']]

        print('\t' + str(len(imgs)) + ' images')
        print('\t' + str(len(np.unique(classes))) + ' classes')

        self.ids = im_ids
        self.root = root
        self.mode = mode
        self.imgs = imgs
        self.targets = classes
        self.idx_to_class = idx_to_class
        
        self.target_transform = target_transform
        self.loader = loader
        super(plantae_Loader, self).__init__()

    def __getitem__(self, index):
        path = self.imgs[index]
        target = self.targets[index]
        # im_id = self.ids[index]
        img = self.loader(path)
        
        template = self.loader(self.root + 'templates/'+self.mode+'/class_tensor('+str(target)+')')
        
        if self.transform is not None:
            img = self.transform(img)
            template = self.transform(template)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, template

    def __len__(self):
        return len(self.imgs)


# identity = lambda x:x

# class Plantae:
#   def __init__(self, data_file,resize,batch_size=32):
#     self.mean_pix = np.array([0.485, 0.456, 0.406])
#     self.std_pix = np.array([0.229, 0.224, 0.225])
#     self.resize = resize
#     transform = transforms.Compose([
#                 transforms.Resize((self.resize[0] , self.resize[1] )),
#                 transforms.CenterCrop(self.resize),
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=self.mean_pix, std=self.std_pix)
#             ])
#     with open(data_file, 'r') as f:
#       self.meta = json.load(f)

#     self.cl_list = np.unique(self.meta['image_labels']).tolist()

#     self.sub_meta = {}
#     for cl in self.cl_list:
#       self.sub_meta[cl] = []
#     self.targets = []
#     for x,y in zip(self.meta['image_names'],self.meta['image_labels']):
#       self.targets.append(y)
#       self.sub_meta[y].append(x)

#     self.sub_dataloader = []
#     sub_data_loader_params = dict(batch_size = batch_size,
#         shuffle = True,
#         num_workers = 0, #use main thread only or may receive multiple batches
#         pin_memory = False)
#     for cl in self.cl_list:
#       self.sub_dataset = SubDataset(self.sub_meta[cl], cl, transform = transform )
#       self.sub_dataloader.append( torch.utils.data.DataLoader(self.sub_dataset, **sub_data_loader_params) )

#   def __getitem__(self,i):
#     x, y = next(iter(self.sub_dataset[i]))

#   def __len__(self):
#     return len(self.cl_list)

# class SubDataset:
#   def __init__(self, sub_meta, cl, transform=transforms.ToTensor(), target_transform=identity, min_size=50):
#     self.sub_meta = sub_meta
#     self.cl = cl
#     self.transform = transform
#     self.target_transform = target_transform
#     if len(self.sub_meta) < min_size:
#       idxs = [i % len(self.sub_meta) for i in range(min_size)]
#       self.sub_meta = np.array(self.sub_meta)[idxs].tolist()

#   def __getitem__(self,i):
#     image_path = os.path.join( self.sub_meta[i])
#     img = Image.open(image_path).convert('RGB')
#     img = self.transform(img)
#     targets = self.target_transform(self.cl)
#     return img, targets

#   def __len__(self):
    # return len(self.sub_meta)