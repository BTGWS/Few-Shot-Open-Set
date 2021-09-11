
import os
import random
from matplotlib import pyplot as plt
import numpy as np
import math
import os
from pathlib import Path
from PIL import Image
import torch
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
from parser import get_args
from loader import get_loader, get_data_path
from datasets.augmentations import *
from datasets.sampler import *
from models.model import *
from train import get_trainer,get_tester

# Setup

args = get_args()
# print(args.recon)
x = input('enter cuda #: ')
device = torch.device(x if torch.cuda.is_available() else 'cpu')

random.seed(args.seed)
torch.manual_seed(args.seed)

if args.stn :
  stn = [[200,300,200], None, [150, 150, 150]]
else:
  stn = [None, None, None]

# 02 rotation + flip augmentation option
# Setup Augmentations
data_aug_tr= Compose([Scale(args.img_cols), # resize longer side of an image to the defined size
                      CenterPadding([args.img_rows, args.img_cols]), # zero pad remaining regions
                      RandomHorizontallyFlip(), # random horizontal flip
                      RandomRotate(180)])  # ramdom rotation

data_aug_te= Compose([Scale(args.img_cols), 
                     CenterPadding([args.img_rows, args.img_cols])])

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Data
data_loader = get_loader(args.dataset)
data_path = get_data_path(args.dataset)
trainer = get_trainer(args.trainer_type)
tester = get_tester(args.tester_type)
if(args.dataset == 'miniimagenet'):
   tr_loader = data_loader(root=data_path, resize=(args.img_rows, args.img_cols), mode='train', augment=1)
   te_loader = data_loader(root=data_path, resize=(args.img_rows, args.img_cols), mode='test', augment=2)
else: 
    tr_loader = data_loader(data_path, args.exp, is_transform=True, split='train', img_size=(args.img_rows, args.img_cols), augmentations=data_aug_tr)
    te_loader = data_loader(data_path, args.exp, is_transform=True, split='test', img_size=(args.img_rows, args.img_cols), augmentations=data_aug_te)

tr_sampler = ProtoBatchSampler(tr_loader.targets, iterations=args.episodes_per_epoch_train, num_support=args.n, num_query=args.q//2, classes_in=args.k_train, classes_out=args.k_test)
te_sampler = ProtoBatchSampler(te_loader.targets, iterations=args.episodes_per_epoch_test, num_support=args.n, num_query=args.q//2, classes_in=args.k_test, classes_out=args.k_test)

trainloader = DataLoader(tr_loader, batch_sampler=tr_sampler, num_workers=args.workers, shuffle=False, pin_memory=True)
testloader = DataLoader(te_loader, batch_sampler=te_sampler, num_workers=args.workers, shuffle=False, pin_memory=True)
# trainloader = DataLoader(tr_loader, batch_size=args.batch_size, num_workers=args.workers, shuffle=True,  pin_memory=True)

data = next(iter(trainloader))

model = Encoder(backbone = args.backbone,mlp_inp_dim=args.mlp_inp_dim,mlp_hid_layers=args.mlp_hid_layers,inp_channels=data[0].shape[1],
        hid_dim=args.conv_hid_layers,conv_filters=args.enc_conv_filters,linear = args.linear_embedding,linear_inp_siz=args.linear_embedding_size,
        stn =stn,z_dim=args.z_dim,stride=args.enc_stride)
# print(model)
final_model,best_model = trainer(model=model,device=device,train_loader=trainloader,val_loader=testloader,opts=args)

final_model.eval()
best_model.eval()

Accuracy_mean1,Au_ROC_mean1,Accuracy_std1,Au_ROC_std1 = tester(model=final_model,device=device,test_loader=testloader,opts=args)
Accuracy_mean2,Au_ROC2_mean2,Accuracy_std2,Au_ROC_std2= tester(model=best_model,device=device,test_loader=testloader,opts=args)
print('For %s %d shot final model test accuracy = %.3f + %.3f and AuROC =  %.3f + %.3f'%(args.dataset,args.n,Accuracy_mean1,Accuracy_std1,Au_ROC_mean1,Au_ROC_std1))
print('For %s %d shot best validation model test accuracy = %.3f + %.3f and AuROC =  %.3f + %.3f'%(args.dataset,args.n,Accuracy_mean2,Accuracy_std2,Au_ROC2_mean2,Au_ROC_std2))

mdl = input('Enter which model to save 1 or 2:')
if int(mdl) == 1:
  save_model = final_model
else:
  save_model = best_model

save_model_path = '/home/snag005/Desktop/fs_ood/trial2/models/'+args.dataset+'/'
mdl_no = args.mdl_no
if os.path.exists(save_model_path):
  mdl_path = save_model_path +str(args.n)+'shot_model'+str(mdl_no)
    # str(args.img_cols)+'x'+str(args.img_cols)+'_'+str(args.n)+'shot_model'+str(x))
  torch.save(save_model,mdl_path)
else:
  os.mkdir(save_model_path )
  mdl_path = save_model_path +str(args.n)+'shot_model'+str(mdl_no)
  torch.save(save_model,mdl_path)