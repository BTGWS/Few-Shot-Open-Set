
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
from train.train_utils import *
from train.pretrainer import *
# Setup

args = get_args()
# print(args.recon)
x = input('enter gpu id: ')

device = torch.device('cuda:'+str(x) if torch.cuda.is_available() else 'cpu')
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
   val_loader = data_loader(root=data_path, resize=(args.img_rows, args.img_cols), mode='val', augment=2)
   te_loader = data_loader(root=data_path, resize=(args.img_rows, args.img_cols), mode='test', augment=2)
elif(args.dataset == 'plantae'):
   tr_loader = data_loader(root=data_path, resize=(args.img_rows, args.img_cols),mode= 'train', transform=1)
   val_loader = data_loader(root=data_path, resize=(args.img_rows, args.img_cols),mode= 'val', transform=2)
   te_loader = data_loader(root=data_path, resize=(args.img_rows, args.img_cols),mode= 'test', transform=2)
elif (args.dataset == 'omniglot'):
    tr_loader = data_loader(mode='train', root=data_path, resize=(args.img_rows, args.img_cols))
    val_loader = data_loader(mode='val', root=data_path, resize=(args.img_rows, args.img_cols))
    te_loader = data_loader(mode='test', root=data_path, resize=(args.img_rows, args.img_cols))
else: 
    tr_loader = data_loader(data_path, args.exp, is_transform=True, split='train', img_size=(args.img_rows, args.img_cols), augmentations=data_aug_tr)    
    val_loader = data_loader(data_path, args.exp, is_transform=True, split='val', img_size=(args.img_rows, args.img_cols), augmentations=data_aug_te)
    te_loader = data_loader(data_path, args.exp, is_transform=True, split='test', img_size=(args.img_rows, args.img_cols), augmentations=data_aug_te)
tr_sampler = ProtoBatchSampler(tr_loader.targets, iterations=args.episodes_per_epoch_train, num_support=args.k, num_query=args.q//2, classes_in=args.n, classes_out=args.n)
# val_sampler = ProtoBatchSampler(val_loader.targets, iterations=args.episodes_val, num_support=args.k, num_query=args.q//2, classes_in=args.n, classes_out=args.n)
te_sampler = ProtoBatchSampler(te_loader.targets, iterations=args.episodes_test, num_support=args.k, num_query=args.q//2, classes_in=args.n, classes_out=args.n)

trainloader = DataLoader(tr_loader, batch_sampler=tr_sampler, num_workers=args.workers, shuffle=False, pin_memory=True)
# valloader = DataLoader(val_loader, batch_sampler=val_sampler, num_workers=args.workers, shuffle=False, pin_memory=True)
testloader = DataLoader(te_loader, batch_sampler=te_sampler, num_workers=args.workers, shuffle=False, pin_memory=True)

data = next(iter(trainloader))

if(args.backbone == 'MLP'):
  ab_inp_size = args.mlp_hid_layers[-1] + 2*args.n
elif args.backbone == 'conv_layers' and not args.linear_embedding :
  enc =  Encoder(inp_channels=data[0].shape[1],hid_dim=args.conv_hid_layers,conv_filters=args.enc_conv_filters,stride=args.enc_stride,stn=stn)
  data_emb,_,_,_ = enc(data[0])
  data_emb = torch.flatten(data_emb,start_dim=1)
  ab_inp_size = data_emb.shape[1]+2*args.n
elif args.trainer_type == 'no_recon' or args.trainer_type == 'no_clf' or args.trainer_type == 'bifurcated_no_clf':
  ab_inp_size = args.z_dim + args.n
elif args.trainer_type == 'no_embedding' or args.trainer_type =='bifurcated_no_embedding':
  ab_inp_size = 2*args.n
else:
  ab_inp_size = args.z_dim + 2*args.n


model =  Encoder(backbone = args.backbone,mlp_inp_dim=args.mlp_inp_dim,mlp_hid_layers=args.mlp_hid_layers,inp_channels=data[0].shape[1],
      hid_dim=args.conv_hid_layers,conv_filters=args.enc_conv_filters,linear = args.linear_embedding,linear_inp_siz=args.linear_embedding_size,
      stn =stn,z_dim=args.z_dim,stride=args.enc_stride,branch=True)
# model = nn.DataParallel(model)


# net = pre_model(model.enc_module,args.z_dim,ncls=64)
# print(net)
# print('===============================')
# print(list(net.children())[:-1])
pepoch = args.epoch
plr = args.lr
save_model = pretrain(tr_loader,model,device,args.z_dim,pepoch,args.schedular,lr=plr,bsz=256,ncls=64)


save_model_path = '/home/snag005/Desktop/fs_ood/trial2/models/'+args.dataset+'/pretrain/'
mdl_no = args.mdl_no
if os.path.exists(save_model_path):
  mdl_path = save_model_path +str(mdl_no)
    # str(args.img_cols)+'x'+str(args.img_cols)+'_'+str(args.n)+'shot_model'+str(x))
  torch.save(save_model.cpu(),mdl_path)
else:
  os.mkdir(save_model_path )
  mdl_path = save_model_path + str(mdl_no)
  torch.save(save_model.cpu(),mdl_path)