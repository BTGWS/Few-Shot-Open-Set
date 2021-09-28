
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
# from datasets.few_shot_loader import FewShotDataloader
# from tqdm import tqdm
from models.model import *
from train import get_trainer,get_tester
from train.train_utils import *
from train.pretrainer import *
# Setup

args = get_args()
# print(args.recon)

device = torch.device('cuda:'+str(args.rank) if torch.cuda.is_available() else 'cpu')
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed )
random.seed(args.seed)

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
   tr_loader = data_loader(phase='train',data_path=data_path,img_size=[args.img_cols,args.img_rows])   #data_loader(root=data_path, resize=(args.img_rows, args.img_cols), mode='train', augment=1)
   val_loader = data_loader(phase='val',data_path=data_path,img_size=[args.img_cols,args.img_rows])   #data_loader(root=data_path, resize=(args.img_rows, args.img_cols), mode='val', augment=2)
   te_loader = data_loader(phase='test',data_path=data_path,img_size=[args.img_cols,args.img_rows])      #data_loader(root=data_path, resize=(args.img_rows, args.img_cols), mode='test', augment=2)
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

# if args.dataset == 'miniimagenet':
tr_sampler = ProtoBatchSampler(tr_loader.targets, iterations=args.episodes_per_epoch_train, num_support=args.k, num_query=args.q//2, classes_in=args.n, classes_out=args.n)
# val_sampler = ProtoBatchSampler(val_loader.targets, iterations=args.episodes_val, num_support=args.k, num_query=args.q//2, classes_in=args.n, classes_out=args.n)
te_sampler = ProtoBatchSampler(te_loader.targets, iterations=args.episodes_test, num_support=args.k, num_query=args.q//2, classes_in=args.n, classes_out=args.n)

trainloader = DataLoader(tr_loader, batch_sampler=tr_sampler, num_workers=args.workers, shuffle=False, pin_memory=True)
# valloader = DataLoader(val_loader, batch_sampler=val_sampler, num_workers=args.workers, shuffle=False, pin_memory=True)
testloader = DataLoader(te_loader, batch_sampler=te_sampler, num_workers=args.workers, shuffle=False, pin_memory=True)
# else:
#   trainloader = FewShotDataloader(dataset=tr_loader,
#         nKnovel=args.n,
#         nKbase=0,
#         nExemplars=args.k, # num training examples per novel category
#         nTestNovel=args.n * (args.q//2), # num test examples for all the novel categories
#         nTestBase=0, # num test examples for all the base categories
#         batch_size=1,
#         num_workers=0, #args.workers,
#         epoch_size=args.episodes_per_epoch_train )# num of episodes per epoch
#   testloader = FewShotDataloader(dataset=te_loader,
#         nKnovel=args.n,
#         nKbase=0,
#         nExemplars=args.k, # num training examples per novel category
#         nTestNovel=args.n * (args.q//2), # num test examples for all the novel categories
#         nTestBase=0, # num test examples for all the base categories
#         batch_size=1,
#         num_workers=0,
#         epoch_size=args.episodes_test )

# data = next(iter(trainloader))

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

if args.trainer_type == 'proto' or args.tester_type =='proto':
  model =  Encoder(backbone = args.backbone,mlp_inp_dim=args.mlp_inp_dim,mlp_hid_layers=args.mlp_hid_layers,inp_channels=3,
        hid_dim=args.conv_hid_layers,conv_filters=args.enc_conv_filters,linear = args.linear_embedding,linear_inp_siz=args.linear_embedding_size,
        stn =stn,z_dim=args.z_dim,stride=args.enc_stride,branch=True,temperature=args.temperature)
else:
  model = Proto_ND(ab_inp_size=ab_inp_size,backbone = args.backbone,mlp_inp_dim=args.mlp_inp_dim,mlp_hid_layers=args.mlp_hid_layers,inp_channels=3, 
      hid_dim=args.conv_hid_layers,enc_conv_filters=args.enc_conv_filters,dec_conv_filters=args.dec_conv_filters,linear = args.linear_embedding,
  		linear_inp_siz=args.linear_embedding_size,stride=args.enc_stride,outsize=[args.img_cols, args.img_rows], ab_layers = args.ab_module_layers,
      z_dim=args.z_dim, stn=stn,temperature=args.temperature)
# model = nn.DataParallel(model)

# if args.pretrain:
#   m = '/home/snag005/Desktop/fs_ood/trial2/models/miniimagenet/pretrain/adam_200_18_lr1e3'
#   net = torch.load(m)
#   pretrained_dict = net.state_dict()
# #     model.enc_module.load_state_dict(pretrained_dict)
#   tmp = list(pretrained_dict.items())[:-1]
  
#   mod_dict = model.enc_module.state_dict()
#   count=0
#   for key,value in mod_dict.items():
#       layer_name,weights=tmp[count]      
#       mod_dict[key]=weights
#       count+=1
#   model.enc_module.load_state_dict(mod_dict)
  # tr_loader = data_loader(root=data_path, resize=(args.img_rows, args.img_cols), mode='train', augment=0)
  # trainloader = DataLoader(tr_loader, batch_sampler=tr_sampler, num_workers=args.workers, shuffle=False, pin_memory=True)

torch.autograd.set_detect_anomaly(True)
final_model_state_dict,best_model_state_dict = trainer(model=model,device=device,train_loader=trainloader,val_loader=testloader,tester=tester,opts=args)

for k, v in final_model_state_dict.items():
    final_model_state_dict[k] = v.cpu()

for k, v in best_model_state_dict.items():
    best_model_state_dict[k] = v.cpu()

save_model_path = args.output_dir+args.dataset+'/'+str(args.k)+'shot_model_'+args.model_id+'/'
if not os.path.exists(save_model_path):
  os.mkdir(save_model_path)

torch.save({
            'best_model_state_dict': best_model_state_dict,
            'final_model_state_dict': final_model_state_dict
            }, save_model_path+'models.pth')
# m = '/home/snag005/Desktop/fs_ood/trial2/models/miniimagenet/1shot_modeltemp_1000'
# best_model= torch.load(m)
# best_model.to(device)


# # final_model = model
# final_model.eval()
# best_model.eval()
# # _,_,_,recon_query,_= best_model(data[0].to(device))
# # recon = recon_query.detach().cpu().numpy()
# # np.save('/home/snag005/Desktop/fs_ood/trial2/'+'recon.npy', recon, allow_pickle=True, fix_imports=True)
# Accuracy_mean1,Au_ROC_mean1,Accuracy_std1,Au_ROC_std1 = tester(model=final_model,device=device,test_loader=testloader,opts=args)
# Accuracy_mean2,Au_ROC2_mean2,Accuracy_std2,Au_ROC_std2= tester(model=best_model,device=device,test_loader=testloader,opts=args)
# print('For %s %d shot final model test accuracy = %.6f + %.6f and AuROC =  %.6f + %.6f'%(args.dataset,args.k,Accuracy_mean1,Accuracy_std1,Au_ROC_mean1,Au_ROC_std1))
# print('For %s %d shot best validation model test accuracy = %.6f + %.6f and AuROC =  %.6f + %.6f'%(args.dataset,args.k,Accuracy_mean2,Accuracy_std2,Au_ROC2_mean2,Au_ROC_std2))

# mdl = input('Press 1 to save final_model and 2 for best_validation_model: ')
# if int(mdl) == 1:
#   save_model = final_model
# else:
#   save_model = best_model

# save_model_path = '/home/eegrad/snag/Desktop/fs_ood/src/models/output/'+args.dataset+'/'
# mdl_no = args.mdl_no
# if os.path.exists(save_model_path):
#   mdl_path = save_model_path +str(args.k)+'shot_model'+str(mdl_no)
#     # str(args.img_cols)+'x'+str(args.img_cols)+'_'+str(args.n)+'shot_model'+str(x))
#   torch.save(save_model.cpu(),mdl_path)
# else:
#   os.mkdir(save_model_path )
#   mdl_path = save_model_path +str(args.k)+'shot_model'+str(mdl_no)
#   torch.save(save_model.cpu(),mdl_path)