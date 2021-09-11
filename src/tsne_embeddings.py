
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
from sklearn import metrics
import scipy.stats as st
from matplotlib import pyplot as plt
# Setup
def plotter(q,path,nm):

  fig, axs = plt.subplots(1,q.shape[0],sharey=True)
  axs = axs.ravel()
  for i in range(0,q.shape[0]):
    img = q[i,:]
    tmp = img.transpose(1,2,0)   
    axs[i].imshow(tmp)
    axs[i].axis('off')
  plt.show() 
  fig.savefig(path+nm+'.pdf') 
  plt.close()
  
  # plt.close()

def emb_extract(test_loader,model,device,opts):
  backbone = opts.backbone
  n_support = opts.k
  n_query = opts.q
  if opts.metric == 'euclidean':
    euc = True
  else:
    euc = False
  wts = opts.weighted_mean
  emh = opts.emb_enhance
  temp = opts.temperature
  with torch.no_grad():
        for i, episode in enumerate(test_loader):   
            # c=c+1
            # print(c)    
            test_x, test_y, proto_sym = episode
            if(len(test_y.shape)>1):
                test_y = torch.squeeze(test_y,dim=1)  
            classes,test_y = class_scaler(test_y,n_support,n_query)
            test_x = test_x.to(device)
            test_y = test_y.to(device)
            proto_sym = proto_sym.to(device)  
            classes = classes.to(device)

            real_proto_k = torch.Tensor().to(device)
            all_sym_protos = torch.Tensor().to(device)
            for k in classes:    
                 Dx_k,Dy_k,Qx_k,Qy_k = extract_episode(test_x[test_y==k,:,:,:],test_y[test_y==k],n_support,device)   
                 emb_support,_,_,_,_ = model(Dx_k)      
                 # _,emb_support,_,_, = model(Dx_k)         
                 emb_support = torch.flatten(emb_support,start_dim=1)
                 symbolic_proto_k = proto_sym[test_y==k,:,:,:] ## question1
                 # templates = torch.cat((templates,symbolic_proto_k[0,:].unsqueeze(dim=0)),dim=0)
                 # embedding of prototype symbol
                 emb_proto_k,_,_,_,_ = model(symbolic_proto_k[0,:,:,:].unsqueeze(dim=0))
                 # _,emb_proto_k,_,_ = model(symbolic_proto_k[0,:,:,:].unsqueeze(dim=0))
                 emb_proto_k = torch.flatten(emb_proto_k,start_dim=1)

                 tmp = proto_rectifier(emb_support,emb_proto_k,euc=euc,wts=wts)
                 real_proto_k = torch.cat((real_proto_k,tmp),dim=0)        
                 symbolic_proto_k = proto_sym[test_y==k,:,:,:]
                 all_sym_protos = torch.cat((all_sym_protos,symbolic_proto_k[0,:,:,:].unsqueeze(dim=0)),dim=0)# C x dim
            all_sym_proto_embs,_,_,_,_ = model(all_sym_protos) 
            # _,all_sym_proto_embs,_,_, = model(all_sym_protos)  
            all_sym_proto_embs = torch.flatten(all_sym_proto_embs,start_dim=1)
            
            support_embedding = torch.Tensor().to(device)
            in_query_embedding = torch.Tensor().to(device)
            in_query_embedding_mod = torch.Tensor().to(device)
            out_query_embedding = torch.Tensor().to(device)
            out_query_embedding_mod = torch.Tensor().to(device)
            
            in_query_recon = torch.Tensor().to(device)
            out_query_recon = torch.Tensor().to(device)
            templates = torch.Tensor().to(device)
            in_query = torch.Tensor().to(device)
            out_query = torch.Tensor().to(device)
            for k in classes:
                 Dx_k,Dy_k,Qx_k,Qy_k = extract_episode(test_x[test_y==k,:,:,:],test_y[test_y==k],n_support,device)   
                 
                 ## embedding of support and reconstruction
                 emb_support,_,_,_,_ = model(Dx_k)      
                 # _,emb_support,_,_, = model(Dx_k)         
                 emb_support = torch.flatten(emb_support,start_dim=1)
                 symbolic_proto_k = proto_sym[test_y==k,:,:,:] ## question1
                 templates = torch.cat((templates,symbolic_proto_k[0,:].unsqueeze(dim=0)),dim=0)  
                 in_query = torch.cat((in_query,Qx_k[:n_query//2]),dim=0) 
                 out_query = torch.cat((out_query,Qx_k[n_query//2:]),dim=0)                     
                 # embedding of prototype symbol
                 # emb_proto_k,_,_,_,_ = model(symbolic_proto_k[0,:,:,:].unsqueeze(dim=0))
                 # # _,emb_proto_k,_,_ = model(symbolic_proto_k[0,:,:,:].unsqueeze(dim=0))
                 # emb_proto_k = torch.flatten(emb_proto_k,start_dim=1)

                 # tmp = proto_rectifier(emb_support,emb_proto_k,euc=euc,wts=wts)
                 # real_proto_k = torch.cat((real_proto_k,tmp),dim=0)# size should be Nc x embedding size, if not check
                 emb_query,_,_,recon_query,_ = model(Qx_k)  
                 # _,emb_query,_,_, = model(Qx_k)                        
                 emb_query = torch.flatten(emb_query,start_dim=1)  
                 emb_query_emh = emb_enhance(emb_query,all_sym_proto_embs,real_proto_k,device,emh=True)

                 support_embedding = torch.cat((support_embedding,emb_support),dim=0)
                 in_query_embedding = torch.cat((in_query_embedding,emb_query[:n_query//2]),dim=0)
                 in_query_embedding_mod = torch.cat((in_query_embedding_mod,emb_query_emh[:n_query//2]),dim=0)
                 in_query_recon = torch.cat((in_query_recon,recon_query[:n_query//2]),dim=0)
                 out_query_embedding = torch.cat((out_query_embedding,emb_query[n_query//2:]),dim=0)
                 out_query_embedding_mod = torch.cat((out_query_embedding_mod,emb_query_emh[n_query//2:]),dim=0)
                 out_query_recon = torch.cat((out_query_recon,recon_query[n_query//2:]),dim=0)
            # for k in classes:
            #     Dx_k,Dy_k,Qx_k,Qy_k = extract_episode(test_x[test_y==k,:,:,:],test_y[test_y==k],n_support,device) 

  support_embedding = support_embedding.detach().cpu().numpy()
  all_sym_proto_embs = all_sym_proto_embs.detach().cpu().numpy()
  in_query_embedding = in_query_embedding.detach().cpu().numpy()
  in_query_embedding_mod = in_query_embedding_mod.detach().cpu().numpy()
  in_query_recon = in_query_recon.detach().cpu().numpy()
  in_query = in_query.detach().cpu().numpy()
  out_query = out_query.detach().cpu().numpy()
  # in_query_recon = np.array(in_query_recon, dtype=np.uint8)
  out_query_embedding = out_query_embedding.detach().cpu().numpy()
  out_query_embedding_mod = out_query_embedding_mod.detach().cpu().numpy()
  out_query_recon = out_query_recon.detach().cpu().numpy()
  # out_query_recon = np.array(out_query_recon, dtype=np.uint8)
  templates = templates.detach().cpu().numpy()
  # templates = np.array(templates, dtype=np.uint8)
  return support_embedding,all_sym_proto_embs,in_query_embedding, in_query_embedding_mod,out_query_embedding,out_query_embedding_mod,in_query_recon,out_query_recon,templates,in_query,out_query
      
args = get_args()
# print(args.recon)
x = 'cuda:0'
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
                      CenterPadding([args.img_rows, args.img_cols])])  # ramdom rotation

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
elif (args.dataset == 'omniglot'):
    tr_loader = data_loader(mode='train', root=data_path, resize=(args.img_rows, args.img_cols))
    te_loader = data_loader(mode='test', root=data_path, resize=(args.img_rows, args.img_cols))
else: 
    tr_loader = data_loader(data_path, args.exp, is_transform=True, split='train', img_size=(args.img_rows, args.img_cols), augmentations=None)
    te_loader = data_loader(data_path, args.exp, is_transform=True, split='test', img_size=(args.img_rows, args.img_cols), augmentations=None)

tr_sampler = ProtoBatchSampler(tr_loader.targets, iterations=args.episodes_per_epoch_train, num_support=args.k, num_query=args.q//2, classes_in=args.n, classes_out=args.n)
te_sampler = ProtoBatchSampler(te_loader.targets, iterations=args.episodes_test, num_support=args.k, num_query=args.q//2, classes_in=args.n, classes_out=args.n)

trainloader = DataLoader(tr_loader, batch_sampler=tr_sampler, num_workers=args.workers, shuffle=False, pin_memory=True)
testloader = DataLoader(te_loader, batch_sampler=te_sampler, num_workers=args.workers, shuffle=False, pin_memory=True)

# final_model,best_model = trainer(model=model,device=device,train_loader=trainloader,val_loader=testloader,opts=args)
m = '/home/snag005/Desktop/fs_ood/trial2/models/'+args.dataset+'/5shot_modelno_augmentation'
path = '/home/snag005/Desktop/fs_ood/trial2/models/'+args.dataset+'/tsne/refocs_new/'
model = torch.load(m)
model = model.to(device)
model.eval()

support,exemplar,in_query,in_query_mod,out_query,out_query_mod,in_query_recon,out_query_recon,templates,in_query,out_query = emb_extract(test_loader=testloader,model=model,device=device,opts=args)


if not os.path.exists(path):
  os.mkdir(path)
plotter(templates,path=path,nm='template')
plotter(in_query,path=path,nm='inq')
plotter(out_query,path=path,nm='outq')
plotter(in_query_recon,path=path,nm='inq_recon')
plotter(out_query_recon,path=path,nm='outq_recon')
  # plt.figure(3)
  # plt.imshow(tmp)
  # plt.show()
  # in_recon = Image.fromarray(inr, 'RGB')
  # out_recon = Image.fromarray(outr, 'RGB')
  # in_recon.save(path+'in_recon'+str(i)+'.jpg')
  # out_recon.save(path+'out_recon'+str(i)+'.jpg')  
  # temp.save(path+'template'+str(i)+'.jpg')
  

# np.save(path+'in_recon.npy', in_query_recon, allow_pickle=True, fix_imports=True)
# np.save(path+'out_recon.npy', out_query_recon, allow_pickle=True, fix_imports=True)
# np.save(path+'templates.npy', templates, allow_pickle=True, fix_imports=True)
# np.save(path+'support_emb.npy', support, allow_pickle=True, fix_imports=True)
# np.save(path+'exemplar_emb.npy', exemplar, allow_pickle=True, fix_imports=True)
# np.save(path+'in_query_emb.npy', in_query, allow_pickle=True, fix_imports=True)
# np.save(path+'mod_in_query_emb.npy', in_query_mod, allow_pickle=True, fix_imports=True)
# np.save(path+'out_query_emb.npy', out_query, allow_pickle=True, fix_imports=True)
# np.save(path+'mod_out_query_emb.npy', out_query_mod, allow_pickle=True, fix_imports=True)

print('done')