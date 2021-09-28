from argparse import ArgumentParser
import sys
import random
from matplotlib import pyplot as plt
import numpy as np
import math
import os
from pathlib import Path

import torch
import torchvision.models as models
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms

sys.path.append('../')
from loader import get_loader, get_data_path

def template_extractor(loader,uniq_labels,extractor):
	feat = torch.tensor([])
	data_x = torch.tensor([])
	all_labels = torch.tensor([]).type(torch.LongTensor)
	for i,batch in enumerate(loader):
		 samples, labels = batch
		 data_x = torch.cat((data_x,samples),dim=0)
		 samples = samples.to(device)
		 feat_batch = extractor(samples)
		 feat_batch = feat_batch.view(samples.shape[0],-1).cpu()
		 feat = torch.cat((feat,feat_batch),dim=0)
		 all_labels = torch.cat((all_labels,labels),dim=0)
	cos = nn.CosineSimilarity(dim=1, eps=1e-6)
	
	template = {}	
	for c in  uniq_labels:
		print(c)
		tmp_x = data_x[all_labels==c]
		f = feat[all_labels==c]

		mu = torch.mean(f,dim =0)
		if(len(mu.shape)<2):
			mu = mu.unsqueeze(0)
		dist = 1 - cos(f,mu)
		# dist = ((f - mu)**2).sum(dim=1)
		_,min_idx = torch.min(dist,dim=0)
		x = tmp_x[min_idx]
		# plt.imshow(transforms.ToPILImage()(x).convert("RGB"))		
		# plt.imshow(transforms.ToPILImage()(x))
		template[c.item()]=transforms.ToPILImage()(x).convert("RGB")

	return template
		
def save_images(uniq_labels,template,root,mode='train'):
	
	for c in uniq_labels:
		if os.path.exists(root+'templates/'):
			if os.path.exists(root+'templates/'+mode+'/'):
				img = template[c.item()]
				img.save(root+'templates/'+mode+'/class_'+str(c.data),'PNG')
			else:
				os.mkdir(root+'templates/'+mode+'/')
				img = template[c.item()]
				img.save(root+'templates/'+mode+'/class_'+str(c.data),'PNG')
		else:		
			os.mkdir(root+'templates')		
			os.mkdir(root+'templates/'+mode+'/')
			img = template[c.item()]
			img.save(root+'templates/'+mode+'/class_'+str(c.data),'PNG')

parser = ArgumentParser(description='prototype selector')
parser.add_argument('--dataset',    type=str,   default='miniimagenet', help='dataset to use [gtsrb, gtsrb2TT100K, belga2flickr, belga2toplogo, miniimagenet]')
parser.add_argument('--img_cols',   type=int,   default=84,  help='resized image width')
parser.add_argument('--img_rows',   type=int,   default=84,  help='resized image height')
parser.add_argument('--seed',   type=int,   default=42,  help='psuedorandomseed')
args =  parser.parse_args()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed )
random.seed(args.seed)

# m = '/home/snag005/Desktop/fs_ood/trial2/models/miniimagenet/pretrain/adam_200_18_lr1e3'
# resnet18 = torch.load(m)
resnet18 = models.resnet18(pretrained=True)
resnet18 = resnet18.to(device)
data_loader = get_loader(args.dataset)
data_path = get_data_path(args.dataset)
# tr_loader = data_loader(root=data_path, resize=(args.img_rows, args.img_cols), mode='train', augment=0)
# val_loader = data_loader(root=data_path, resize=(args.img_rows, args.img_cols), mode='val', augment=0)
# te_loader = data_loader(root=data_path, resize=(args.img_rows, args.img_cols), mode='test', augment=0)
tr_loader = data_loader(phase='train',data_path = data_path,img_size=[args.img_rows,args.img_cols],do_not_use_random_transf=True)
val_loader = data_loader(phase='val',data_path = data_path,img_size=[args.img_rows,args.img_cols])
te_loader = data_loader(phase='test',data_path = data_path,img_size=[args.img_rows,args.img_cols])
label_train = torch.tensor(tr_loader.targets)
label_val = torch.tensor(val_loader.targets)
label_test= torch.tensor(te_loader.targets)
# full_dataset = [tr_loader,val_loader,te_loader]
# final_dataset = torch.utils.data.ConcatDataset(full_dataset)
sampler_tr = torch.utils.data.SequentialSampler(tr_loader)
sampler_val = torch.utils.data.SequentialSampler(val_loader)
sampler_test = torch.utils.data.SequentialSampler(te_loader)
print(tr_loader.__len__(),val_loader.__len__(),te_loader.__len__())
trainloader = DataLoader(tr_loader, batch_size=100, num_workers=2, shuffle=False, collate_fn=tr_loader.collate_fn ,sampler=sampler_tr)
valloader = DataLoader(val_loader, batch_size=100, num_workers=2, shuffle=False, collate_fn=val_loader.collate_fn ,sampler=sampler_val)
testloader = DataLoader(te_loader, batch_size=100, num_workers=2, shuffle=False, collate_fn=te_loader.collate_fn ,sampler=sampler_test)

uniq_labels_train = torch.unique(label_train)
uniq_labels_val = torch.unique(label_val)
uniq_labels_test = torch.unique(label_test)
# print(uniq_labels_train,uniq_labels_val,uniq_labels_test)
extractor = torch.nn.Sequential(*list(resnet18.children())[:-1])
for p in extractor.parameters():
   p.requires_grad = False
extractor.eval()

train_template = template_extractor(trainloader,uniq_labels_train,extractor)
val_template = template_extractor(valloader,uniq_labels_val,extractor)
test_template = template_extractor(testloader,uniq_labels_test,extractor)

root = data_path
save_images(uniq_labels = uniq_labels_train, template=train_template,root=root,mode='train')
save_images(uniq_labels = uniq_labels_val, template=val_template,root=root,mode='val')
save_images(uniq_labels = uniq_labels_test, template=test_template,root=root,mode='test')
print('done')



