from torchvision import transforms
import torchvision
import torch
import torchvision.models as models
from argparse import ArgumentParser
import os
import random
import numpy as np
from loader.omniglot import OmniglotDataset
from torch.utils.data import DataLoader

from torch import nn
from torch.nn import functional as F
from PIL import Image

from sklearn import metrics
def template_extractor(loader,uniq_labels,extractor):
  feat = torch.tensor([])
  data_x = torch.tensor([])
  all_labels = torch.tensor([]).type(torch.LongTensor)
  for i,batch in enumerate(loader):
     samples, labels = batch
     data_x = torch.cat((data_x,samples),dim=0)
     samples = samples.to(device)
     feat_batch = extractor(samples)
     feat_batch = feat_batch.view(samples.shape[0],512).cpu()
     feat = torch.cat((feat,feat_batch),dim=0)
     all_labels = torch.cat((all_labels,labels),dim=0)
  cos = nn.CosineSimilarity(dim=1, eps=1e-6)
  
  template = {} 
  for c in  uniq_labels:
    tmp_x = data_x[all_labels==c]
    f = feat[all_labels==c]

    mu = torch.mean(f,dim =0)
    if(len(mu.shape)<2):
      mu = mu.unsqueeze(0)
    # dist = cos(f,mu)
    dist = ((f - mu)**2).sum(dim=1)
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
parser.add_argument('--img_cols',   type=int,   default=28,  help='resized image width')
parser.add_argument('--img_rows',   type=int,   default=28,  help='resized image height')
parser.add_argument('--seed', type=int,   default=42, help='Random seed')
parser.add_argument('--epoch', type=int, default=30, help='Number of epochs (default: 30)')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate (default: 1e-4)')
args = parser.parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)
transform = transforms.Compose(
    [transforms.Resize((args.img_cols ,args.img_rows)),
     transforms.ToTensor(),
     transforms.Normalize((0.1307,), (0.3081,))])
emnist_train = torchvision.datasets.EMNIST('/home/snag005/Desktop/fs_ood/trial2/datasets/db/EMNIST/',split='letters',train=True,download=True,transform=transform)
emnist_test = torchvision.datasets.EMNIST('/home/snag005/Desktop/fs_ood/trial2/datasets/db/EMNIST/',split='letters',train=False,download=True,transform=transform)

trainloader = DataLoader(emnist_train,
                        batch_size=1024,
                        shuffle=True,pin_memory=True,
                        num_workers=4)

testloader = torch.utils.data.DataLoader(emnist_test,
                                          batch_size=1024,
                                          shuffle=True,pin_memory=True,
                                          num_workers=4)
  
device = torch.device('cuda:0,1' if torch.cuda.is_available() else 'cpu')
resnet18 = models.resnet18(pretrained=False)
resnet18.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,bias=False)
resnet18 = resnet18.to(device)
loss_class = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(resnet18.parameters(), lr=args.lr)
data = next(iter(trainloader))

for epoch in range(1,args.epoch+1):
  for i,batch in enumerate(trainloader):
    optimizer.zero_grad()
    train_x, train_y = batch
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    logits = resnet18(train_x)
    loss = loss_class(logits,train_y)
    loss.backward()
    optimizer.step()
    print('[%d/%d] loss=%.3f'%(epoch,args.epoch,loss))

pred_full= torch.LongTensor().to(device)
gt = torch.LongTensor().to(device)
counter = 0
for i,batch in enumerate(testloader):
  counter=counter+1
  test_x, test_y = batch
  test_x = test_x.to(device)
  test_y = test_y.to(device)
  pred = resnet18(test_x)
  _,Hyp = torch.max(pred, dim=1)
  pred_full = torch.cat((pred_full,Hyp),dim=0)
  gt = torch.cat((gt,test_y),dim=0)
acc = metrics.accuracy_score(test_y.cpu().data,Hyp.cpu().data )
print('Accuracy=%.4f'%((acc)))
train_transforms = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
                                                  torchvision.transforms.Resize((args.img_cols,args.img_rows),
                                                  interpolation=Image.BICUBIC),torchvision.transforms.ToTensor()])
test_transforms = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
                                                  torchvision.transforms.Resize((args.img_cols,args.img_rows),
                                                  interpolation=Image.BICUBIC),torchvision.transforms.ToTensor()])
tr_loader = OmniglotDataset(mode='train',transform=train_transforms)
te_loader = OmniglotDataset(mode='test', transform = test_transforms)
val_loader = OmniglotDataset(mode='val', transform = test_transforms)
label_train = torch.tensor(tr_loader.y)
label_val = torch.tensor(val_loader.y)
label_test= torch.tensor(te_loader.y)
# full_dataset = [tr_loader,val_loader,te_loader]
# final_dataset = torch.utils.data.ConcatDataset(full_dataset)
trainloader = DataLoader(tr_loader, batch_size=1024, num_workers=4, shuffle=False, pin_memory=True)
valloader = DataLoader(val_loader, batch_size=1024, num_workers=4, shuffle=False, pin_memory=True)
testloader = DataLoader(te_loader, batch_size=1024, num_workers=4, shuffle=False, pin_memory=True)

uniq_labels_train = torch.unique(label_train)
uniq_labels_val = torch.unique(label_val)
uniq_labels_test = torch.unique(label_test)

extractor = torch.nn.Sequential(*list(resnet18.children())[:-1])
for p in extractor.parameters():
   p.requires_grad = False
extractor.eval()

train_template = template_extractor(trainloader,uniq_labels_train,extractor)
val_template = template_extractor(valloader,uniq_labels_val,extractor)
test_template = template_extractor(testloader,uniq_labels_test,extractor)

root = '/home/snag005/Desktop/fs_ood/trial2/datasets/db/omniglot/'
save_images(uniq_labels = uniq_labels_train, template=train_template,root=root,mode='train')
save_images(uniq_labels = uniq_labels_val, template=val_template,root=root,mode='val')
save_images(uniq_labels = uniq_labels_test, template=test_template,root=root,mode='test')
print('done')