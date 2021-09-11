import torch
import torch.nn as nn
import torchvision
import sys
import numpy as np
from train.train_utils import *
from models.model import *
# from train.tester import *
from torch.optim.lr_scheduler import MultiStepLR

from torch.utils.data import DataLoader

class pre_model(nn.Module):
    def __init__(self, encoder, z_dim,ncls):
        super(pre_model, self).__init__()
        self.enc = encoder
        self.fc = nn.Sequential(nn.ReLU(),nn.Linear(z_dim,ncls))
    def forward(self,x):
        x,_,_,_ = self.enc(x)
        x = self.fc(x)
        return x

def pretrain(dataset,model,device,z_dim,epoch,sch,lr,bsz,ncls=64):
    loader  = DataLoader(dataset, batch_size=bsz, num_workers=4, shuffle=True)
    net = pre_model(model,z_dim,ncls)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=5e-4)
    
    scheduler = MultiStepLR(optimizer, milestones=sch, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    train_accuracy = []
    train_loss = []
    for e in range(epoch):  #
        running_loss = 0.0
        test_min_acc = 0
        total = 0
        correct = 0
        for i, data in enumerate(loader, 0):
            inputs,labels,sym = data
            optimizer.zero_grad()
            outputs = net(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            #scheduler.step()
            # print statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            acc = predicted.eq(labels.to(device)).sum().item()
            total += bsz
            correct += acc
            
            train_loss.append(running_loss/total)
            train_accuracy.append(100.0*correct/total)

            if i % 20 == 19:    # print every 20 mini-batches
                print('Train: [%d, %5d] loss: %.3f acc: %.3f' %
                      (e + 1, i + 1, running_loss / 20,100.0*correct/total))
                running_loss = 0.0  
        scheduler.step(e) 
    net = net.cpu()
#     tmp = nn.Sequential(*list(net.children())[:-1])
    pretrained_dict = net.state_dict()
#     model.enc_module.load_state_dict(pretrained_dict)
    
    tmp = list(pretrained_dict.items())[:-1]
    # print(tmp)
    # mod_dict = model.enc_module.state_dict()
    # count=0
    # for key,value in mod_dict.items():
    #     layer_name,weights=tmp[count]      
    #     mod_dict[key]=weights
    #     count+=1
    # model.enc_module.load_state_dict(mod_dict)
    return net