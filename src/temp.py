import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np 
import sys
import torch.nn.functional as F
from models.Resnet152 import *
from itertools import chain 
class perceptualLoss(nn.Module):
    def __init__(self,device):
        super(perceptualLoss, self).__init__()
        model = models.vgg19(pretrained=True)
        model.to(device)
        for param in model.features.parameters():
            param.requires_grad = False
        self.modules = list(model.features.children())[:-1]
        ## conv1_2,conv2_2,conv3_2,conv4_2,conv5_2
        self.layers = [2,7,12,21,30]
        ## ImageNet statistics
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)   

    def forward(self, inp, target,device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        inp = (inp-self.mean) / self.std
        target = (target-self.mean) / self.std
        inp = F.interpolate(inp, mode='bilinear', size=(224, 224), align_corners=False)
        target = F.interpolate(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.
        x = inp 
        y = target
        for i, m in enumerate(self.modules):
            x = m(x)
            y = m(y)
            if i in self.layers:
                loss += F.l1_loss(x,y)
        return loss
class vae(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = ResNet_152_Encoder()
        self.decode =  ResNet_152_Decoder(out_size = [224,224])
    def forward(self,x):
        y,_ = self.embed(x)
        y = self.decode(y)
        return y



# resnet18 = models.resnet50(pretrained=True).cuda()
# vgg19 = models.vgg19(pretrained = True).cuda()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = vae()
model = nn.DataParallel(model)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(),0.001)
t = torch.randn([,3,224,224]).to(device)
p = perceptualLoss(device)
while True:
    optimizer.zero_grad()
    y = model(t)
    l = p(t,y,device)
    l.backward()
    optimizer.step()
    print(l)