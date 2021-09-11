import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F
class MLP_encode(nn.Module):
    def __init__(self, x_dim=512, h_dim=[256,128,64]):
        super().__init__()  
        # resnet18 = models.resnet18(pretrained=True)
        # extractor = torch.nn.Sequential(*list(resnet18.children())[:-1])
        # for p in extractor.parameters():
        #    p.requires_grad = False
        # extractor.eval()
        # self.feat_extractor = extractor
        # encoder part
        encoder = []
        for i in range(0,len(h_dim)-1):
            if i == 0:
                encoder.append(nn.Linear(x_dim, h_dim[i]))
                encoder.append(nn.BatchNorm1d(h_dim[i]))
                encoder.append(nn.ReLU())
            else:
                encoder.append(nn.Linear(h_dim[i-1], h_dim[i]))
                encoder.append(nn.BatchNorm1d(h_dim[i]))
                encoder.append(nn.ReLU())
        self.embed = nn.Sequential(*encoder)
        self.mean = nn.Sequential(nn.Linear(h_dim[-2], h_dim[-1]),
                                 nn.BatchNorm1d(h_dim[-1]),
                                 nn.ReLU())
        self.var = nn.Sequential(nn.Linear(h_dim[-2], h_dim[-1]),
                                 nn.BatchNorm1d(h_dim[-1]),
                                 nn.ReLU())
    def forward(self, x):
        x = self.embed(x)
        mu = self.mean(x)
        var = self.var(x)       
        return mu, var

class MLP_decode(nn.Module):
    def __init__(self, x_dim=512, h_dim=[256,128,64]):
        super().__init__()  
        decoder=[]
        for i in range(len(h_dim)-1,-1,-1):
            if i==0:
                decoder.append(nn.Linear( h_dim[i],x_dim))
                decoder.append(nn.BatchNorm1d(x_dim))
                decoder.append(nn.Sigmoid())
            else:
                decoder.append(nn.Linear( h_dim[i],h_dim[i-1]))
                decoder.append(nn.BatchNorm1d(h_dim[i-1]))
                decoder.append(nn.ReLU())
        self.decoder = nn.Sequential(*decoder)

    def forward(self,x):
        x = self.decoder(x)
        return x

