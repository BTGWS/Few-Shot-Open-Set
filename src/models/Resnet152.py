import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
import torchvision.transforms as transforms

def conv2D_output_size(img_size, padding, kernel_size, stride):
    # compute output shape of conv2D
    outshape = (np.floor((img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int),
                np.floor((img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int))
    return outshape

def convtrans2D_output_size(img_size, padding, kernel_size, stride):
    # compute output shape of conv2D
    outshape = ((img_size[0] - 1) * stride[0] - 2 * padding[0] + kernel_size[0],
                (img_size[1] - 1) * stride[1] - 2 * padding[1] + kernel_size[1])
    return outshape

## ---------------------- ResNet VAE ---------------------- ##


class ResNet_152_Encoder(nn.Module):
    def __init__(self, fc_hidden1=1024, fc_hidden2=768, drop_p=0.3, CNN_embed_dim=256):
        super(ResNet_152_Encoder, self).__init__()

        self.fc_hidden1, self.fc_hidden2, self.CNN_embed_dim = fc_hidden1, fc_hidden2, CNN_embed_dim

        # CNN architechtures
        self.ch1, self.ch2, self.ch3, self.ch4 = 16, 32, 64, 128
        self.k1, self.k2, self.k3, self.k4 = (5, 5), (3, 3), (3, 3), (3, 3)      # 2d kernal size
        self.s1, self.s2, self.s3, self.s4 = (2, 2), (2, 2), (2, 2), (2, 2)      # 2d strides
        self.pd1, self.pd2, self.pd3, self.pd4 = (0, 0), (0, 0), (0, 0), (0, 0)  # 2d padding

        # encoding components
        resnet = models.wide_resnet50_2(pretrained=False)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.fc1 = nn.Linear(resnet.fc.in_features, self.fc_hidden1)
        self.relu1 = nn.ReLU(inplace=True)
        # self.bn1 = nn.BatchNorm1d(self.fc_hidden1, momentum=0.01,track_running_stats=True)
        self.fc2 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
        self.relu2 = nn.ReLU(inplace=True)
        # self.bn2 = nn.BatchNorm1d(self.fc_hidden2, momentum=0.01,track_running_stats=True)
        # Latent vectors mu and sigma
        self.fc3_mu = nn.Linear(self.fc_hidden2, self.CNN_embed_dim)      # output = CNN embedding latent variables
        self.fc3_logvar = nn.Linear(self.fc_hidden2, self.CNN_embed_dim)  # output = CNN embedding latent variables
        

    def encode(self, x):
        x = self.resnet(x)  # ResNet

        x = x.view(x.size(0), -1)  # flatten output of conv

        # FC layers
        # x = self.bn1(self.fc1(x))
        x = (self.fc1(x))
        x = self.relu1(x)
        # x = self.bn2(self.fc2(x))
        x = (self.fc2(x))
        x = self.relu2(x)
        # x = F.dropout(x, p=self.drop_p, training=self.training)
        mu, logvar = self.fc3_mu(x), self.fc3_logvar(x)
        return mu, logvar


    def forward(self, x):
        mu, logvar = self.encode(x)
        return  mu, logvar

class ResNet_152_Decoder(nn.Module):
    def __init__(self, fc_hidden1=1024, out_c =3, out_size=[84,84], fc_hidden2=768, drop_p=0.3, CNN_embed_dim=256):
        super(ResNet_152_Decoder, self).__init__()

        self.fc_hidden1, self.fc_hidden2, self.CNN_embed_dim = fc_hidden1, fc_hidden2, CNN_embed_dim
        # CNN architechtures
        self.ch1, self.ch2, self.ch3, self.ch4 = 16, 32, 64, 128
        self.k1, self.k2, self.k3, self.k4 = (5, 5), (3, 3), (3, 3), (3, 3)      # 2d kernal size
        self.s1, self.s2, self.s3, self.s4 = (2, 2), (2, 2), (2, 2), (2, 2)      # 2d strides
        self.pd1, self.pd2, self.pd3, self.pd4 = (0, 0), (0, 0), (0, 0), (0, 0)  # 2d padding
        self.outsize = out_size
        self.out_c = out_c
        # Sampling vector
        self.fc4 = nn.Linear(self.CNN_embed_dim, self.fc_hidden2)
        # self.fc_bn4 = nn.BatchNorm1d(self.fc_hidden2,track_running_stats=True)
        self.fc5 = nn.Linear(self.fc_hidden2, 64 * 4 * 4)
        # self.fc_bn5 = nn.BatchNorm1d(64 * 4 * 4,track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        # Decoder
        self.convTrans6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=self.k4, stride=self.s4,
                               padding=self.pd4),
            nn.BatchNorm2d(32, momentum=0.01,track_running_stats=False),
            nn.ReLU(inplace=True),
        )
        self.convTrans7 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=self.k3, stride=self.s3,
                               padding=self.pd3),
            nn.BatchNorm2d(8, momentum=0.01,track_running_stats=False),
            nn.ReLU(inplace=True),
        )

        self.convTrans8 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8, out_channels=self.out_c, kernel_size=self.k2, stride=self.s2,
                               padding=self.pd2),
            nn.BatchNorm2d(3, momentum=0.01,track_running_stats=False),
            nn.Sigmoid()    # y = (y1, y2, y3) \in [0 ,1]^3
        )


    def decode(self, z):
        # x = self.relu(self.fc_bn4(self.fc4(z)))
        x = F.relu((self.fc4(z)))
        # x = self.relu(self.fc_bn5(self.fc5(x))).view(-1, 64, 4, 4)        
        x = F.relu((self.fc5(x))).view(-1, 64, 4, 4)
        x = self.convTrans6(x)
        x = self.convTrans7(x)
        x = self.convTrans8(x)
        x = F.interpolate(x, size=self.outsize, mode='bilinear')
        return x

    def forward(self, x):
        x_reconst = self.decode(x)
        return x_reconst