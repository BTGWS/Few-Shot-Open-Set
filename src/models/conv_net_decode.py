import torch
import torch.nn as nn
import numpy as np
import math

class Flat_to_conv(nn.Module):
    def __init__(self,h,w,c):
        super(Flat_to_conv, self).__init__()
        self.height = h
        self.width = w
        self.channels = c

    def forward(self, x):
        return x.view((x.size(0),self.channels,self.height,self.width))

class conv_net_decode(nn.Module):
    def __init__(self, inp_channels=1, hid_dim=[64,64,64,64],conv_filters=[3,3,3,3],linear = False,linear_inp_siz=16000,
        z_dim=300,stride=1, num_layers=4, out_size=[64,64],init_weights=True):
        super().__init__()
        decoder=[]
        if(linear):
            c = hid_dim[-1]
            h = int(math.sqrt(linear_inp_siz/c))
            w = int(math.sqrt(linear_inp_siz/c))
            decoder.append(nn.Linear(z_dim,linear_inp_siz))
            decoder.append(nn.ReLU())
            decoder.append(Flat_to_conv(h=h,w=w,c=c))

        for i in range(len(hid_dim)-1,-1,-1):
          if i==0:
            decoder.append(nn.Upsample(size=out_size, mode='nearest'))
            decoder.append(self.conv_block(in_channels=hid_dim[i], out_channels=inp_channels,fs=conv_filters[i],stride=stride,enc=False))
            decoder.append(nn.Sigmoid())
          else:
            decoder.append(nn.Upsample(scale_factor=2, mode='nearest'))
            decoder.append(self.conv_block(in_channels=hid_dim[i], out_channels=hid_dim[i-1],fs=conv_filters[i],stride=stride,enc=False))
          
        self.decoder = nn.Sequential(*decoder)
        

    def conv_block(self,in_channels,out_channels,bias=True,fs=3,stride=1, pad=0,enc=True):
        padder=nn.ReplicationPad2d(int((fs - 1) / 2))
        convolver=nn.Conv2d(in_channels, out_channels, fs, stride=stride, padding=pad, bias=bias)
        layers=nn.Sequential(padder, 
                             convolver,
                             nn.BatchNorm2d(out_channels,1e-3,track_running_stats=False ),
                             nn.LeakyReLU(0.2))
        
        return layers

    

    def forward(self,x):
        x = self.decoder(x);
        return x
