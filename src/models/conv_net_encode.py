import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

class View(nn.Module):
    def __init__(self):
        super(View, self).__init__()
   
    def forward(self, x):
        numel = x.numel() / x.shape[0]
        return x.view(-1, int(numel)) 

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

def convNoutput(convs, input_size): # predict output size after conv layers
    input_size = int(input_size)
    input_channels = convs[0][0].weight.shape[1] # input channel
    output = torch.Tensor(1, input_channels, input_size, input_size)
    with torch.no_grad():
        for conv in convs:
            output = conv(output)
    return output.numel(), output.shape

class Spatial_X(nn.Module):
    def __init__(self, input_channels, input_size, params):
        super(Spatial_X, self).__init__()

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(
                    nn.Conv2d(input_channels, params[0], kernel_size=5, stride=1, padding=2),
                    nn.ReLU(True),
                    nn.MaxPool2d(kernel_size=2, stride=2)
                    )
        self.conv2 = nn.Sequential(
                    nn.Conv2d(params[0], params[1], kernel_size=5, stride=1, padding=2),
                    nn.ReLU(True),
                    nn.MaxPool2d(kernel_size=2, stride=2)
                    )

        out_numel, out_size = convNoutput([self.conv1, self.conv2], input_size/2)
        # set fc layer based on predicted size
        self.fc = nn.Sequential(
                View(),
                nn.Linear(out_numel, params[2]),
                nn.ReLU()
                )
        self.classifier = classifier = nn.Sequential(
                View(),
                nn.Linear(params[2], 6) # affine transform has 6 parameters
                )
        # initialize stn parameters (affine transform)
        self.classifier[1].weight.data.fill_(0)
        self.classifier[1].bias.data = torch.FloatTensor([1, 0, 0, 0, 1, 0])

   

    def localization_network(self, x):
        x = self.maxpool(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc(x)
        x = self.classifier(x)
        return x

    def forward(self, x):
        theta = self.localization_network(x)
        theta = theta.view(-1,2,3)
        grid = F.affine_grid(theta, x.size(),align_corners=True)
        x = F.grid_sample(x, grid,align_corners=True)
        return x

class conv_net_encode(nn.Module):
    def __init__(self, inp_channels=1, hid_dim=[64,64,64,64],conv_filters=[3,3,3,3],linear = False,linear_inp_siz=16000,z_dim=300,stride=1,
        stn = [[200,300,200], None, [150, 150, 150]],init_weights=True):
        super().__init__()
        
        self.stn = stn
        self.linear = linear
        encoder = []
        for i in range(1,len(hid_dim)):
          if i==1:
            
            if self.stn[i-1] is not None:                
                encoder.append(Spatial_X(inp_channels,64,self.stn[i-1]))
               
            encoder.append(self.conv_block(in_channels=inp_channels, out_channels=hid_dim[i-1],fs=conv_filters[i-1],stride=stride))
          else:           
            if self.stn[i-1] is not None:
                encoder.append(Spatial_X(hid_dim[i-1],32,self.stn[i-1]))
                

            encoder.append(self.conv_block(in_channels=hid_dim[i-2], out_channels=hid_dim[i-1],fs=conv_filters[i-1],stride=stride)) 
        if linear:         
            if self.stn[-1] is not None:
                encoder.append(Spatial_X(hid_dim[-2],16,self.stn[-1]))

            encoder.append(self.conv_block(in_channels=hid_dim[-2], out_channels=hid_dim[-1],fs=conv_filters[-1],stride=stride))   
            encoder.append(Flatten())
            self.embed = nn.Sequential(*encoder)
            self.mean = nn.Sequential(nn.Linear(linear_inp_siz, z_dim))
            self.var = nn.Sequential(nn.Linear(linear_inp_siz, z_dim))
        else:
            self.embed = nn.Sequential(*encoder)
            self.mean = nn.Sequential(self.conv_block(in_channels=hid_dim[-2], out_channels=hid_dim[-1],fs=conv_filters[-1],stride=stride))
            self.var = nn.Sequential(self.conv_block(in_channels=hid_dim[-2], out_channels=hid_dim[-1],fs=conv_filters[-1],stride=stride))

        self.tau = nn.Parameter(torch.tensor(10.0))

    def conv_block(self,in_channels,out_channels,bias=True,fs=3,stride=1, pad=0,enc=True):
        
        convolver=nn.Conv2d(in_channels, out_channels, fs, stride=stride, padding=int((fs - 1) / 2), bias=bias)
        layers=nn.Sequential(convolver,
                             nn.BatchNorm2d(out_channels,track_running_stats=False ),
                             nn.LeakyReLU(0.2))
        if(enc and not self.linear):
          layers= nn.Sequential(layers,
              nn.MaxPool2d(2))

        return layers

    

    def forward(self,x):
        x = self.embed(x)
        mu = self.mean(x)
        # mu_flat = torch.flatten(mu,start_dim=1)
        var = self.var(x)  
        tau = self.tau      
        # var_flat = torch.flatten(var,start_dim=1)
        return mu,var,tau
