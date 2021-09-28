from torch import nn
import torchvision.models as torchmodels
from torch.nn import functional as F
from models.Resnet18 import *
from models.conv_net_encode import *
from models.conv_net_decode import *
from models.MLP_VAE import *
from models.Resnet152 import *
from models.Resnet12 import build_resnet12,build_resnet12dec
import math


class Feature_extractor(nn.Module):
    def __init__(self):
        super().__init__() 
        resnet18 = torchmodels.resnet18(pretrained=True)
        extractor = torch.nn.Sequential(*list(resnet18.children())[:-1])
        for p in extractor.parameters():
           p.requires_grad = False
        extractor.eval()
        self.extractor = extractor
    def forward(self,x):
        x = self.extractor(x)        
        # x= F.sigmoid(x)
        x = x.view(x.size(0), -1)
        return x

class Encoder(nn.Module):
    def __init__(self, layers=[2, 2, 2, 2], norm_layer=None,
        branch=False,backbone = 'conv_layers',mlp_inp_dim=512,mlp_hid_layers=[256,128,64], inp_channels=3, hid_dim=[64,64,64,64],conv_filters=[3,3,3,3],linear = False,
        linear_inp_siz=16000,stn = [[200,300,200], None, [150, 150, 150]],z_dim=300,stride=1,temperature=100.0):
        super(Encoder, self).__init__() 
        self.backbone = backbone
        

        if(self.backbone == 'resnet18'):
            block = BasicBlockEnc
            self.embed = ResNet18Enc(block, layers, norm_layer=norm_layer,z_dim=z_dim, branch=branch)
        elif(self.backbone == 'resnet12'):
            # print(self.backbone)
            block = BasicBlockEnc
            layers = [2, 1, 1, 1]
            self.embed = ResNet18Enc(block, layers=layers, norm_layer=norm_layer,z_dim=z_dim, branch=branch)
        elif(self.backbone == 'custom_resnet12'):
            self.embed = build_resnet12(avg_pool=True, drop_rate=0.0, dropblock_size=5,branch=True,tau=temperature)
        elif(self.backbone == 'resnet152'):
            self.embed = ResNet_152_Encoder(CNN_embed_dim = z_dim)
        elif(self.backbone == 'MLP'):
            self.embed = MLP_encode(x_dim=mlp_inp_dim, h_dim=mlp_hid_layers)
        else:
            self.embed = conv_net_encode(inp_channels=inp_channels,hid_dim=hid_dim,
                conv_filters=conv_filters,linear=linear,linear_inp_siz=linear_inp_siz,z_dim=z_dim,stn=stn,stride=stride )
    

    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    def forward(self, x): 
        mu,log_var,tau = self.embed(x)
        mu = mu.view(mu.size(0),-1)
        log_var = log_var.view(log_var.size(0),-1)
        z = self.sampling(mu, log_var)
        z = z.view(z.size(0),-1)
        return z,mu,log_var,tau

class Decoder(nn.Module):
    def __init__(self, backbone = 'conv_layers',mlp_inp_dim=512,mlp_hid_layers=[256,128,64], inp_channels=1,layers=[2, 2, 2, 2], hid_dim=[64,64,64,64],
        conv_filters=[3,3,3,3],linear = False, linear_inp_siz=16000,outsize=[64,64],z_dim=300,stride=1):
        super(Decoder, self).__init__() 
        self.backbone = backbone
        if(self.backbone == 'resnet18'):
            self.decode = ResNet18Dec(num_Blocks=layers,z_dim=z_dim,outsize=outsize,nc=inp_channels)
        elif(self.backbone == 'resnet12'):
            layers = [1,1,1,1]
            self.decode = ResNet18Dec(num_Blocks=layers,z_dim=z_dim,outsize=outsize,nc=inp_channels)
        elif self.backbone == 'custom_resnet12':
            self.decode = build_resnet12dec(avg_pool=False, drop_rate=0.0, dropblock_size=5,outsize=outsize,z_dim=z_dim)
        elif(self.backbone == 'resnet152'):
            self.decode = ResNet_152_Decoder(CNN_embed_dim=z_dim,out_c=inp_channels, out_size=outsize)
        elif(self.backbone =='MLP'):
            self.decode = MLP_decode(x_dim=mlp_inp_dim, h_dim=mlp_hid_layers)
        else:
            self.decode = conv_net_decode(inp_channels=inp_channels,hid_dim=hid_dim,
                conv_filters=conv_filters,linear=linear,linear_inp_siz=linear_inp_siz,z_dim=z_dim,stride=stride,out_size=outsize )

    def forward(self, x):        
        img = self.decode(x)
        return img

class Rel_net_detect(nn.Module):
    def __init__(self, z_dim):
        super(Rel_net_detect, self).__init__()
        
        self.rel_fc1 = nn.Linear(2*z_dim,z_dim)
        self.rel_fc2 = nn.Linear(z_dim,1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data = torch.ones(m.bias.data.size())
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self,x):

        x = F.relu(self.rel_fc1(x))
        x = torch.sigmoid(self.rel_fc2(x))
        return x

class Ab_module(nn.Module):
    def __init__(self, inp_size, layers = [256,256]):
        super(Ab_module, self).__init__()
        detector = []
        for h in range(0,len(layers)):

            if(h==0):                
                detector.append(nn.Linear(inp_size,layers[h]))
                detector.append(nn.ReLU())
                # detector.append(nn.InstanceNorm1d(layers[h],track_running_stats=False))
            else:
                detector.append(nn.Linear(layers[h-1],layers[h]))
                detector.append(nn.ReLU())
        detector.append(nn.Linear(layers[h],1))
        detector.append(nn.Sigmoid())
        self.detector = nn.Sequential( *detector )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data = torch.ones(m.bias.data.size())
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self,x):
        out = self.detector(x)
        return out

class Proto_ND(nn.Module):

    def __init__(self,ab_inp_size, resnet_layers=[2, 2, 2, 2], norm_layer=None,
        branch=True,backbone='conv_layers',mlp_inp_dim=512,mlp_hid_layers=[256,128,64],inp_channels=1, hid_dim=[64,64,64,64],enc_conv_filters=[3,3,3,3],
        dec_conv_filters=[3,3,3,3],linear = False,linear_inp_siz=16000,stride=1, outsize=[64,64],stn = [[200,300,200], None, [150, 150, 150]],
        ab_layers = [256,256],z_dim=300,init_weights=True,clf_mode='cosine',temperature=100.0 ):

        super(Proto_ND,self).__init__()
        self.clf_mode = clf_mode
        self.backbone = backbone
        # self.feat_ext_module = Feature_extractor()
        self.enc_module = Encoder(layers=resnet_layers, norm_layer=norm_layer,
        branch=branch,backbone = self.backbone,mlp_inp_dim=mlp_inp_dim,mlp_hid_layers=mlp_hid_layers,inp_channels=inp_channels, hid_dim=hid_dim,
        conv_filters=enc_conv_filters,linear = linear,linear_inp_siz=linear_inp_siz,stn =stn,z_dim=z_dim,stride=stride,temperature=temperature)

        self.dec_module = Decoder(backbone = self.backbone,mlp_inp_dim=mlp_inp_dim,mlp_hid_layers=mlp_hid_layers,layers=resnet_layers,
                                inp_channels=inp_channels, hid_dim=hid_dim,outsize=outsize,
                                conv_filters=dec_conv_filters,linear = linear,linear_inp_siz=linear_inp_siz,z_dim=z_dim,stride=1)
        
        self.nd_module =  Ab_module(inp_size=ab_inp_size, layers = ab_layers)
        self.classifier = Rel_net_detect(z_dim=z_dim)
        # if init_weights:
        #     self._initialize_weights()

    # def feat_extractor(self,x):
    #     x = self.feat_ext_module(x)
    #     return x

    def classify(self,x):
        return self.classifier(x)

    def encode(self,x):
        z,mu,log_var,tau = self.enc_module(x)
        return z,mu,log_var,tau     

    def decode(self,z):
        x = self.dec_module(z)
        return x

    def nd_clf(self,x):
        out = self.nd_module(x)
        return out

    def forward(self,x):
        if(self.backbone == 'MLP'):
            x = self.feat_extractor(x)
        z,mu,log_var,tau = self.encode(x)
        out = self.decode(z)
        return z,mu,log_var,out,tau
        

    def _initialize_weights(self):


          for m in self.modules():
              if isinstance(m, nn.Conv2d):
                  nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                  if m.bias is not None:
                      nn.init.constant_(m.bias, 0)
              elif isinstance(m, nn.BatchNorm2d):
                  nn.init.uniform_(m.weight)
                  nn.init.constant_(m.bias, 0)
              elif isinstance(m, nn.Linear):
                  nn.init.normal_(m.weight, 0, 0.01)
                  nn.init.constant_(m.bias, 0)








""" class VAE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(VAE, self).__init__()  
        # encoder part
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)
        self.fc32 = nn.Linear(h_dim2, z_dim)
        # decoder part
        self.fc4 = nn.Linear(z_dim, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)
    def encoder(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h) # mu, log_var
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample
    def decoder(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        return F.sigmoid(self.fc6(h)) 
    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, 784))
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var
def loss_function(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD """