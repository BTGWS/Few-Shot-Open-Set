import torch
from torch import nn, optim
import torch.nn.functional as F



class ResizeConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor=2,outsize=[84,84],final=False, mode='nearest'):
        super().__init__()
        self.outsize = outsize
        self.scale_factor = scale_factor
        self.mode = mode
        self.final = final
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)

    def forward(self, x):
        if self.final:
            x = F.interpolate(x, size=self.outsize, mode=self.mode)
        else:
            x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)

        x = self.conv(x)
        return x


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlockEnc(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super(BasicBlockEnc, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes,track_running_stats=False )
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes,track_running_stats=False )
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet18Enc(nn.Module):

    def __init__(self, block, layers, norm_layer=None,z_dim=512, branch=False,tau=1000.0):
        super(ResNet18Enc, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.n_blocks = len(layers)
        self.branch = branch
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes,track_running_stats=False )
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(2, stride=1)
        # self.avg_pool0 = torch.nn.AdaptiveAvgPool2d(1)
        self.linear0 = nn.Sequential(conv1x1(512, z_dim, stride=2))
        # self.linear01 = nn.Sequential(conv1x1(512, z_dim, stride=2))
        if branch:
            # self.inplanes = 256 * block.expansion
            # self.layer4_1 = self._make_layer(block, 512, layers[3], stride=2)
            # self.avg_pool1 = torch.nn.AdaptiveAvgPool2d(1)
            self.linear1 = nn.Sequential(conv1x1(512, z_dim, stride=2))
            # self.linear11 = nn.Sequential(conv1x1(512, z_dim, stride=2))
        self.tau = nn.Parameter(torch.tensor(tau))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.InstanceNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion,track_running_stats=False ),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                norm_layer=norm_layer))
        # print(nn.Sequential(*layers))
        return nn.Sequential(*layers)

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        
        x = self.layer2(x)
        
        x = self.layer3(x)
        
        x = self.layer4(x)
        # 
        mu = self.linear0(x) 
        mu = self.avgpool(mu)
        tau = self.tau
        # mu = mu.view(mu.size(0), -1)  
        # mu = mu.unsqueeze(2)
        # mu = mu.unsqueeze(3)  
        
        # mu = self.linear01(mu)   
        # mu = self.linear0(mu)
        if self.branch:
            sig = self.linear1(x)
            sig = self.avgpool(sig)
            # sig = sig.view(sig.size(0), -1)  
            # sig = sig.unsqueeze(2)
            # sig = sig.unsqueeze(3)  
            # sig = self.linear10(sig)
            # sig = self.linear1(x) 
            # sig = sig.view(sig.size(0),-1)
            # sig = self.avg_pool0(sig)   
            # sig = self.linear1(sig)
            return [mu,sig,tau]
        else:
            return mu,tau

        return x

class BasicBlockDec(nn.Module):

    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = int(in_planes/stride)

        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)
        # self.bn1 could have been placed here, but that messes up the order of the layers when printing the class

        if stride == 1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
                nn.InstanceNorm2d(planes)
            )

    def forward(self, x):
        out = torch.nn.functional.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = torch.nn.functional.relu(out)
        return out


class ResNet18Dec(nn.Module):

    def __init__(self, num_Blocks=[2,2,2,2], outsize=[84,84], z_dim=2048,  nc=3):
        super(ResNet18Dec,self).__init__()
        self.in_planes = z_dim
        self.outsize = outsize

        self.linear = nn.Linear(z_dim, 512)

        self.layer4 = self._make_layer(BasicBlockDec, 256, num_Blocks[3], stride=2)
        self.layer3 = self._make_layer(BasicBlockDec, 128, num_Blocks[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, 64, num_Blocks[1], stride=2)
        self.layer1 = self._make_layer(BasicBlockDec, 64, num_Blocks[0], stride=1)
        self.conv1 = ResizeConv2d(64, nc, kernel_size=3, final=True,outsize=self.outsize)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            

    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, stride)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.view((x.size(0), -1))
        x = F.relu(self.linear(x))
        x = x.view(x.size(0), 512, 1, 1)
        x = F.interpolate(x, scale_factor=4)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = self.conv1(x)
        x = torch.sigmoid(x)
        # x = torch.tanh(self.conv1(x))
        x = x.view(x.size(0), 3, self.outsize[0], self.outsize[1])
        return x


# class BasicBlockEnc(nn.Module):

#     def __init__(self, in_planes, stride=1):
#         super().__init__()

#         planes = in_planes*stride

#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.InstanceNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.InstanceNorm2d(planes)

#         if stride == 1:
#             self.shortcut = nn.Sequential()
#         else:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
#                 nn.InstanceNorm2d(planes)
#             )

#     def forward(self, x):
#         out = torch.nn.functional.leaky_relu(self.bn1(self.conv1(x)),0.2)
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(x)
#         out = torch.nn.functional.leaky_relu(out,0.2)
#         return out

# class ResNet18Enc(nn.Module):

#     def __init__(self, num_Blocks=[2,2,2,2], z_dim=10, nc=3):
#         super().__init__()
#         self.in_planes = 64
#         self.z_dim = z_dim
#         self.conv1 = nn.Conv2d(nc, 64, kernel_size=3, stride=2, padding=1, bias=False)
#         self.bn1 = nn.InstanceNorm2d(64)
#         self.layer1 = self._make_layer(BasicBlockEnc, 64, num_Blocks[0], stride=1)
#         self.layer2 = self._make_layer(BasicBlockEnc, 128, num_Blocks[1], stride=2)
#         self.layer3 = self._make_layer(BasicBlockEnc, 256, num_Blocks[2], stride=2)
#         self.layer4 = self._make_layer(BasicBlockEnc, 512, num_Blocks[3], stride=2)
#         self.linear1 = nn.Sequential(nn.Linear(512, z_dim),
#                                     nn.LeakyReLU(0.2),
#                                     nn.BatchNorm1d(z_dim))
#         self.linear2 = nn.Sequential(nn.Linear(512, z_dim),
#                                     nn.ReLU(),
#                                     nn.BatchNorm1d(z_dim))
#     def _make_layer(self, BasicBlockEnc, planes, num_Blocks, stride):
#         strides = [stride] + [1]*(num_Blocks-1)
#         layers = []
#         for stride in strides:
#             layers += [BasicBlockEnc(self.in_planes, stride)]
#             self.in_planes = planes
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         x = torch.nn.functional.leaky_relu(self.bn1(self.conv1(x)),0.2)
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = F.adaptive_avg_pool2d(x, 1)
#         x = x.view(x.size(0), -1)
#         mu = self.linear1(x)
#         logvar = self.linear2(x)
#         # mu = x[:, :self.z_dim]
#         # logvar = x[:, self.z_dim:]
#         return mu, logvar


