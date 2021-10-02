import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np 
import sys

def loss_vpe(recon_x, x, mu, log_var,device,percept=None,recon='ce'):
    
    if recon == 'ce':
        BCE = nn.BCELoss(reduction='sum')(recon_x, x)
    elif recon == 'l1':
        BCE = nn.L1Loss(reduction='sum')(recon_x, x)
    elif recon == 'l2':
        BCE = nn.MSELoss(reduction='sum')(recon_x, x)
    else:
        BCE = percept(recon_x,x)
    KLD = -0.5* torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    l = (BCE + KLD)#*(1/recon_x.shape[0])
    return l
def loss_recon(recon_x, x):
    l1_l =torch.nn.L1Loss(reduction='sum')
    return l1_l(recon_x,x)
def hot_encoding(labels,classes,device):
    labels_encoded = torch.zeros((labels.shape[0],classes.shape[0])).to(device)
    in_idx = (labels!=-1).nonzero()
    out_idx = (labels==-1).nonzero()
    labels_encoded[in_idx,labels[in_idx[0]]] = 1    
    labels_encoded[out_idx,:] = (1/classes.shape[0])*torch.ones((1,classes.shape[0])).to(device)
    return labels_encoded

def loss_clf(logits,labels,classes,device):
    soft_max = nn.Softmax(dim = 1)
    log_softmax = nn.LogSoftmax(dim = 1)
    ml,_ = torch.max(logits,dim=1)
    ml = ml.unsqueeze(dim=1)
    logits = logits - ml.repeat((1,logits.shape[1]))
    pred = soft_max(logits)
    label_encoded = hot_encoding(labels,classes,device)  
    log_probabilities = log_softmax(logits)
    # print(label_encoded.shape[0]//2)
    # print(label_encoded[:label_encoded.shape[0]//2])
    loss = torch.mean(-label_encoded[:label_encoded.shape[0]//2]*log_probabilities[:label_encoded.shape[0]//2])
    return pred,loss

def loss_novel(logits,labels,device):
    BCE = nn.BCELoss()
    logits = logits.type(torch.FloatTensor).to(device)
    labels = labels.type(torch.FloatTensor).to(device)
    loss = BCE(logits,labels)
    return loss

def class_renumb(train_y):    
    classes = torch.unique(train_y)
    d = {classes[i].item():i for i in range(classes.shape[0])}
    for i in range(train_y.shape[0]):
      train_y[i] = torch.tensor(d[train_y[i].item()])
    classes = torch.unique(train_y)
    return classes,train_y

def class_scaler(train_y,n,q):
    num_in = n + q//2
    num_out = q//2
    classes_in = []  
    classes,train_y = class_renumb(train_y)
    for k in classes:
        num_sample = train_y[train_y==k].shape[0]        
        in_idx = (train_y==k).nonzero()
        if(num_sample == num_in and (in_idx[-1]-in_idx[0] == (num_in-1))):
            classes_in.append(k)
    for k in classes_in:
        in_idx = (train_y==k).nonzero()
        train_y[in_idx[-1]+1:in_idx[-1]+1+num_out]=k  
    classes,train_y = class_renumb(train_y)
    return classes,train_y

def proto_rectifier(emb_support,emb_proto_k,euc=False,wts=True):
    if(len(emb_proto_k.shape)>2 or len(emb_support.shape)>2):
      sys.exit('Error: embedding should be 1d !')
    # emb_proto_k = torch.reshape(emb_proto_k, (emb_proto_k.shape[0],emb_proto_k.shape[1]))
    # emb_support = torch.reshape(emb_support, (emb_support.shape[0],emb_support.shape[1]))
    w_gen = nn.Softmax(dim = 0)
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    if wts:
        if euc:
            d = -((emb_support - emb_proto_k)**2).sum(dim=1)
        else:
            d = cos(emb_support,emb_proto_k)
    else:
        get_cuda_device = emb_support.get_device()
        # d = torch.ones((emb_support.shape[0])).to(get_cuda_device)
        proto_new = torch.mean(emb_support,dim=0,keepdim=True)
        # print(proto_new.shape)
        return proto_new
    weights = w_gen(d)
    weights= torch.unsqueeze(weights,dim=0)
    proto_new = torch.matmul(weights,emb_support)
    return proto_new
    

def cosine_classifier(emb_query,real_proto,device,euc=False,test=False,tau=1):
    if(len(emb_query.shape)>2 or len(real_proto.shape)>2):
      sys.exit('Error: embedding should be 1d !')   
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    logits = torch.zeros((emb_query.shape[0],real_proto.shape[0])).to(device)
    for i in range(0,real_proto.shape[0]):
        tmp = real_proto[i,:].unsqueeze(dim=0)
        if euc:
            eq = emb_query
            # eq = emb_query/(torch.norm(emb_query) + 1e-8)
            # tmp = tmp/(torch.norm(tmp) + 1e-8)
            l = -((tmp - eq)**2).sum(dim=1)
        else:
            l = cos(tmp,emb_query) 
        logits[:,i] = torch.reshape(l,[l.shape[0],]) #Nq x C
    logits = logits*tau
    if test:
        logits = nn.Softmax(dim=1)(logits)

    return logits

def Recon_diff(recon_query,symbolic_proto,device):
    class_num = symbolic_proto.shape[0]
    l1_diff = torch.Tensor().to(device)
    for i in range(0,class_num):
        # tmp = torch.abs(recon_query - symbolic_proto[i,:,:,:].unsqueeze(dim=0)) #Nq x dim      

        tmp = (recon_query - symbolic_proto[i,:].unsqueeze(dim=0)) #Nq x dim
        tmp = tmp.pow(2)

        tmp = tmp.mean(dim=list(range(1,len(tmp.shape)))).unsqueeze(dim=1)
        l1_diff = torch.cat((l1_diff,tmp),dim=1)# Nq x C
    return l1_diff

def emb_enhance(query,symbolic_proto,real_proto,device,emh=True):
    class_num = symbolic_proto.shape[0]
    l1_diff = torch.Tensor().to(device)
    if not emh:
        return query
    for i in range(0,class_num):
        proto = symbolic_proto[i,:]
        tmp = torch.abs(query - proto) #Nq x dim         
        tmp = tmp.mean(dim=[1]).unsqueeze(dim=1)        
        l1_diff = torch.cat((l1_diff,tmp),dim=1)# Nq x C
    min_diff,_ = torch.min(l1_diff,dim=1)
    min_diff = 1/(min_diff+10**(-24))
    min_diff = min_diff.unsqueeze(1)
    min_diff = min_diff.expand(min_diff.shape[0],query.shape[1])
    new_query = min_diff*query
    return new_query

def extract_episode(train_x,train_y,n_support,device):  
    n_examples =  train_x.shape[0]
    n_query_tot = n_examples - n_support
    n_query_out = n_query_tot//2
    example_inds = torch.arange(n_examples)[:(n_support+n_query_tot)]
    support_inds = example_inds[:n_support]
    query_inds = example_inds[n_support:]
    Dx_k = train_x[support_inds,:,:,:]
    Dy_k = train_y[support_inds]
    Qx_k = train_x[query_inds,:,:,:]
    Qy_k_lb = train_y[query_inds]
    Qy_k_lb[n_query_out:] = -1 # for ood samples
    Qy_k_ab = torch.zeros(n_query_tot).type(torch.LongTensor).to(device)
    Qy_k_ab[n_query_out:] = 1
    Qy_k_lb = torch.unsqueeze(Qy_k_lb,dim=1)
    Qy_k_ab = torch.unsqueeze(Qy_k_ab,dim=1)
    Qy_k = torch.cat((Qy_k_lb,Qy_k_ab),dim = 1)
    return Dx_k,Dy_k,Qx_k,Qy_k

class NegativeEntropy(nn.Module):
    def __init__(self):
        super(NegativeEntropy, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = 1.0 * b.sum()
        b = b.mean()
        return b

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
        self.device = device

    def forward(self, inp, target):
        self.mean = self.mean.to(self.device)
        self.std = self.std.to(self.device)
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



class Logger(object):
    def __init__(self, log_file):
        self.file = log_file
    def __call__(self, msg, init=False, quiet_ter=False ):
        if not quiet_ter:
            print(msg,end=" ")

        if init:
            try:
                os.remove(self.file)
            except:
                pass
        
        with open(self.file, 'a') as log_file:
                log_file.write('%s\n' % msg)