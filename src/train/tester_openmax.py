import torch
import torch.nn as nn
import torchvision
import sys
import numpy as np
from train.train_utils import *
from models.model import *
from sklearn import metrics
import scipy.stats as st
import torch.nn.functional as F

def roc_area_calc(dist, closed, descending, total_height, total_width):
    _, p = dist.sort(descending=descending)
    closed_p = closed[p]

    height = 0.0
    width = 0.0
    area = 0.0
    pre = 0  # (0: width; 1: height)

    for i in range(len(closed_p)):
        if closed_p[i] == 1:
            if pre == 0:
                area += height * width
                width = 0.0
                height += 1.0
                pre = 1
            else:
                height += 1.0
        else:
            pre = 0
            width += 1.0
    if pre == 0:
        area += height * width

    area = area / total_height / total_width
    return area

def weibull_fit(x,proto):
    
    d = -((x - proto)**2).sum(dim=1)
    d = torch.unsqueeze(d,dim=1)
    mu = torch.mean(d,dim=0)
    en = torch.abs(d-mu)
    en = en.cpu().numpy()
    k , tau, lmb = st.weibull_min.fit(en, floc=0)
    return (k,tau,lmb)

def openmax(x,ro,device):
    omega = torch.zeros((x.shape[0],x.shape[1])).to(device)
    v = torch.zeros((x.shape[0],x.shape[1]+1)).to(device)
    # print(omega.shape)
    # print(x.shape)
    for i in range(0,x.shape[1]):
        k,tau,lmb = ro[i]
        omega[:,i] = 1 - ((x.shape[1]-i-1)/(x.shape[1]))*torch.exp(-((torch.norm(x[:,i])-tau)/lmb)**k)
    v[:,0:x.shape[1]] = torch.mul(x,omega)
    omega_1 = 1 - omega
    v[:,x.shape[1]] = torch.mul(x,omega_1).sum(dim=1)
    pred = F.softmax(v,dim=1)
    in_pred = pred[:,0:x.shape[1]]
    out_pred = pred[:,x.shape[1]]
    # print(out_pred.shape)
    return in_pred,out_pred
    

def tester(model,device,test_loader,opts):
   
    backbone = opts.backbone
    n_support = opts.k
    n_query = opts.q
    if opts.metric == 'euclidean':
      euc = True
    else:
      euc = False
    wts = opts.weighted_mean
    emh = opts.emb_enhance

    temp = opts.temperature

    total_count = 0
    print(len(test_loader))
    Au_ROC = np.zeros(len(test_loader))
    Accuracy = np.zeros(len(test_loader))
    c=0;
    with torch.no_grad():
        for i, episode in enumerate(test_loader):   
            # c=c+1
            # print(c)    
            test_x, test_y, proto_sym = episode
            if(len(test_y.shape)>1):
                test_y = torch.squeeze(test_y,dim=1)  
            classes,test_y = class_scaler(test_y,n_support,n_query)
            test_x = test_x.to(device)
            test_y = test_y.to(device)
            proto_sym = proto_sym.to(device)  
            classes = classes.to(device)

            
            all_sym_protos = torch.Tensor().to(device)
            for k in classes:                
                 symbolic_proto_k = proto_sym[test_y==k,:,:,:]
                 all_sym_protos = torch.cat((all_sym_protos,symbolic_proto_k[0,:,:,:].unsqueeze(dim=0)),dim=0)# C x dim
            _,all_sym_proto_embs,_,_ = model(all_sym_protos)  
            all_sym_proto_embs = torch.flatten(all_sym_proto_embs,start_dim=1)
            real_proto_k = torch.Tensor().to(device)
            query_x = {}
            query_y = {}
            ro = []
            for k in classes:
                 Dx_k,Dy_k,Qx_k,Qy_k = extract_episode(test_x[test_y==k,:,:,:],test_y[test_y==k],n_support,device)   
                 query_x[k.item()] =  Qx_k
                 query_y[k.item()] =  Qy_k
                 ## embedding of support and reconstruction
                 _,emb_support,_,_ = model(Dx_k)              
                 emb_support = torch.flatten(emb_support,start_dim=1)
                 symbolic_proto_k = proto_sym[test_y==k,:,:,:] ## question1
                 # embedding of prototype symbol
                 _,emb_proto_k,_,_ = model(symbolic_proto_k[0,:,:,:].unsqueeze(dim=0))
                 emb_proto_k = torch.flatten(emb_proto_k,start_dim=1)
                 # real image prototype
                 # tmp = torch.mean(emb_support,dim=0).unsqueeze(dim=0)
                 # real_proto_k = torch.cat((real_proto_k,tmp),dim=0)
                 tmp = proto_rectifier(emb_support,emb_proto_k,euc=euc,wts=wts)
                 real_proto_k = torch.cat((real_proto_k,tmp),dim=0)# size should be Nc x embedding size, if not check
                 ro.append(weibull_fit(emb_support,tmp))
            ## classification and novelty detection
            Acc = 0
            
            Novelty_pred = torch.Tensor().float().to(device)
            Ground_truth = torch.Tensor().float().to(device)
            for k in classes:     
                 Qx_k = query_x[k.item()]
                 Qy_k = query_y[k.item()]
                 _,emb_query,_,tau = model(Qx_k)                         
                 emb_query = torch.flatten(emb_query,start_dim=1)  
                 emb_query = emb_enhance(emb_query,all_sym_proto_embs,real_proto_k,device,emh=emh)
                 symbolic_proto_k = proto_sym[test_y==k,:,:,:]
                 #classification prototypical
                 # if temp:
                 #    pred = cosine_classifier(emb_query,real_proto_k,device,euc=euc,test=True,tau=tau)
                 # else:
                 #    pred = cosine_classifier(emb_query,real_proto_k,device,euc=euc,test=True)  
                 logits = cosine_classifier(emb_query,real_proto_k,device,euc=euc)  
                 
                 
                 in_pred,out_pred = openmax(logits,ro,device)
                 max_pred,Hyp = torch.max(logits, dim=1)#Nq/2 x 1    
                 Acc = Acc + metrics.accuracy_score(Qy_k[:n_query//2,0].cpu().data,Hyp[:n_query//2].cpu().data)
                 # emb_query = emb_enhance(emb_query,all_sym_proto_embs,device)
                 # novelty detection
                 Novelty_pred = torch.cat((Novelty_pred,out_pred),dim=0)
                 # max_pred,_ = torch.max(in_pred, dim=1)
                 # Novelty_pred = torch.cat((Novelty_pred,max_pred),dim=0)
                 
                 Ground_truth = torch.cat((Ground_truth,Qy_k[:,1].float()),dim=0)
            # print(Ground_truth.shape)
            # Au_ROC[i] = roc_area_calc(dist=Novelty_pred.cpu().data, closed=Ground_truth.cpu().data, descending=False, total_height=Ground_truth.shape[0]//2, total_width=Ground_truth.shape[0]//2)
            fpr,tpr,_ = metrics.roc_curve(Ground_truth.cpu().data, Novelty_pred.cpu().data, pos_label=1)
            Au_ROC[i] = (metrics.auc(fpr, tpr))
            Accuracy[i] = (Acc/classes.shape[0] )
            total_count = total_count + 1 
        Au_ROC_mean = np.mean(Au_ROC)
        Accuracy_mean = np.mean(Accuracy)
        
        _,Accuracy_std = st.t.interval(0.95, len(Accuracy)-1, loc=Accuracy_mean, scale=st.sem(Accuracy)) # np.std(Accuracy)
        Accuracy_std = Accuracy_std - Accuracy_mean
        _,Au_ROC_std = st.t.interval(0.95, len(Au_ROC)-1, loc=Au_ROC_mean, scale=st.sem(Au_ROC))  #np.std(Au_ROC)
        Au_ROC_std = Au_ROC_std - Au_ROC_mean
    return Accuracy_mean,Au_ROC_mean,Accuracy_std,Au_ROC_std