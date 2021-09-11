import torch
import torch.nn as nn
import torchvision
import sys
import numpy as np
from train.train_utils import *
from models.model import *
from sklearn import metrics
import scipy.stats as st
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
            all_sym_proto_embs,_,_,_,_ = model(all_sym_protos)  
            all_sym_proto_embs = torch.flatten(all_sym_proto_embs,start_dim=1)
            real_proto_k = torch.Tensor().to(device)
            query_x = {}
            query_y = {}
            for k in classes:
                 Dx_k,Dy_k,Qx_k,Qy_k = extract_episode(test_x[test_y==k,:,:,:],test_y[test_y==k],n_support,device)   
                 query_x[k.item()] =  Qx_k
                 query_y[k.item()] =  Qy_k
                 ## embedding of support and reconstruction
                 emb_support,_,_,_,_ = model(Dx_k)              
                 emb_support = torch.flatten(emb_support,start_dim=1)
                 symbolic_proto_k = proto_sym[test_y==k,:,:,:] ## question1
                 # embedding of prototype symbol
                 emb_proto_k,_,_,_,_ = model(symbolic_proto_k[0,:,:,:].unsqueeze(dim=0))
                 emb_proto_k = torch.flatten(emb_proto_k,start_dim=1)
                 # real image prototype
                 # tmp = torch.mean(emb_support,dim=0).unsqueeze(dim=0)
                 # real_proto_k = torch.cat((real_proto_k,tmp),dim=0)
                 tmp = proto_rectifier(emb_support,emb_proto_k,euc=euc,wts=wts)
                 real_proto_k = torch.cat((real_proto_k,tmp),dim=0)# size should be Nc x embedding size, if not check
                 
            ## classification and novelty detection
            Acc = 0
            
            Novelty_pred = torch.Tensor().float().to(device)
            Ground_truth = torch.Tensor().float().to(device)
            for k in classes:     
                 Qx_k = query_x[k.item()]
                 Qy_k = query_y[k.item()]
                 emb_query,_,_,recon_query,tau = model(Qx_k)                         
                 emb_query = torch.flatten(emb_query,start_dim=1)  
                 # emb_query = emb_enhance(emb_query,all_sym_proto_embs,real_proto_k,device,emh=emh)
                 symbolic_proto_k = proto_sym[test_y==k,:,:,:]
                 #reconstruction loss
                 if(backbone =='MLP'):
                    als = model.feat_extractor(all_sym_protos)
                    diff_metric = Recon_diff(recon_query,als,device) # Nq x C
                 else:
                    diff_metric = Recon_diff(recon_query,all_sym_protos,device) # Nq x C
                 #classification prototypical
                 if temp:
                    pred = cosine_classifier(emb_query,real_proto_k,device,euc=euc,test=True,tau=tau)
                 else:
                    pred = cosine_classifier(emb_query,real_proto_k,device,euc=euc,test=True) 
                 _,Hyp = torch.max(pred, dim=1)#Nq/2 x 1    
                 Acc = Acc + metrics.accuracy_score(Qy_k[:n_query//2,0].cpu().data,Hyp[:n_query//2].cpu().data)

                 emb_query = emb_enhance(emb_query,all_sym_proto_embs,real_proto_k,device,emh=emh)
                 # novelty detection
                 ND_input = torch.cat((pred,diff_metric),dim=1) # Nq x dim+2C
                 
                 nd_score = model.nd_clf(ND_input)
                 nd_score = nd_score.squeeze(dim=1)
                 Novelty_pred = torch.cat((Novelty_pred,nd_score),dim=0)

                 Ground_truth = torch.cat((Ground_truth,Qy_k[:,1].float()),dim=0)
            # print(Novelty_pred)
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