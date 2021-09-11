import torch
import torch.nn as nn
import torchvision
import sys
import numpy as np
from train.train_utils import *
from models.model import *
from torch.optim.lr_scheduler import MultiStepLR
# from train.tester_no_recon import tester 



def train(model,device,train_loader,val_loader,opts,tester):

    max_epoch = opts.epoch
    entropy = opts.entropy
    backbone = opts.backbone
    n_support = opts.k
    n_query = opts.q
    LR = opts.lr
    lambdas = opts.lambdas
    recon_loss = opts.recon
    if opts.metric == 'euclidean':
      euc = True
    else:
      euc = False
    wts = opts.weighted_mean
    emh = opts.emb_enhance
    sch = opts.schedular
    temp = opts.temperature
    # tester = get_tester(opts.tester_type)
    model = model.to(device)
    model.train()
    best_accuracy = 0
    best_auroc = 0
    best_model = model

    optimizer = torch.optim.Adam(list(model.enc_module.parameters())+list(model.nd_module.parameters()), lr=LR)
    if entropy:
        neg_entr = NegativeEntropy()
    counter = 0
    if len(sch) != 0:
       scheduler = MultiStepLR(optimizer, milestones=sch, gamma=0.1)
    for epoch in range(1,max_epoch+1):
        # loss_class = torch.nn.CrossEntropyLoss()
        for i,episode in enumerate(train_loader):
            optimizer.zero_grad()
            train_x, train_y, proto_sym = episode
            if(len(train_y.shape)>1):
                train_y = torch.squeeze(train_y,dim=1)  
            ## class rescaling  
            classes,train_y = class_scaler(train_y,n_support,n_query)
            train_x = train_x.to(device)
            train_y = train_y.to(device)
            proto_sym = proto_sym.to(device)  
            classes = classes.to(device)
            real_proto_k = torch.Tensor().to(device)
            query_x = {}
            query_y = {}
            for k in classes:
                 Dx_k,Dy_k,Qx_k,Qy_k = extract_episode(train_x[train_y==k,:,:,:],train_y[train_y==k],n_support,device=device)  
                 query_x[k.item()] =  Qx_k
                 query_y[k.item()] =  Qy_k
                 ## embedding of support and reconstruction
                 _,emb_support,_,_, = model.encode(Dx_k)              
                 emb_support = torch.flatten(emb_support,start_dim=1)
                 symbolic_proto_k = proto_sym[train_y==k,:,:,:] ## question1
                 # embedding of prototype symbol
                 _,emb_proto_k,_,_, = model.encode(symbolic_proto_k[0,:,:,:].unsqueeze(dim=0))
                 emb_proto_k = torch.flatten(emb_proto_k,start_dim=1)
                 # real image prototype
                 tmp = proto_rectifier(emb_support,emb_proto_k,euc=euc,wts=wts)
                 real_proto_k = torch.cat((real_proto_k,tmp),dim=0)# size should be C x embedding size, if not check
            
            ## classification and novelty detection
            l_class = 0 
            l_nov = 0
            l_open = 0
            all_sym_protos = torch.Tensor().to(device)
            for k in classes:                
                 symbolic_proto_k = proto_sym[train_y==k,:,:,:]
                 all_sym_protos = torch.cat((all_sym_protos,symbolic_proto_k[0,:,:,:].unsqueeze(dim=0)),dim=0)# C x dim
            _,all_sym_proto_embs,_,_ = model.encode(all_sym_protos)
            all_sym_proto_embs = torch.flatten(all_sym_proto_embs,start_dim=1)
            for k in classes:     
                 Qx_k = query_x[k.item()]
                 Qy_k = query_y[k.item()]
                 _,emb_query,_,tau = model.encode(Qx_k)                         
                 emb_query = torch.flatten(emb_query,start_dim=1)
                 emb_query = emb_enhance(emb_query,all_sym_proto_embs,real_proto_k,device,emh=emh)
                 symbolic_proto_k = proto_sym[train_y==k,:,:,:]
                
                 #classification prototypical
                 if temp:
                    logits = cosine_classifier(emb_query,real_proto_k,device,euc=euc,tau=tau)
                 else:
                    logits = cosine_classifier(emb_query,real_proto_k,device,euc=euc)
                 pred,l_tmp = loss_clf(logits,Qy_k[:,0],classes,device)
                 l_class = l_class + l_tmp
                 if entropy:
                    l_open = l_open + neg_entr(pred[n_query//2:])
                 # novelty detection
                 ND_input = torch.cat((pred,emb_query),dim=1) # Nq x dim+C emb_enhance(emb_query,all_sym_protos)
                 nd_logits = model.nd_clf(ND_input)
                 nd_logits = nd_logits.squeeze(dim=1)
                 l_nov = l_nov + loss_novel(nd_logits,Qy_k[:,1],device)

            l_nov = l_nov/classes.shape[0]
            l_class = l_class/classes.shape[0]
            total_loss =  lambdas[0]*l_class + lambdas[1]*l_nov
            if entropy:
                l_open = l_open/classes.shape[0]
                total_loss = total_loss + lambdas[2]*l_open
            total_loss.backward()
            optimizer.step()
            counter = counter + 1
            if counter % opts.val_check == 0  or counter == max_epoch*len(train_loader):
                model.eval()
                Accuracy,Au_ROC,_,_ = tester(model=model,device=device,test_loader=val_loader,opts=opts)
                if Accuracy > best_accuracy :
                    best_accuracy = Accuracy
                    best_model = model
                    best_auroc = Au_ROC
                    print('New model with validation accuracy= %.3f and validation AuROC= %.3f'%(best_accuracy,best_auroc))
            model.train()
            if len(sch) != 0:    
                scheduler.step(counter)   
        if entropy:
            print('[%d/%d]  classification loss = %.3f ,novelty detection loss = %.3f and entropy = %.3f' %(epoch,max_epoch,lambdas[0]*l_class,lambdas[1]*l_nov,lambdas[3]*l_open))
        else:
            print('[%d/%d]  classification loss = %.3f and novelty detection loss = %.3f' %(epoch,max_epoch,lambdas[0]*l_class,lambdas[1]*l_nov))
    return model,best_model
            
            

