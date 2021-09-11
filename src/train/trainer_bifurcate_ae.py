import torch
import torch.nn as nn
import torchvision
import sys
import numpy as np
from train.train_utils import *
from models.model import *
# from train.tester import *
from torch.optim.lr_scheduler import MultiStepLR
def train(model,device,train_loader,val_loader,tester,opts):
    
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
    model = model.to(device)
    model.train()
    if entropy:
        neg_entr = NegativeEntropy()
    counter = 0
    best_accuracy = 0
    best_auroc = 0
    best_model = model
    percept = None
    optimizer = torch.optim.Adam(model.enc_module.parameters(), lr=LR)
    # optimizer = torch.optim.SGD(model.enc_module.parameters(), lr=LR, momentum=0.9, nesterov=True)
    if len(sch) != 0:
       scheduler = MultiStepLR(optimizer, milestones=sch, gamma=opts.lr_gamma)

    model.enc_module.train()
    # percept = perceptualLoss(device)
    # scheduler = MultiStepLR(optimizer, milestones=[500,800], gamma=0.1)
    for epoch in range(1,max_epoch+1):
        # l_class = 0 
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
            l_vpe_s = 0
            for k in classes:
                 Dx_k,Dy_k,Qx_k,Qy_k = extract_episode(train_x[train_y==k,:,:,:],train_y[train_y==k],n_support,device=device)  
                 query_x[k.item()] =  Qx_k
                 query_y[k.item()] =  Qy_k
                 ## embedding of support and reconstruction
                 _,emb_support,_,_ = model.encode(Dx_k)  
                 # _,emb_support,_ = model.encode(Dx_k)             
                 emb_support = torch.flatten(emb_support,start_dim=1)
                 # embedding of prototype symbol
                 symbolic_proto_k = proto_sym[train_y==k,:,:,:]
                 _,emb_proto_k,_,_ = model.encode(symbolic_proto_k[0,:,:,:].unsqueeze(dim=0))                 
                 # _,emb_proto_k,_ = model.encode(symbolic_proto_k[0,:,:,:].unsqueeze(dim=0))
                 emb_proto_k = torch.flatten(emb_proto_k,start_dim=1)
                
                 # real image prototype
                 tmp = proto_rectifier(emb_support,emb_proto_k,euc=euc,wts=wts)
                 real_proto_k = torch.cat((real_proto_k,tmp),dim=0)# size should be C x embedding size, if not check
            
            ## classification 
            l_class = 0 
            if entropy:
                l_open = 0

            all_sym_protos = torch.Tensor().to(device)
            for k in classes:                
                 symbolic_proto_k = proto_sym[train_y==k,:,:,:]
                 all_sym_protos = torch.cat((all_sym_protos,symbolic_proto_k[0,:,:,:].unsqueeze(dim=0)),dim=0)# C x dim
            all_sym_proto_embs,_,_,_ = model.encode(all_sym_protos)
            all_sym_proto_embs = torch.flatten(all_sym_proto_embs,start_dim=1)

            for k in classes:     
                 Qx_k = query_x[k.item()]
                 Qy_k = query_y[k.item()]
                 _,emb_query,_,tau = model.encode(Qx_k) 
                 # _,emb_query,_ = model.encode(Qx_k)                  
                 emb_query = torch.flatten(emb_query,start_dim=1)
                 # emb_query = emb_enhance(emb_query,all_sym_proto_embs,real_proto_k,device,emh=emh)
                 symbolic_proto_k = proto_sym[train_y==k,:,:,:] 

                 #classification prototypical
                 if temp:
                    logits = cosine_classifier(emb_query,real_proto_k,device,euc=euc,tau=tau)
                 else:
                    logits = cosine_classifier(emb_query,real_proto_k,device,euc=euc)
                 pred,l_tmp = loss_clf(logits,Qy_k[:,0],classes,device)
                 if entropy:
                    l_open = l_open + neg_entr(pred[n_query//2:])
                 l_class = l_class + l_tmp

            l_class = lambdas[1]*(l_class/classes.shape[0])
            # l_class = l_class + lambdas[1]*(l_class/classes.shape[0])
            if entropy:
                l_open = l_open/classes.shape[0]
                l_class = lambdas[1]*l_class + lambdas[3]*l_open            
            l_class.backward()
            optimizer.step()
            counter = counter+1
            if len(sch) != 0:
                scheduler.step(counter)  

            if counter % opts.val_check == 0 or counter == max_epoch*len(train_loader):
                model.eval()
                Accuracy,_,_,_ = tester(model=model,device=device,test_loader=val_loader,opts=opts)
                print('validation accuracy= %.3f '%(Accuracy))
                if Accuracy > best_accuracy :
                    best_accuracy = Accuracy
                    best_model = model
                    print('New model with validation accuracy= %.3f '%(best_accuracy))
            model.train() 
        # l_class = l_class/opts.episodes_per_epoch_train
        # l_class.backward()
        # optimizer.step()
        # if len(sch) != 0:
        #         scheduler.step(counter)  
        if entropy:
            print('[%d/%d] classification loss = %.3f and entropy = %.3f' %(epoch,max_epoch,lambdas[1]*l_class,lambdas[3]*l_open))
        else:
            print('[%d/%d] classification loss = %.3f' %(epoch,max_epoch,l_class))
    model.eval()
    best_model.eval()
    Accuracy1,_,_,_ = tester(model=model,device=device,test_loader=val_loader,opts=opts)
    Accuracy2,_,_,_ = tester(model=best_model,device=device,test_loader=val_loader,opts=opts)
    print('validation accuracy= %.3f for final model'%(Accuracy1))
    print('validation accuracy= %.3f for best validation model'%(Accuracy2))
    mdl = 2#input('choose model to continue 1 or 2:')
    if int(mdl) == 2:
      model = best_model
       
    for p in model.enc_module.parameters():
           p.requires_grad = False

    
    # optimizer = torch.optim.Adam(model.parameters(), lr=LR/10)
    optimizer = torch.optim.Adam(list(model.dec_module.parameters())+list(model.nd_module.parameters()), lr=LR)
    # optimizer = torch.optim.SGD(list(model.dec_module.parameters())+list(model.nd_module.parameters()), lr=LR, momentum=0.9, nesterov=True)
    counter = 0
    best_auroc = 0
    best_model = model
    sch = opts.schedular
    if len(sch) != 0:
       scheduler = MultiStepLR(optimizer, milestones=sch, gamma=opts.lr_gamma)
    model.train()
    max_epoch = 1000;#max_epoch
    for epoch in range(1,max_epoch+1):
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
            l_vpe_s = 0
            for k in classes:
                 Dx_k,Dy_k,Qx_k,Qy_k = extract_episode(train_x[train_y==k,:,:,:],train_y[train_y==k],n_support,device=device)  
                 query_x[k.item()] =  Qx_k
                 query_y[k.item()] =  Qy_k
                 ## embedding of support and reconstruction
                 _,emb_support,_,_,_ = model(Dx_k)              
                 emb_support = torch.flatten(emb_support,start_dim=1)
                 symbolic_proto_k = proto_sym[train_y==k,:,:,:] 
                 # embedding of prototype symbol
                 _,emb_proto_k,_,_,_ = model(symbolic_proto_k[0,:,:,:].unsqueeze(dim=0))
                 emb_proto_k = torch.flatten(emb_proto_k,start_dim=1)                
                 # real image prototype
                 tmp = proto_rectifier(emb_support,emb_proto_k,euc=euc,wts=wts)
                 real_proto_k = torch.cat((real_proto_k,tmp),dim=0)# size should be C x embedding size, if not check

                 if(backbone == "MLP"):
                    s = model.feat_extractor(symbolic_proto_k[:n_support,:,:,:])
                    l_vpe_s = l_vpe_s + loss_recon(recon,s)
                 else:
                    l_vpe_s = l_vpe_s + loss_recon(recon,symbolic_proto_k[:n_support,:,:,:])


            l_vpe_support = l_vpe_s/classes.shape[0] 
            l_vpe_q = 0
            l_nov = 0

            all_sym_protos = torch.Tensor().to(device)
            for k in classes:                
                 symbolic_proto_k = proto_sym[train_y==k,:,:,:]
                 all_sym_protos = torch.cat((all_sym_protos,symbolic_proto_k[0,:,:,:].unsqueeze(dim=0)),dim=0)# C x dim
            _,all_sym_proto_embs,_,_,_ = model(all_sym_protos)
            all_sym_proto_embs = torch.flatten(all_sym_proto_embs,start_dim=1)

            for k in classes:     
                 Qx_k = query_x[k.item()]
                 Qy_k = query_y[k.item()]
                 _,emb_query,_,recon_query,tau  = model(Qx_k)                         
                 emb_query = torch.flatten(emb_query,start_dim=1)
                 symbolic_proto_k = proto_sym[train_y==k,:,:,:]

                 if temp:
                    logits = cosine_classifier(emb_query,real_proto_k,device,euc=euc,tau=tau)
                 else:
                    logits = cosine_classifier(emb_query,real_proto_k,device,euc=euc)
                 pred,_ = loss_clf(logits,Qy_k[:,0],classes,device)

                 emb_query = emb_enhance(emb_query,all_sym_proto_embs,real_proto_k,device,emh=emh)
                 #reconstruction loss
                 if(backbone == "MLP"):
                    s = model.feat_extractor(symbolic_proto_k[:n_query,:,:,:])
                    l_vpe_q = l_vpe_q + loss_recon(recon_query,s)
                 else:
                    l_vpe_q = l_vpe_q + loss_recon(recon_query[:n_query//2],symbolic_proto_k[:n_query//2])
                 if(backbone == "MLP"):
                    als = model.feat_extractor(all_sym_protos)
                    diff_metric = Recon_diff(recon_query,als,device) # Nq x C
                 else:
                    diff_metric = Recon_diff(recon_query,all_sym_protos,device) # Nq x C

                 # novelty detection
                 ND_input = torch.cat((pred,diff_metric,emb_query),dim=1) # Nq x dim+2C 
                 nd_logits = model.nd_clf(ND_input)
                 nd_logits = nd_logits.squeeze(dim=1)
                 l_nov = l_nov + loss_novel(nd_logits,Qy_k[:,1],device)             

            l_nov = l_nov/classes.shape[0]
            l_vpe_q = l_vpe_q/classes.shape[0]
            l_vpe = l_vpe_support + l_vpe_q
            total_loss = lambdas[0]*l_vpe + lambdas[2]*l_nov
                     
            total_loss.backward()
            optimizer.step()                        
            counter = counter + 1             
            if len(sch) != 0:
               scheduler.step(counter)  
            if counter % opts.val_check == 0 or counter == 1 or counter == max_epoch*len(train_loader):
                model.eval()
                Accuracy,Au_ROC,_,_ = tester(model=model,device=device,test_loader=val_loader,opts=opts)
                print('validation accuracy= %.3f and validation AuROC= %.3f'%(Accuracy,Au_ROC))
                if Au_ROC > best_auroc :
                    best_accuracy = Accuracy
                    best_model = model
                    best_auroc = Au_ROC
                    print('New model with validation accuracy= %.3f and validation AuROC= %.3f'%(best_accuracy,best_auroc))
            model.train()
        print('[%d/%d] recon loss = %.3f,novelty detection loss = %.3f'%(epoch,max_epoch,lambdas[0]*l_vpe,lambdas[2]*l_nov))
    return model,best_model
            

