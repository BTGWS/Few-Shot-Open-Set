import torch
import torch.nn as nn
import torchvision
import sys
import numpy as np
from train.train_utils import *
from models.model import *
# from train.tester import *
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import StepLR
def train(model,device,train_loader,val_loader,tester,opts):
    
    max_epoch = opts.epoch
    entropy = opts.entropy
    backbone = opts.backbone
    n_support = opts.k
    num_classes = opts.n
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
    
    if entropy:
        neg_entr = NegativeEntropy()
    
    
    percept = None
    

    

    if opts.clf_mode == 'rel_net':

        for n,p in model.named_parameters():
          if 'classifier' in n or 'enc_module' in n:
            p.requires_grad = True
          else:
            p.requires_grad = False

       
    else:
        for n,p in model.named_parameters():
          if 'enc_module' in n:
            p.requires_grad = True
          else:
            p.requires_grad = False

    param_dict = [
            {'params': model.parameters()}
        ]
    # model = nn.DataParallel(model)
    model.to(device)
    model.train()
    counter = 0
    best_accuracy = 0
    best_auroc = 0
    best_model_state_dict = model.state_dict()
    optimizer = torch.optim.Adam(param_dict, lr=LR)
    # optimizer = torch.optim.SGD(model.enc_module.parameters(), lr=LR, momentum=0.9, nesterov=True)
    if sch is not None:
       scheduler = StepLR(optimizer, sch, gamma=opts.lr_gamma)

    logger = Logger(opts.output_dir+'log_files/evaluation_logs.txt')
    # percept = perceptualLoss(device)
    # scheduler = MultiStepLR(optimizer, milestones=[500,800], gamma=0.1)
    for epoch in range(1,max_epoch+1):
        for i,episode in enumerate(train_loader):
            train_x, train_y, exemplar = episode
            train_x = train_x.to(device)
            train_y = train_y.to(device)
            exemplar = exemplar.to(device) 
            # data_support = train_x[:num_classes*n_support]
            labels_support_org = train_y[:num_classes*n_support]
            classes_in,labels_support = class_renumb(labels_support_org)
            # queries_x = train_x[num_classes*n_support:]
            queries_y = train_y[num_classes*n_support:]
            idx_query_in = [i for i,_ in enumerate(queries_y) if queries_y[i] in classes_in]
            idx_query_out = [i for i,_ in enumerate(queries_y) if queries_y[i] not in classes_in]
            labels_query_in = queries_y[idx_query_in]
            _,labels_query_in = class_renumb(labels_query_in)
            
            exemplar_sup = exemplar[:num_classes*n_support]
            emb_exemplar_sup,_,_,_ = model.encode(exemplar_sup)
            emb_data,_,_,tau = model.encode(train_x)

            # k_exemplar = torch.stack([exemplar_sup[labels_support==i][0] for i in range(num_classes)],dim=0)            
            # k_exemplar_emb = torch.stack([exemplar_sup[labels_support==i][0] for i in range(num_classes)],dim=0)

            emb_support = emb_data[:num_classes*n_support]
            emb_query = emb_data[num_classes*n_support:]
            
            real_proto_k = proto_rectifier(emb_support=emb_support,emb_proto_k=emb_exemplar_sup,labels_support=labels_support,n_support=n_support,num_classes=num_classes,euc=euc,wts=wts)
            if opts.pre_emh:
                emb_query = emb_enhance(emb_query,real_proto_k,device,emh=emh)
            if opts.clf_mode == 'rel_net':
                emb_query_ = emb_query.unsqueeze(0).repeat(num_classes,1,1)#C x Nq x dim
                emb_query_ = emb_query_.transpose(0,1) #Nq x C x dim
                real_proto_k_ = real_proto_k.unsqueeze(0).repeat(emb_query.shape[0],1,1)
                relation_pairs = torch.cat((real_proto_k_,emb_query_),2).view(-1,2*emb_query.shape[1])
                logits = model.classify(relation_pairs).view(emb_query.shape[0],num_classes)
            else:
                if temp is not None:
                        logits = cosine_classifier(emb_query,real_proto_k,device,euc=euc,tau=tau)
                else:
                        logits = cosine_classifier(emb_query,real_proto_k,device,euc=euc)
            if not opts.pre_emh:
                emb_query = emb_enhance(emb_query,real_proto_k,device,emh=emh)
            logits_in = logits[idx_query_in]
            pred,l_clf = loss_clf(logits_in,labels_query_in,device,opts.clf_mode)
            
            optimizer.zero_grad()
            l_clf.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step() 
            
            counter = counter + 1 
             
            print(str(n_support)+'_shot_'+opts.model_id+"======>"+'[%d/%d]  classification loss = %.3f ' %(epoch,max_epoch,l_clf))
            if counter % opts.val_check == 0  or counter == max_epoch*len(train_loader):
                model.eval()
                Accuracy,_,_,_ = tester(model=model,device=device,test_loader=val_loader,opts=opts)
                if Accuracy > best_accuracy :
                    eqn = '>'
                
                    msg = str(n_support)+'_shot_'+opts.model_id+"======>"+'At Epoch [{}]/[{}] \t\tCurrent Acc is {:.5f} {:s}  previous best Acc is {:.5f} '.format(epoch,
                        max_epoch,Accuracy, eqn, best_accuracy)
                    best_accuracy = Accuracy
                    best_model_state_dict = model.state_dict()
                else:
                    eqn = '<'
                
                    msg = str(n_support)+'_shot_'+opts.model_id+"======>"+'At Epoch [{}]/[{}] \t\tCurrent Acc is {:.5f} {:s}  previous best Acc is {:.5f} '.format(epoch,
                        max_epoch,Accuracy, eqn, best_accuracy)
                logger(msg)
        if sch is not None:
                scheduler.step() 
    
    model.load_state_dict(best_model_state_dict)
    # model = nn.DataParallel(model)
    model.to(device)
    for n,p in model.named_parameters():
          if 'dec_module' in n or 'nd_module' in n:
            p.requires_grad = True
          else:
            p.requires_grad = False

    param_dict = [
            {'params': model.parameters()}
        ]
   
    # optimizer = torch.optim.Adam(model.parameters(), lr=LR/10)
    assert opts.lr_decoder is not None
    optimizer = torch.optim.Adam(param_dict, lr=opts.lr_decoder)
    # optimizer = torch.optim.SGD(list(model.dec_module.parameters())+list(model.nd_module.parameters()), lr=LR, momentum=0.9, nesterov=True)
    counter = 0
    best_auroc = 0
    best_model_state_dict = model.state_dict()
    sch = sch
    if sch is not None:
       scheduler = StepLR(optimizer, sch, gamma=opts.lr_gamma)
    model.train()
    # max_epoch = 1000;#max_epoch
    for epoch in range(1,max_epoch+1):
        for i,episode in enumerate(train_loader):

            train_x, train_y, exemplar = episode
            train_x = train_x.to(device)
            train_y = train_y.to(device)
            exemplar = exemplar.to(device) 
            # data_support = train_x[:num_classes*n_support]
            labels_support_org = train_y[:num_classes*n_support]
            classes_in,labels_support = class_renumb(labels_support_org)
            # queries_x = train_x[num_classes*n_support:]
            queries_y = train_y[num_classes*n_support:]
            idx_query_in = [i for i,_ in enumerate(queries_y) if queries_y[i] in classes_in]
            idx_query_out = [i for i,_ in enumerate(queries_y) if queries_y[i] not in classes_in]

            labels_query_in = queries_y[idx_query_in]
            _,labels_query_in = class_renumb(labels_query_in)

            labels_novelty = torch.zeros(queries_y.shape[0]).to(device)
            labels_novelty[idx_query_out] = 1            
            

            exemplar_sup = exemplar[:num_classes*n_support]
            exemplar_query = exemplar[num_classes*n_support:]
            exemplar_query_in = exemplar_query[idx_query_in]

            emb_exemplar_sup,_,_,_,_ = model(exemplar_sup)
            emb_data,mu,log_var,recon_data,tau = model(train_x) 

            emb_support = emb_data[:num_classes*n_support]
            emb_query = emb_data[num_classes*n_support:]
            emb_query_in = emb_query[idx_query_in]
            emb_query_out = emb_query[idx_query_out]

            recon_s = recon_data[:num_classes*n_support]
            recon_q = recon_data[num_classes*n_support:]
            recon_q_in = recon_q[idx_query_in]
            recon_q_out = recon_q[idx_query_out]

            mu_s = mu[:num_classes*n_support]
            mu_q = mu[num_classes*n_support:]
            mu_q_in = mu_q[idx_query_in]

            log_var_s = log_var[:num_classes*n_support]
            log_var_q = log_var[num_classes*n_support:]
            log_var_q_in = log_var_q[idx_query_in]

            l_vpe_s = loss_vpe(recon_s,exemplar_sup,mu_s,log_var_s,device,percept=percept,recon = recon_loss)
            l_vpe_q = loss_vpe(recon_q_in,exemplar_query_in,mu_q_in,log_var_q_in,device,percept=percept,recon = recon_loss)

            l_vpe = (1/(exemplar_sup.shape[0]+exemplar_query_in.shape[0]))*(l_vpe_s+l_vpe_q)

            real_proto_k = proto_rectifier(emb_support=emb_support,emb_proto_k=emb_exemplar_sup,labels_support=labels_support,n_support=n_support,num_classes=num_classes,euc=euc,wts=wts)
            if opts.pre_emh:
                emb_query = emb_enhance(emb_query,real_proto_k,device,emh=emh)
            if opts.clf_mode == 'rel_net':
                emb_query_ = emb_query.unsqueeze(0).repeat(num_classes,1,1)#C x Nq x dim
                emb_query_ = emb_query_.transpose(0,1) #Nq x C x dim
                real_proto_k_ = real_proto_k.unsqueeze(0).repeat(emb_query.shape[0],1,1)
                relation_pairs = torch.cat((real_proto_k_,emb_query_),2).view(-1,2*emb_query.shape[1])
                logits = model.classify(relation_pairs).view(emb_query.shape[0],num_classes)
                pred = logits
            else:
                if temp is not None:
                        logits = cosine_classifier(emb_query,real_proto_k,device,euc=euc,tau=tau)
                        pred = logits.softmax(-1)
                else:
                        logits = cosine_classifier(emb_query,real_proto_k,device,euc=euc)
                        pred = logits.softmax(-1)
            if not opts.pre_emh:
                emb_query = emb_enhance(emb_query,real_proto_k,device,emh=emh)

            k_exemplar = torch.stack([exemplar_sup[labels_support==i][0] for i in range(num_classes)],dim=0)
            diff_metric = Recon_diff(recon_q,k_exemplar,device)
            ND_input = torch.cat((pred,diff_metric,emb_query),dim=1) # Nq x dim+2C 
            nd_logits = model.nd_clf(ND_input)
            nd_logits = nd_logits.squeeze(dim=1)
            l_nov = loss_novel(nd_logits,labels_novelty,device)     
            
            total_loss = lambdas[0]*l_vpe + lambdas[2]*l_nov
            optimizer.zero_grad()         
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()  

            counter = counter + 1    
            print(str(n_support)+'_shot_'+opts.model_id+"======>"+'[%d/%d] recon loss = %.3f,novelty detection loss = %.3f'\
                %(epoch,max_epoch,lambdas[0]*l_vpe,lambdas[2]*l_nov))         
            
            if counter % opts.val_check == 0  or counter == max_epoch*len(train_loader):
                model.eval()
                Accuracy,Au_ROC,Accuracy_std,Au_ROC_std = tester(model=model,device=device,test_loader=val_loader,opts=opts)
                if Au_ROC > best_auroc :
                    eqn = '>'
                
                    msg = str(n_support)+'_shot_'+opts.model_id+"======>"+'At Epoch [{}]/[{}] \t\tCurrent Acc is {:.5f} + {:.5f} and current Auroc is {:.5f} + {:.5f} \
                     {:s} previous best Auroc is {:.5f} '.format(epoch,max_epoch,Accuracy,Accuracy_std,Au_ROC,Au_ROC_std, eqn, best_auroc)
                    best_auroc = Au_ROC
                    best_model_state_dict = model.state_dict()
                else:
                    eqn = '<'
                
                    msg = str(n_support)+'_shot_'+opts.model_id+"======>"+'At Epoch [{}]/[{}] \t\tCurrent Acc is {:.5f} + {:.5f} and current Auroc is {:.5f} + {:.5f} \
                     {:s} previous best Auroc is {:.5f} '.format(epoch,max_epoch,Accuracy,Accuracy_std,Au_ROC,Au_ROC_std, eqn, best_auroc)
                    
                logger(msg)
        if sch is not None:
            scheduler.step(counter)  
    return model.state_dict(),best_model_state_dict
            

