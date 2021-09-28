import torch
import torch.nn as nn
import torchvision
import sys
import numpy as np
from train.train_utils import *
from models.model import *
from sklearn import metrics
import scipy.stats as st
import warnings
def tester(model,device,test_loader,opts):
   
    warnings.filterwarnings("ignore")
   
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

    total_count = 0
    Au_ROC = np.zeros(len(test_loader))
    Accuracy = np.zeros(len(test_loader))
    c=0;
    with torch.no_grad():
        for i, episode in enumerate(test_loader):   
            test_x, test_y, exemplar = episode
            test_x = test_x.to(device)
            test_y = test_y.to(device)
            exemplar = exemplar.to(device) 
            # data_support = test_x[:num_classes*n_support]
            labels_support_org = test_y[:num_classes*n_support]
            classes_in,labels_support = class_renumb(labels_support_org)
            # queries_x = test_x[num_classes*n_support:]
            queries_y = test_y[num_classes*n_support:]
            idx_query_in = [i for i,_ in enumerate(queries_y) if queries_y[i] in classes_in]
            idx_query_out = [i for i,_ in enumerate(queries_y) if queries_y[i] not in classes_in]

            labels_query_in = queries_y[idx_query_in]
            labels_novelty = torch.zeros(queries_y.shape[0]).to(device)
            labels_novelty[idx_query_out] = 1            
            _,labels_query_in = class_renumb(labels_query_in)

            exemplar_sup = exemplar[:num_classes*n_support]
            exemplar_query = exemplar[num_classes*n_support:]
            exemplar_query_in = exemplar_query[idx_query_in]

            emb_exemplar_sup,_,_,_,_ = model(exemplar_sup)
            emb_data,mu,log_var,recon_data,tau = model(test_x) 

            emb_support = emb_data[:num_classes*n_support]
            emb_query = emb_data[num_classes*n_support:]
            emb_query_in = emb_query[idx_query_in]
            emb_query_out = emb_query[idx_query_out]

            recon_s = recon_data[:num_classes*n_support]
            recon_q = recon_data[num_classes*n_support:]
            recon_q_in = recon_q[idx_query_in]
            recon_q_out = recon_q[idx_query_out]


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
            _,Hyp = torch.max(pred, dim=1)
            Hyp_in = Hyp[idx_query_in]
            k_exemplar = torch.stack([exemplar_sup[labels_support==i][0] for i in range(num_classes)],dim=0)
            diff_metric = Recon_diff(recon_q,k_exemplar,device)
            ND_input = torch.cat((pred,diff_metric,emb_query),dim=1) # Nq x dim+2C 
            nd_logits = model.nd_clf(ND_input)
            nd_score = nd_logits.squeeze(dim=1)  
            fpr,tpr,_ = metrics.roc_curve(labels_novelty.cpu().data, nd_score.cpu().data, pos_label=1)
            Au_ROC[i] = metrics.auc(fpr, tpr)
            Accuracy[i] = metrics.accuracy_score(labels_query_in.cpu().data,Hyp_in.cpu().data)
            total_count = total_count + 1 
            

    Au_ROC_mean = np.mean(Au_ROC)
    Accuracy_mean = np.mean(Accuracy)
    _,Accuracy_std = st.t.interval(0.95, len(Accuracy)-1, loc=Accuracy_mean, scale=st.sem(Accuracy)) # np.std(Accuracy)
    Accuracy_std = Accuracy_std - Accuracy_mean
    _,Au_ROC_std = st.t.interval(0.95, len(Au_ROC)-1, loc=Au_ROC_mean, scale=st.sem(Au_ROC))  #np.std(Au_ROC)
    Au_ROC_std = Au_ROC_std - Au_ROC_mean
    return Accuracy_mean,Au_ROC_mean,Accuracy_std,Au_ROC_std