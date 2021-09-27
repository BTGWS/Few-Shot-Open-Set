import torch
import torch.nn as nn
import torchvision
import sys
import numpy as np
from train.train_utils import *
from models.model import *
from sklearn import metrics
import scipy.stats as st
from tqdm import tqdm
import warnings
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
def tester(model,device,test_loader,opts):
    warnings.filterwarnings("ignore")
    backbone = opts.backbone
    num_classes = opts.n
    n_support = opts.k
    n_query = opts.q
    if opts.metric == 'euclidean':
      euc = True
    else:
      euc = False
    wts = opts.weighted_mean
    emh = opts.emb_enhance
    output_dir = opts.output_dir
    temp = opts.temperature

    total_count = 0
    Au_ROC = np.zeros(len(test_loader))
    Accuracy = np.zeros(len(test_loader))
    c=0;
    with torch.no_grad():
        # for i, episode in enumerate(test_loader(1), 0):   
        for i, episode in enumerate(test_loader, 0):
            train_x, train_y,_ = episode
            train_x = train_x.to(device)
            train_y = train_y.to(device)
            data_support = train_x[:num_classes*n_support]
            labels_support = train_y[:num_classes*n_support]
            classes_in,labels_support = class_renumb(labels_support)
            queries_x = train_x[num_classes*n_support:]
            queries_y = train_y[num_classes*n_support:]
            data_query = torch.stack([q for i,q in enumerate(queries_x) if queries_y[i] in classes_in],dim=0)
            labels_query = torch.stack([q for i,q in enumerate(queries_y) if queries_y[i] in classes_in],dim=0)
            _,labels_query = class_renumb(labels_query)
            # c=c+1
            # print(c)    
            # data_support, labels_support, data_query, labels_query, _, _ = [x.to(device) for x in episode]
            # data_support, labels_support, data_query, labels_query, _, _ = [x.to(device) for x in episode]
            # data_support = data_support.squeeze(0)
            # labels_support = labels_support.squeeze(0)
            # data_query = data_query.squeeze(0)
            # labels_query = labels_query.squeeze(0)
            # if(len(test_y.shape)>1):
            #     test_y = torch.squeeze(test_y,dim=1)  
            # classes,test_y = class_scaler(test_y,n_support,n_query)
            _,emb_support,_,_ = model(data_support)  
            support_one_hot_labels = torch.zeros((data_support.shape[0], num_classes),device=data_support.device)
            support_one_hot_labels = torch.tensor(support_one_hot_labels.scatter_(1, labels_support.view(-1,1), 1))

            real_proto_k = (1/n_support)*torch.matmul(support_one_hot_labels.transpose(0,1), emb_support)
            # Divide with the number of examples per novel category.
            _,emb_query,_,tau = model(data_query)                         
            emb_query = torch.flatten(emb_query,start_dim=1)
            if temp:
                pred = cosine_classifier(emb_query,real_proto_k,device,euc=euc,test=True,tau=tau)
            else:
                pred = cosine_classifier(emb_query,real_proto_k,device,euc=euc,test=True)  

            max_pred,Hyp = torch.max(pred, dim=1)#Nq/2 x 1  
            Accuracy[i] =  metrics.accuracy_score(labels_query.cpu().data,Hyp.cpu().data)

            
        
        Accuracy_mean = np.mean(Accuracy)
        
        _,Accuracy_std = st.t.interval(0.95, len(Accuracy)-1, loc=Accuracy_mean, scale=st.sem(Accuracy)) # np.std(Accuracy)
        Accuracy_std = Accuracy_std - Accuracy_mean
        # _,Au_ROC_std = st.t.interval(0.95, len(Au_ROC)-1, loc=Au_ROC_mean, scale=st.sem(Au_ROC))  #np.std(Au_ROC)
        # Au_ROC_std = Au_ROC_std - Au_ROC_mean
    return Accuracy_mean,Accuracy_mean,Accuracy_std,Accuracy_std