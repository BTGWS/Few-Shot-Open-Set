import torch
import torch.nn as nn
import torchvision
import sys
import numpy as np
from train.train_utils import *
from models.model import *
from torch.optim.lr_scheduler import MultiStepLR
# from train.tester_proto import tester 
from tqdm import tqdm


def train(model,device,train_loader,val_loader,tester,opts):

    
    max_epoch = opts.epoch
    entropy = opts.entropy
    backbone = opts.backbone
    num_classes = opts.n
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
    logger = Logger(opts.output_dir+'log_files/evaluation_logs.txt')

    optimizer = torch.optim.Adam((model.parameters()), lr=LR)
    # optimizer = torch.optim.SGD((model.parameters()), lr=LR,momentum=0.9,nesterov=True,weight_decay=5e-4)
    if entropy:
        neg_entr = NegativeEntropy()
    counter = 0
    if len(sch) != 0:
       scheduler = MultiStepLR(optimizer, milestones=sch, gamma=0.1)
    for epoch in range(1,max_epoch+1):
        # loss_class = torch.nn.CrossEntropyLoss()
        # for i, episode in enumerate(train_loader(epoch), 0):
        for i, episode in enumerate(train_loader, 0):
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
            _,emb_support,_,_ = model(data_support)  
            support_one_hot_labels = torch.zeros((data_support.shape[0], num_classes),device=data_support.device)
            support_one_hot_labels = torch.tensor(support_one_hot_labels.scatter_(1, labels_support.view(-1,1), 1))

            real_proto_k = (1/n_support)*torch.matmul(support_one_hot_labels.transpose(0,1), emb_support)
            # Divide with the number of examples per novel category.
            _,emb_query,_,tau = model(data_query)                         
            # emb_query = torch.flatten(emb_query,start_dim=1)
            if temp:
                    logits = cosine_classifier(emb_query,real_proto_k,device,euc=euc,tau=tau)
            else:
                    logits = cosine_classifier(emb_query,real_proto_k,device,euc=euc)
            pred,l_clf = loss_clf(logits,labels_query,device)
            
            optimizer.zero_grad()
            l_clf.backward()
            optimizer.step()
            print(opts.model_id+"======>"+'[%d/%d]  classification loss = %.3f ' %(epoch,max_epoch,l_clf))
            counter = counter + 1
            if counter % opts.val_check == 0  or counter == max_epoch*len(train_loader):
                model.eval()
                Accuracy,_,_,_ = tester(model=model,device=device,test_loader=val_loader,opts=opts)
                if Accuracy > best_accuracy :
                    eqn = '>'
                
                    msg = opts.model_id+"======>"+'At Epoch [{}]/[{}] \t\tCurrent Acc is {:.5f} {:s}  previous best Acc is {:.5f} '.format(epoch,
                        max_epoch,Accuracy, eqn, best_accuracy)
                    best_accuracy = Accuracy
                    best_model = model
                else:
                    eqn = '<'
                
                    msg = opts.model_id+"======>"+'At Epoch [{}]/[{}] \t\tCurrent Acc is {:.5f} {:s}  previous best Acc is {:.5f} '.format(epoch,
                        max_epoch,Accuracy, eqn, best_accuracy)
                    # best_auroc = Au_ROC
                logger(msg)
            model.train()
            if len(sch) != 0:    
                scheduler.step(counter)   
    return model,best_model
            
            

