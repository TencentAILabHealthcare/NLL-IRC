from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

import pandas as pd
import logging

from sqlalchemy import true
from dataset import FastTensorDataLoader
from sklearn import metrics
from model import MLP, Transformer_effecient, Transformer
from torch.cuda.amp import autocast, GradScaler
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import os
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch CMV Training')
parser.add_argument('--batch_size', default=256, type=int, help='train batchsize') 
parser.add_argument('--epochs', default=100, type=int, help='train epochs') 
parser.add_argument('--lr', '--learning_rate', default=0.005, type=float, help='initial learning rate')
parser.add_argument('--save_name', type=str, default='co', help='save name')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--resume', default='', type=str)

parser.add_argument('--loss', default='CE', type=str)

parser.add_argument('--ema', default=0.99, type=float, help='ema coeffecient')
parser.add_argument('--warm_epoch', default=15, type=int, help='epoch for warm up')
parser.add_argument('--start_fold', default=4, type=int)
parser.add_argument('--label_smoothing', default=0.7, type=float)

parser.add_argument('--data_dir', default='data/cmv/', type=str)
parser.add_argument('--n_layers', default=1, type=int)
parser.add_argument('--n_heads', default=1, type=int)
parser.add_argument('--d_token', default=16, type=int)
parser.add_argument('--attention_dropout', default=0.0, type=float)
parser.add_argument('--ffn_dropout', default=0.0, type=float)
parser.add_argument('--residual_dropout', default=0.0, type=float)

parser.add_argument('--weight', type=str, default='mean')

args = parser.parse_args()

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.enabled = True
logging.getLogger().setLevel(logging.INFO)

vdj_dim=[68,3,14]

parameters={}
for i in ['batch_size','lr','n_layers','n_heads','d_token','attention_dropout','ffn_dropout','residual_dropout','weight','label_smoothing','warm_epoch','ema']:
    parameters[i] = eval(f'args.{i}')

def load_model(load_path,net1,net2,optimizer1,optimizer2):
    checkpoint = torch.load(load_path)
    epoch=0
    for key in checkpoint.keys():
        if 'net1' in key:
            net1.load_state_dict(checkpoint[key])
        elif 'net2' in key:
            net2.load_state_dict(checkpoint[key])
        elif key == 'optimizer1':
            optimizer1.load_state_dict(checkpoint[key])
        elif key == 'optimizer2':
            optimizer2.load_state_dict(checkpoint[key])
        elif key == 'epoch':
            epoch=checkpoint[key]
    return epoch


def save_model( save_name, save_path,net1,net2,optimizer1,optimizer2,epoch):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_filename = os.path.join(save_path, save_name)
    torch.save({'net1': net1.state_dict(),
                'net2': net2.state_dict(),
                'optimizer1': optimizer1.state_dict(),
                'optimizer2': optimizer2.state_dict(),
                'epoch': epoch,
                }, save_filename)
    logging.error(f"model saved: {save_filename}")

def train(dataloader1,dataloader2, model1,model2, loss_fn, optimizer1,optimizer2,warming=True):
    def get_inputs(batch):
        B=batch['y'].shape[0]
        seq=batch['seq'].view(B,-1).float()
        vdj=batch['vdj']
        y=batch['y'].view(-1).long().cuda()
        freq=batch['freq'].view(B).float().cuda()
        freq=freq/torch.sum(freq)
        index=torch.from_numpy(batch['index']).cuda()
        return seq,vdj,y,freq,B,index
    model1.train()
    model2.train()
    scaler=GradScaler()
    loss_estimation=0
    n=0
    for batch1,batch2 in zip(dataloader1,dataloader2):
        seq1,vdj1,y1,freq1,B,index1=get_inputs(batch1)
        seq2,vdj2,y2,freq2,B,index2=get_inputs(batch2)
        assert seq1.shape[0]==seq2.shape[0]
        n=n+1
        with autocast():
            X_num,X_cat=torch.cat([seq1,seq2],dim=0).cuda(), torch.cat([vdj1,vdj2],dim=0).cuda()
            pred1 = model1(X_num,X_cat).chunk(2)
            pred2 = model2(X_num,X_cat).chunk(2)
         
            if parameters['weight']=='mean':
                instance_weight1=1/B
                instance_weight2=1/B
            elif parameters['weight']=='freq':
                instance_weight1=freq1
                instance_weight2=freq2

            ema_target1[index1]=parameters['ema']*ema_target1[index1]+(1-parameters['ema'])*(torch.softmax(pred1[0].detach(),-1)[:,1])
            ema_target2[index2]=parameters['ema']*ema_target2[index2]+(1-parameters['ema'])*(torch.softmax(pred2[1].detach(),-1)[:,1])
            if not warming:
                loss1 = torch.sum(loss_fn(pred1[0], torch.stack([1-ema_target2[index1],ema_target2[index1]],dim=1))*instance_weight1)
                loss2 = torch.sum(loss_fn(pred2[1], torch.stack([1-ema_target1[index2],ema_target1[index2]],dim=1))*instance_weight2)
            else:
                loss1 = torch.sum(loss_fn(pred1[0], y1)*instance_weight1)
                loss2 = torch.sum(loss_fn(pred2[1], y2)*instance_weight2)

        optimizer1.zero_grad()
        scaler.scale(loss1).backward()
        scaler.step(optimizer1)
        scaler.update()

        optimizer2.zero_grad()
        scaler.scale(loss2).backward()
        scaler.step(optimizer2)
        scaler.update()
        loss_estimation+=loss1.item()*B
    print(f"{loss_estimation/dataloader1.dataset_len:>7f}",end='\t')
        


def test_associated_seq(folds, model1,model2, loss_fn,repertoire_label_dir='config/CMVstatus',associated_seq_dir='config/CMV-associated'):
    dataloader=FastTensorDataLoader(folds=folds,batch_size=parameters['batch_size'],shuffle=False,data_dir=data_dir,in_memory=True)
    with open(associated_seq_dir) as f:
        associatedSeq=eval(f.readline())
    with open(repertoire_label_dir) as f:
        labels=eval(f.readline())

    if isinstance(folds,int):
        data=pd.read_csv(f'{args.data_dir}/{folds}fold.tsv',sep='\t',header=None)
    elif isinstance(folds,list):
        data=[]
        for fold in folds:
            data.append(pd.read_csv(f'{args.data_dir}/{fold}fold.tsv',sep='\t',header=None))
        data=pd.concat(data,ignore_index=True)
    data.iloc[:, 0]=data.iloc[:, 0].map(labels)
    associated_mask=data.iloc[:, 1].isin(associatedSeq).values
    neg_mask=(data.iloc[:, 0]==0).values
    print(associated_mask.sum(),'in ',associated_mask.shape[0])
    print(neg_mask.sum(),'in ',neg_mask.shape[0])
    data.iloc[associated_mask, 0]=1

    model1.eval()
    with torch.no_grad():
        all_pred=[]
        all_y=[]
        for batch_idx, batch in enumerate(dataloader):
            loss_estimation=0
            B=batch['y'].shape[0]

            vdj=batch['vdj'].cuda()
            freq=batch['freq'].view(B,-1)

            seq=batch['seq'].view(B,-1).float().cuda()
            y=batch['y'].view(-1).long().cuda()

            sample_name=batch['sample_name']
            fake_pred=(torch.ones_like(freq)/10000).float().cuda()
            X_cat=vdj
            pred1 = torch.softmax(model1(seq,X_cat),-1)
            pred2 = torch.softmax(model2(seq,X_cat),-1)

            pred = (pred1+pred2)/2
            pred = pred[:,1].cpu()

            all_pred.append(pred)

        all_pred=torch.cat(all_pred,dim=0).numpy()
        assert all_pred.shape[0]== data.shape[0]

        label=data.iloc[:,0].values[(associated_mask | neg_mask)]
        fpr, tpr, thresholds = metrics.roc_curve(label, all_pred[(associated_mask | neg_mask)])
        auc=metrics.auc(fpr, tpr)
        print('ROC','\t',auc,'\t',fpr.tolist(),'\t',tpr.tolist())# 

    return auc

def test(dataloader, model1,model2 , loss_fn,label_dir='config/CMVstatus',threshold=None):
    model1.eval()
    model2.eval()
    test_loss, correct = 0, 0
    sample_prediction={}
    all_pred=[]
    with torch.no_grad():
        all_pred=[]
        for batch_idx, batch in enumerate(dataloader):
            B=batch['y'].shape[0]

            vdj=batch['vdj'].cuda()
            freq=batch['freq'].view(B,-1)

            seq=batch['seq'].view(B,-1).float().cuda()
            y=batch['y'].view(-1).long().cuda()

            sample_name=batch['sample_name']
            X_cat=vdj
            pred1 = torch.softmax(model1(seq,X_cat),-1)
            pred2 = torch.softmax(model2(seq,X_cat),-1)

            pred = (pred1+pred2)/2
            pred = pred[:,1].cpu()

            all_pred.append(pred)

            for b in range(B):
                if sample_name[b] not in sample_prediction:
                    sample_prediction[sample_name[b]]=[]
                sample_prediction[sample_name[b]].append([freq[b],pred[b]])

        with open(label_dir) as f:
            # print(label_dir)
            labels=eval(f.readline())
            # print(labels)
            labels=[labels[i] for i in sample_prediction]

        for i in sample_prediction:
            sample_prediction[i]=sum([i*j for i,j in sample_prediction[i]])

        sample_prediction=[sample_prediction[i].item() for i in sample_prediction]
        sample_prediction=np.asarray(sample_prediction)
        sample_prediction=sample_prediction/np.max(sample_prediction)
        labels=np.asarray(labels)

        fpr, tpr, thresholds = metrics.roc_curve(labels, sample_prediction)
        auc=metrics.auc(fpr, tpr)
        print(auc,end='\t')

        # if threshold is None:
        #     return np.median(sample_prediction) 
        # else:
        #     print(f1_score(labels, sample_prediction>threshold , average='macro'), end='\t')
        #     print(accuracy_score(labels, sample_prediction>threshold), end='\t')

    return auc

def creat_model(effecient=True):
    if effecient:
        model=Transformer_effecient(d_numerical=[768],categories=[68,3,14],n_layers=parameters['n_layers'],d_token=parameters['d_token'],n_heads=parameters['n_heads'],d_ffn_factor=1.33333,attention_dropout=parameters['attention_dropout'],ffn_dropout=parameters['ffn_dropout'],residual_dropout=parameters['residual_dropout'],activation="reglu",prenormalization=True,initialization="kaiming",kv_compression = 1, kv_compression_sharing = 'headwise',token_bias=False,d_out=2).cuda() #,kv_compression = 0.064, kv_compression_sharing = 'headwise'
        return model
    else:
        model=Transformer(d_numerical=768,categories=[68,3,14],n_layers=parameters['n_layers'],d_token=parameters['d_token'],n_heads=parameters['n_heads'],d_ffn_factor=1.33333,attention_dropout=parameters['attention_dropout'],ffn_dropout=parameters['ffn_dropout'],residual_dropout=parameters['residual_dropout'],activation="reglu",prenormalization=True,initialization="kaiming",token_bias=False,kv_compression = 0.064, kv_compression_sharing = 'headwise',d_out=2).cuda()
        return model

for cross_val in range(args.start_fold,-1,-1):
    print('cross',cross_val)
    logging.error('loading data')
    data_dir=args.data_dir
    z = list(set([0,1,2,3,4]) - set([cross_val]))
    if len(args.resume)!=0:
        test_dataloader=FastTensorDataLoader(folds=[cross_val],batch_size=parameters['batch_size'],shuffle=False,data_dir=data_dir,in_memory=True)
        valid_dataloader=FastTensorDataLoader(folds=[z[3]],batch_size=parameters['batch_size'],shuffle=False,data_dir=data_dir,in_memory=True)
    else:
        train_dataloader1=FastTensorDataLoader(folds=z[:3],batch_size=parameters['batch_size'],shuffle=True,data_dir=data_dir,in_memory=True)
        train_dataloader2=FastTensorDataLoader(folds=z[:3],batch_size=parameters['batch_size'],shuffle=True,data_dir=data_dir,in_memory=True)
        valid_dataloader=FastTensorDataLoader(folds=[z[3]],batch_size=parameters['batch_size'],shuffle=False,data_dir=data_dir,in_memory=True)
        test_dataloader=FastTensorDataLoader(folds=[cross_val],batch_size=parameters['batch_size'],shuffle=False,data_dir=data_dir,in_memory=True)
        N=train_dataloader2.dataset_len
        ema_target1=torch.zeros(N).cuda()
        ema_target2=torch.zeros(N).cuda()
        for i in train_dataloader1:
            index=torch.tensor(i['index']).cuda()
            y=i['y'].cuda()
            ema_target1[index]=y.float()*parameters['label_smoothing'] 
            ema_target2[index]=y.float()*parameters['label_smoothing']

    logging.error('finished loading')

    model1=creat_model(effecient=False)
    model2=creat_model(effecient=False)

    optimizer1 = torch.optim.Adam(model1.parameters(), lr=parameters['lr'])
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=parameters['lr'])

    CEloss = nn.CrossEntropyLoss(reduction='none') #, label_smoothing=parameters['label_smoothing']
    class SoftCELoss(object):
        def __call__(self, outputs, targets):
            probs= torch.softmax(outputs, dim=1)
            Lx = -torch.sum(F.log_softmax(outputs, dim=1) * targets, dim=1)
            return Lx
    SCEloss=SoftCELoss()

    loss_fn=eval(f'{args.loss}loss')



    if len(args.resume)!=0:

        start_epoch=load_model(args.resume,model1,model2,optimizer1,optimizer2)
        threshold=test(valid_dataloader,model1,model2,loss_fn)
        print()
        test_acc=test(test_dataloader,model1,model2,loss_fn,threshold= threshold)
        exit()
    else:
        test(test_dataloader,model1,model2,CEloss)
        print()
        start_epoch=0

    warm_epoch=parameters['warm_epoch']

    best=0
    best_test_acc=0
    print('Epoch','\t','training loss','\t','validation AUC','\t','test AUC')
    for epoch in range(start_epoch,args.epochs):
        print(epoch,end='\t')

        if epoch<warm_epoch:
            loss_fn=CEloss
            warming=True
        else:
            loss_fn=SCEloss
            warming=False

        train(train_dataloader1,train_dataloader2,model1,model2,loss_fn,optimizer1,optimizer2,warming=warming)
        current_auc=test(valid_dataloader,model1,model2,loss_fn)

        if current_auc>best:
            best=current_auc
            test_acc=test(test_dataloader,model1,model2,loss_fn)
            best_test_acc=test_acc
            save_model(f'{args.save_name}_fold{cross_val}','checkpoint/',model1,model2,optimizer1,optimizer2,epoch)

        print()

    print('best test auc',best_test_acc)
