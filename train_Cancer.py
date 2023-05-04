from operator import index

import pandas as pd
import logging

from sqlalchemy import true
from dataset import get_dataloader
from sklearn import metrics
from model import MLP, Transformer_effecient, Transformer
from torch.cuda.amp import autocast, GradScaler
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
parser.add_argument('--epochs', default=200, type=int, help='epochs') 
parser.add_argument('--lr', '--learning_rate', default=0.0005, type=float, help='initial learning rate')
parser.add_argument('--save_name', type=str, default='deepcat')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--resume', default='', type=str)

parser.add_argument('--loss', default='CE', type=str)

parser.add_argument('--ema', default=0.95, type=float)
parser.add_argument('--beta', default=0.333, type=float)
parser.add_argument('--warm_ratio', default=0.47, type=float)
parser.add_argument('--warm_epoch', default=8, type=int)
parser.add_argument('--label_smoothing', default=0.4, type=float)

parser.add_argument('--n_layers', default=1, type=int)
parser.add_argument('--n_heads', default=4, type=int)
parser.add_argument('--d_token', default=192, type=int)
parser.add_argument('--attention_dropout', default=0.1, type=float)
parser.add_argument('--ffn_dropout', default=0.1, type=float)
parser.add_argument('--residual_dropout', default=0.1, type=float)

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
N=70000

pos_conf=[]
neg_conf=[]

parameters={}
for i in ['batch_size','lr','n_layers','n_heads','d_token','attention_dropout','ffn_dropout','residual_dropout','weight','beta','warm_ratio','label_smoothing','warm_epoch','ema']:
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
        print(f"Check Point Loading: {key} is LOADED")
    return epoch


def save_model( save_name, save_path,net1,net2,optimizer1,optimizer2,epoch):
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
        seq=batch['seq'].view(B,-1).float().cuda()
        y=batch['y'].view(-1).long().cuda()
        index=batch['index'].cuda()
        return seq,y,B,index
    model1.train()
    scaler=GradScaler()
    loss_estimation=0
    for batch1,batch2 in zip(dataloader1,dataloader2):
        seq1,y0,B,index1=get_inputs(batch1)
        seq2,y1,B,index2=get_inputs(batch2)
        with autocast():
            pred0 = model1(torch.cat([seq1,seq2],dim=0),None).chunk(2)
            pred1 = model2(torch.cat([seq1,seq2],dim=0),None).chunk(2)
            if parameters['weight']=='mean':
                instance_weight=1/B
            elif parameters['weight']=='freq':
                pass
            ema_target1[index1]=parameters['ema']*ema_target1[index1]+(1-parameters['ema'])*(torch.softmax(pred0[0],-1)[:,1].detach())
            ema_target2[index2]=parameters['ema']*ema_target2[index2]+(1-parameters['ema'])*(torch.softmax(pred1[1],-1)[:,1].detach())
            if not warming:
                loss0 = torch.sum(loss_fn(pred0[0], torch.stack([1-ema_target2[index1],ema_target2[index1]],dim=1))*instance_weight)
                loss1 = torch.sum(loss_fn(pred1[1], torch.stack([1-ema_target1[index2],ema_target1[index2]],dim=1))*instance_weight)
            else:
                loss0 = torch.sum(loss_fn(pred0[0], y0)*instance_weight)
                loss1 = torch.sum(loss_fn(pred1[1], y1)*instance_weight)


            conf=torch.softmax(pred0[0],dim=-1).detach()#.cpu().mean()
            pos_conf.append(conf[:,1][y0==1].mean().item())
            neg_conf.append(conf[:,1][y0==0].mean().item())

        optimizer1.zero_grad()
        scaler.scale(loss0).backward()
        scaler.step(optimizer1)
        scaler.update()

        optimizer2.zero_grad()
        scaler.scale(loss1).backward()
        scaler.step(optimizer2)
        scaler.update()

        loss_estimation+=loss0.item()*B
    print(f"{loss_estimation:>7f}",end='\t')
        
def test(dataloader, model1,model2 , loss_fn,label_dir='config/CMVstatus',trainset=False):
    model1.eval()
    model2.eval()
    all_pred=[]
    all_target=[]
    with torch.no_grad():
        all_pred=[]
        for batch_idx, batch in enumerate(dataloader):
            loss_estimation=0
            B=batch['y'].shape[0]

            seq=batch['seq'].view(B,-1).float().cuda()
            y=batch['y'].view(-1).long().cuda()
            pred1 = torch.softmax(model1(seq,None),-1)
            pred2 = torch.softmax(model2(seq,None),-1)
            if trainset:
                index=batch['index'].cuda()
                ema_target1[index]=parameters['ema']*ema_target1[index]+y*(1-parameters['ema'])*(pred1[:,1])
                ema_target2[index]=parameters['ema']*ema_target2[index]+y*(1-parameters['ema'])*(pred2[:,1])

            pred = (pred1+pred2)/2
            pred = pred[:,1].cpu()

            all_pred.append(pred)
            all_target.append(y)

        all_pred=torch.cat(all_pred,dim=0).cpu().numpy()
        all_target=torch.cat(all_target,dim=0).cpu().numpy()
        fpr, tpr, thresholds = metrics.roc_curve(all_target, all_pred)
        auc=metrics.auc(fpr, tpr)
        print(f'{auc:.4f}',end='\t')

    return auc

def binary_accuracy(predict, label):
    rounded_predict = torch.round(predict) #四舍五入
    correct = (rounded_predict == label).float()
    accuracy = correct.sum() / len(correct)
    return accuracy

def creat_model(effecient=True):
    if effecient:
        model=Transformer_effecient(d_numerical=[768],categories=None,n_layers=parameters['n_layers'],d_token=parameters['d_token'],n_heads=parameters['n_heads'],d_ffn_factor=1.33333,attention_dropout=parameters['attention_dropout'],ffn_dropout=parameters['ffn_dropout'],residual_dropout=parameters['residual_dropout'],activation="reglu",prenormalization=True,initialization="kaiming",kv_compression = 1, kv_compression_sharing = 'headwise',token_bias=False,d_out=2).cuda() #,kv_compression = 0.064, kv_compression_sharing = 'headwise'
        return model
    else:
        model=Transformer(d_numerical=768,categories=None,n_layers=parameters['n_layers'],d_token=parameters['d_token'],n_heads=parameters['n_heads'],d_ffn_factor=1.33333,attention_dropout=parameters['attention_dropout'],ffn_dropout=parameters['ffn_dropout'],residual_dropout=parameters['residual_dropout'],activation="reglu",prenormalization=True,initialization="kaiming",token_bias=False,kv_compression = 0.064, kv_compression_sharing = 'headwise',d_out=2).cuda()
        return model

for fold in range(3):
    ema_target1=torch.zeros(N).cuda()
    ema_target2=torch.zeros(N).cuda()
    train_dataloader1,train_dataloader2,valid_dataloader,test_dataloader=get_dataloader(batch_size=parameters['batch_size'],co=True)
    logging.error('loading data')
    logging.error('finished loading')
    for i in train_dataloader1:
        index=i['index'].cuda()
        y=i['y'].cuda()
        ema_target1[index]=y.float()*args.label_smoothing
        ema_target2[index]=y.float()*args.label_smoothing

    model1=creat_model(effecient=False)
    model2=creat_model(effecient=False)

    optimizer1 = torch.optim.Adam(model1.parameters(), lr=parameters['lr'])
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=parameters['lr'])
    CEloss = nn.CrossEntropyLoss(reduction='none') 
    class SoftCELoss(object):
        def __call__(self, outputs, targets):
            assert outputs.shape
            Lx = -torch.sum(F.log_softmax(outputs, dim=1) * targets, dim=1)
            return Lx

    class WeightSoftCELoss(object):
        def __call__(self, outputs, targets):
            assert outputs.shape
            weights, _ = targets.max(dim=1)
            weights *= targets.shape[0] / weights.sum()
            Lx = -torch.sum(F.log_softmax(outputs, dim=1) * targets, dim=1)*weights
            return Lx
    class SCELoss():
        def __call__(self, outputs, targets):
            assert outputs.shape
            prob=F.log_softmax(outputs, dim=1)
            loss = - torch.sum(targets * torch.log(prob), dim=-1) - torch.sum(prob * torch.log(targets), dim=-1)
            return loss
    SCEloss=SoftCELoss()

    loss_fn=eval(f'{args.loss}loss')

    if len(args.resume)!=0:
        load_model(args.resume,model1,model2,optimizer1,optimizer2)
        test_acc=test(test_dataloader,model1,model2,loss_fn)
        exit()
    else:
        test(test_dataloader,model1,model2,CEloss)
        print()
        start_epoch=0

    if parameters['warm_epoch']!=-1:
        warm_epoch=parameters['warm_epoch']
    else:
        warm_epoch=int(args.epochs*parameters['warm_ratio'])

    best=0
    best_test_acc=0
    print('Epoch','\t','train loss','\t','valid AUC','\t','test AUC')
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
            save_model(f'{args.save_name}_fold{fold}','checkpoint/',model1,model2,optimizer1,optimizer2,epoch)

        print()

    print('best val auc',best, 'best test auc',best_test_acc)

