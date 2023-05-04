from torch.utils.data import Dataset, DataLoader,Subset
import copy
import random
import numpy as np
import json
import os
import torch


aa=['A','R','D','C','Q','E','H','I','G','N','L','K','M','F','P','S','T','W','Y','V','*']

def pad_1D(inputs, PAD=0, max_len=50):
        inputs=inputs+[PAD]*(max_len-len(inputs))
        return inputs

def strtolist(chain,dataaug):
        if dataaug!=0:
            res=[aa.index(i)+1 for i in chain[:3]]+[random.randint(1,21) if random.random()<dataaug else aa.index(i)+1 for i in chain[3:-1]]+[aa.index(i)+1 for i in chain[-1]]
        else:
            res=[aa.index(i)+1 for i in chain]
        return torch.LongTensor(pad_1D(res))

# def __init__(self, folds, data_dir,label_dir='config/CMVstatus',seq_encode_dir='../data/emerson-2017-natgen-seq-encoding/',in_memory=False,batch_size=32, shuffle=True):
class deepcat_dataset(Dataset): 
    def __init__(self, fold, data_dir,val_ratio=0.33,dataaug=0): 
        # {'train':['TumorCDR3','NormalCDR3'],'test':['TumorCDR3_test','NormalCDR3_test']}

        self.fold=fold
        self.val_ratio=val_ratio
        self.dataaug=dataaug
        # seqs=[]
        pos_data=[]
        neg_data=[]
        tcr=[]
        if fold=='train_val':
            file='TumorCDR3.txt' 
            with open(data_dir+file) as f:
                for seq in f.readlines():
                    pos_data.append(seq[:-1])
            file='NormalCDR3.txt'
            with open(data_dir+file) as f:
                for seq in f.readlines():
                    neg_data.append(seq[:-1])
            tcr.append(np.load(data_dir+'TumorCDR3.npy').reshape(-1,768))
            tcr.append(np.load(data_dir+'NormalCDR3.npy').reshape(-1,768))
        elif fold=='test':
            file='TumorCDR3_test.txt' 
            with open(data_dir+file) as f:
                for seq in f.readlines():
                    pos_data.append(seq[:-1])
            file='NormalCDR3_test.txt'
            with open(data_dir+file) as f:
                for seq in f.readlines():
                    neg_data.append(seq[:-1])
            tcr.append(np.load(data_dir+'TumorCDR3_test.npy').reshape(-1,768))
            tcr.append(np.load(data_dir+'NormalCDR3_test.npy').reshape(-1,768))
            # pos_data=np.load(data_dir+'TumorCDR3_test.npy').reshape(-1,768)
            # neg_data=np.load(data_dir+'NormalCDR3_test.npy').reshape(-1,768)
        else: 
            assert False
        pos,neg=len(pos_data),len(neg_data)
        # print(pos,neg)
        self.y=np.concatenate([np.ones(pos),np.zeros(neg)],axis=0)
        self.x=pos_data+neg_data
        self.tcr=np.concatenate(tcr,axis=0)
        #for i in self.x:
        #    print(i,len(i))
            
        #lens=[len(i) for i in self.x]
        #id=lens.index(max(lens))
        #print(len(self.x[id]))
        #exit()

        if fold=='train_val':
            N=self.y.shape[0]
            rand_index=np.random.permutation(N)
            num_to_retain=int(N*self.val_ratio)
            self.val_indices = np.sort(rand_index[:num_to_retain])
            self.train_indices = np.sort(rand_index[num_to_retain:])
            # print(self.val_indices.shape,self.train_indices.shape)

    def __getitem__(self, index):
        seq=strtolist(self.x[index],self.dataaug)
        return {'seq':seq ,'y':self.y[index],'index':index,'tcr':self.tcr[index]}

    def __len__(self):
        return self.y.shape[0]

    def get_train_val_split(self):
        assert self.fold=='train_val'
        
        return Subset(self, self.train_indices),Subset(self, self.val_indices)

def get_dataloader_both(batch_size=32, shuffle=True,co=False,dataaug=0):
    loader_kwargs = {'batch_size':batch_size, 'num_workers':4, 'pin_memory':True}
    train_val_data=deepcat_dataset('train_val','../data/deepcat/',dataaug=dataaug)
    print(len(train_val_data))
    train_data,val_data=train_val_data.get_train_val_split()
    test_data=deepcat_dataset('test','../data/deepcat/')

    train_loader = DataLoader( train_data, shuffle=True, **loader_kwargs)
    # for i in train_loader:
    #     print(i['index'])
    # exit()
    val_loader = DataLoader( val_data, shuffle=False, **loader_kwargs)
    test_loader = DataLoader( test_data, shuffle=False, **loader_kwargs) 
    if co:
        train_loader2 = DataLoader( train_data, shuffle=True, **loader_kwargs)
        return train_loader,train_loader2,val_loader,test_loader
    else:
        return train_loader,val_loader,test_loader

if __name__=="__main__":
    print()
