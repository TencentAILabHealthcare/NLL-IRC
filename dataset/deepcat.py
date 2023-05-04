from torch.utils.data import Dataset, DataLoader,Subset
import copy
import random
import numpy as np
import json
import os
import torch
# def __init__(self, folds, data_dir,label_dir='config/CMVstatus',seq_encode_dir='../data/emerson-2017-natgen-seq-encoding/',in_memory=False,batch_size=32, shuffle=True):
class deepcat_dataset(Dataset): 
    def __init__(self, fold, data_dir,val_ratio=0.33): 
        # {'train':['TumorCDR3','NormalCDR3'],'test':['TumorCDR3_test','NormalCDR3_test']}

        self.fold=fold
        self.val_ratio=val_ratio
        if fold=='train_val':
            pos_data=np.load(data_dir+'TumorCDR3.npy').reshape(-1,768)
            neg_data=np.load(data_dir+'NormalCDR3.npy').reshape(-1,768)

        elif fold=='test':
            pos_data=np.load(data_dir+'TumorCDR3_test.npy').reshape(-1,768)
            neg_data=np.load(data_dir+'NormalCDR3_test.npy').reshape(-1,768)
        else:
            assert False
        pos,neg=pos_data.shape[0],neg_data.shape[0]
        # print(pos,neg)
        self.y=np.concatenate([np.ones(pos),np.zeros(neg)],axis=0)
        self.x=np.concatenate([pos_data,neg_data],axis=0)
        # print(self.x.max(),self.x.min())

        if fold=='train_val':
            N=self.y.shape[0]
            rand_index=np.random.permutation(N)
            num_to_retain=int(N*self.val_ratio)
            self.val_indices = np.sort(rand_index[:num_to_retain])
            self.train_indices = np.sort(rand_index[num_to_retain:])
            # print(self.val_indices.shape,self.train_indices.shape)

    def __getitem__(self, index):
        return {'seq':self.x[index],'y':self.y[index],'index':index}

    def __len__(self):
        return self.y.shape[0]

    def get_train_val_split(self):
        assert self.fold=='train_val'
        
        return Subset(self, self.train_indices),Subset(self, self.val_indices)

def get_dataloader_both(batch_size=32, shuffle=True,co=False):
    loader_kwargs = {'batch_size':batch_size, 'num_workers':4, 'pin_memory':True}
    train_val_data=deepcat_dataset('train_val','../data/deepcat/')
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
