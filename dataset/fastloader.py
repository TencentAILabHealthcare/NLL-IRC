import torch
import numpy as np
import pandas as pd
# https://github.com/hcarlens/pytorch-tabular/blob/master/fast_tensor_data_loader.py
class FastTensorDataLoader:
    def __init__(self, folds, data_dir,label_dir='config/CMVstatus',seq_encode_dir='../data/emerson-2017-natgen-seq-encoding/',in_memory=False,batch_size=32, shuffle=True):
        
        if isinstance(folds,int):
            folds=[folds]
        self.folds = folds
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.in_memory=in_memory

        if in_memory:
            seq_encode=[]
            for i in folds:
                seq_encode.append(np.load(f'{data_dir}{i}fold.npy').reshape(-1,768))
            seq_encode=np.concatenate(seq_encode,axis=0)
            self.seq_encode=seq_encode
            seq_encode = seq_encode/np.linalg.norm(seq_encode,axis=1).reshape(-1,1)
        else:
            self.data_dir=data_dir
            self.seq_encode_dir=seq_encode_dir

        data=[]
        for i in folds:
            data.append(pd.read_csv(f'{data_dir}{i}fold.tsv',sep='\t',header=None))
        data=pd.concat(data,ignore_index=True)
        
        data.iloc[:, -3]=data.iloc[:, -3].fillna(0)
        data.iloc[:, -2]=data.iloc[:, -2].fillna(2)
        data.iloc[:, -1]=data.iloc[:, -1].fillna(13)
        self.data=data
        assert data.shape[0]==seq_encode.shape[0]
        self.dataset_len=data.shape[0]



        with open(label_dir) as f:
            self.labels=eval(f.readline())


        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            self.r=np.random.permutation(self.dataset_len)
        else:
            self.r=np.arange(self.dataset_len)
        self.index = 0
        return self

    def __next__(self):
        if self.index >= self.dataset_len:
            raise StopIteration
        batch={}
        
        current_data=self.data.iloc[self.r[self.index:self.index+self.batch_size]].values.tolist()

        batch['y']=torch.LongTensor([self.labels[i[0]] for i in current_data])
        batch['sample_name']= [i[0] for i in current_data]

        batch['vdj']=torch.LongTensor([i[4:] for i in current_data])

        batch['freq']= torch.Tensor([i[3] for i in current_data]) 

        if self.in_memory:
            batch['seq']=torch.from_numpy(self.seq_encode[self.r[self.index:self.index+self.batch_size]]) 
        else:
            seq=[]
            for i in current_data:
                t=np.load(f'{self.seq_encode_dir}{i[1]}.npy')
                seq.append(t[0])
            batch['seq']=torch.from_numpy(np.concatenate(seq,axis=0)) 
        batch['index']=self.r[self.index:self.index+self.batch_size]

        self.index += self.batch_size
        return batch

    def __len__(self):

        return self.n_batches

if __name__ == "__main__":
    pass
