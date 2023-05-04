import math

import torch
import torch.nn as nn
#max_len=27

class PositionalEncoding(nn.Module):
        def __init__(self, d_model, dropout=0.1,max_len=50):
                super().__init__()
                self.dropout = nn.Dropout(p=dropout)

                pe = torch.zeros(max_len, d_model)
                position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
                div_term = torch.exp(
                        torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model)
                )
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                pe = pe.unsqueeze(0)
                self.register_buffer("pe", pe)

        def forward(self, x):
                x = x + self.pe[:, : x.size(1), :]
                return self.dropout(x)

class Mine_Transformer(nn.Module):
        def __init__( self,d_model, nhead=8, dim_feedforward=64, num_layers=1, dropout=0.1, activation="relu", classifier_dropout=0.1,fixdim=85,max_len=27):
                super().__init__()
                vocab_size=22
                self.embedding = nn.Embedding(vocab_size, d_model,padding_idx=0)
                #print(self.embedding)
                self.pos_encoder = PositionalEncoding( d_model=d_model, dropout=dropout,max_len=max_len)

                encoder_layer = nn.TransformerEncoderLayer( d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout,)
                self.transformer_encoder = nn.TransformerEncoder( encoder_layer, num_layers=num_layers,)
                #self.classifier = nn.Linear(d_model+85, 2)
                self.classifier = torch.nn.Sequential(
                                torch.nn.Linear(d_model+fixdim, 256),
                                torch.nn.BatchNorm1d(256),
                                torch.nn.ReLU(),
                                torch.nn.Linear(256, 256),
                                torch.nn.BatchNorm1d(256),
                                torch.nn.ReLU(),
                                torch.nn.Linear(256, 2),
                                
                                )
                self.d_model = d_model
                # initialization?

        def forward(self, seq, vdj=None):
                #print(seq.shape,vdj.shape)

                
                mask = seq!=0
                #print(x.shape)
                seq = self.embedding(seq)#* math.sqrt(self.d_model)
                #print(x.shape)
                #x = self.emb(x) * math.sqrt(self.d_model) ???
                seq = self.pos_encoder(seq)
                #print(x.shape)
                seq = self.transformer_encoder(seq)
                #print(x,mask.sum(dim=1,keepdim=True))
                #print(x.shape,mask.sum(dim=1,keepdim=True).shape)
                seq = seq.sum(dim=1)/mask.sum(dim=1,keepdim=True)
                #print(seq.shape,vdj.shape)
                if vdj is None:
                        out = self.classifier(seq)
                else:
                        out = self.classifier(torch.cat([seq,vdj],dim=1))
                return out

if __name__=='__main__':
        input = torch.LongTensor([[1,2,0,0],[4,3,2,9]])
        #print(input.shape)

        net=Net(8)
        print(net)
        lr = 1e-4
        optimizer = torch.optim.Adam((p for p in net.parameters() if p.requires_grad), lr=lr)

        #print(list(net.embedding.parameters()))
        output=net(input)
        exit()
        loss=torch.sum(output)
        loss.backward()
        optimizer.step()
        #print(list(net.embedding.parameters()))
