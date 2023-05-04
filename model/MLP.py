import torch
import torch.nn as nn
import torch.nn.functional as F
class MLP(nn.Module):
    def __init__(self,layers=[768+1+68+3+14,128,1],dropout=0.5):
      super(MLP, self).__init__()
      
      self.fc1 = nn.Linear(layers[0], layers[1])
      self.fc2 = nn.Linear(layers[1], layers[2])
      self.dropout = nn.Dropout(p=dropout)

    # x represents our data
    def forward(self, x):
      x = self.dropout(F.relu(self.fc1(x)))
      x = self.fc2(x)
      return x