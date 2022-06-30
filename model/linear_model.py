from __future__ import absolute_import
from __future__ import print_function
import torch.nn as nn

class Net(nn.Module):
    
    def __init__(self, input_shape):
        super().__init__()
        self.fc1 = nn.Linear(51, 64)
        self.relu1 = nn.ReLU()
        self.dout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 128)
        self.prelu = nn.PReLU(1)
        self.out = nn.Linear(128, 1)
        self.out_act = nn.Sigmoid()
        
    def forward(self, input_):
        a1 = self.fc1(input_)
        h1 = self.relu1(a1)
        dout = self.dout(h1)
        a2 = self.fc2(dout)
        h2 = self.prelu(a2)
        a3 = self.out(h2)
        y = self.out_act(a3)
        return y