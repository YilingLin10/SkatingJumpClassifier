import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from config import CONFIG
from typing import Optional
import numpy as np
import math

class PositionalEncoding(nn.Module):

    def __init__(self,
                 d_model, 
                 dropout=0.1, 
                 max_len=300):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model%2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term)[:,0:-1]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):

    def __init__(self,
                 d_model = 51,
                 nhead = CONFIG.NUM_HEADS, 
                 num_encoder_layers = CONFIG.NUM_ENCODER_LAYERS,
                 dim_feedforward = CONFIG.DIM_FEEDFORWARD,
                 dropout = 0.1,
                 activation = 'relu',
                 layer_norm_eps = CONFIG.LAYER_NORM_EPS,
                 batch_first = True,
                 num_class = CONFIG.NUM_CLASS,
                 norm_first = False,
                 device=None, dtype=None):
        super(TransformerModel, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.batch_first = batch_first
        self.pos_encoder = PositionalEncoding(d_model=self.d_model,
                                              dropout=dropout, 
                                              max_len=300)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                   nhead =nhead, 
                                                   dim_feedforward = dim_feedforward, 
                                                   dropout = dropout,
                                                   activation = activation, 
                                                   layer_norm_eps = layer_norm_eps, 
                                                   batch_first = batch_first, 
                                                   **factory_kwargs)
        encoder_norm = nn.LayerNorm(normalized_shape=d_model, 
                                    eps=layer_norm_eps, 
                                    **factory_kwargs)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self.num_class = num_class
        self.out = nn.Linear(d_model, self.num_class)

    def generate_src_mask(self, size):
        ### (S, S)
        ### Not allowed to attend: True
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.bool().masked_fill(mask == 0, True).masked_fill(mask == 1, False)
        return mask
    
    def generate_key_padding_mask(self, batch):
        # (N, S)
        # PAD: 1, CONSIDER:0 
        batch = batch.detach().to('cpu').numpy()
        padding_mask = np.ones((batch.shape[0], batch.shape[1]))
        for i, sample in enumerate(batch):
            # (seq_len, 51)
            for j, skeleton in enumerate(sample):
                if not np.any(skeleton):
                    break
                padding_mask[i][j] = 0
        return torch.FloatTensor(padding_mask)

    def forward(self, src):
        device = src.device
        mask = self.generate_src_mask(src.size()[1]).to(device)
        self.src_mask = mask
        self.src_key_padding_mask = self.generate_key_padding_mask(src).to(device)
        # [batch_size, seq_len, 51]
        src = self.pos_encoder(src)
        # output --> [batch_size, seq_len, 51]
        output = self.encoder(src, self.src_mask, self.src_key_padding_mask)
        # output --> [batch_size * seq_len, 51]
        output = output.contiguous().view(-1, self.d_model)
        # output --> [batch_size * seq_len, 2]
        output = self.out(output)
        output = F.log_softmax(output, dim=1)
        #output = self.out_act(output)

        return output