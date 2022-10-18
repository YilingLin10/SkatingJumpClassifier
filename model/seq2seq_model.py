import torch.nn.functional as F
import numpy as np
import math
from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import math
from .layer.stgcn import ConvTemporalGraphical
from .utils.graph import Graph

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=6)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)
    
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
    
class Transformer(nn.Module):
    """
    Model from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/p/c80afbc9ffb1/
    """
    # Constructor
    def __init__(
        self,
        d_model,
        nhead, 
        num_class,
        num_encoder_layers,
        num_decoder_layers,
        dim_feedforward,
        dropout
    ):
        super().__init__()

        # INFO
        self.model_type = "Transformer"
        self.d_model = d_model

        # LAYERS
        self.tgt_tok_emb = TokenEmbedding(num_class, d_model)
        
        self.positional_encoder = PositionalEncoding(
            d_model=d_model, dropout=dropout, max_len=300
        )
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        # self.transformer = nn.Transformer(
        #     d_model=d_model,
        #     nhead=nhead,
        #     num_encoder_layers=num_encoder_layers,
        #     num_decoder_layers=num_decoder_layers,
        #     dropout=dropout,
        #     batch_first=True
        # )
        self.out = nn.Linear(d_model, num_class)

    def forward(self, src, tgt, tgt_mask, src_pad_mask, tgt_pad_mask):
        # Src size must be (batch_size, src sequence length)
        # Tgt size must be (batch_size, tgt sequence length)
        src = self.positional_encoder(src)
        # tgt = self.positional_encoder(tgt)
        tgt = self.positional_encoder(self.tgt_tok_emb(tgt))
        
        # Transformer blocks - Out size = (sequence length, batch_size, num_tokens)
        memory = self.encoder(src, mask=None, src_key_padding_mask=src_pad_mask)
        transformer_out = self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_pad_mask, memory_key_padding_mask=src_pad_mask )
        
        # transformer_out = self.transformer(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=tgt_pad_mask)
        transformer_out = transformer_out.contiguous().view(-1, self.d_model)
        out = self.out(transformer_out)
        
        return F.log_softmax(out, dim=1)
      
    def generate_square_subsequent_mask(self, size) -> torch.tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
        
        # EX for size=5:
        # [[0., -inf, -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0.,   0., -inf, -inf],
        #  [0.,   0.,   0.,   0., -inf],
        #  [0.,   0.,   0.,   0.,   0.]]
        
        return mask
    
    def create_mask(self, tgt):
        PAD_IDX = 6
        tgt_seq_len = tgt.shape[1]

        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len)
        tgt_padding_mask = (tgt == PAD_IDX)
        # True: <PAD>, False: non-padded
        # positions with True are not allowed to attend while False values will be unchanged.

        return tgt_mask, tgt_padding_mask