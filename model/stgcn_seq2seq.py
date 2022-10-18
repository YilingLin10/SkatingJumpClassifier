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
    
class GCN_Transformer(nn.Module):
    """
    Model from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/p/c80afbc9ffb1/
    """
    # Constructor
    def __init__(
        self,
        hidden_channel,
        out_channel,
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

        # LAYERS
        # ==== STGCN ====
        ## ST-GCN's GCN block
        self.out_channel = out_channel
        self.num_node = 17
        in_channels = 2
        self.self_link = [(i, i) for i in range(self.num_node)]
        self.inward = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 6),
                    (5, 7), (6, 8), (7, 9), (8, 10), (5, 11), (6, 12),
                    (11, 12), (11, 13), (12, 14), (13, 15), (14, 16)]
        self.outward = [(j, i) for (i, j) in self.inward]
        self.neighbor = self.inward + self.outward
        graph_args = {
            'num_node': self.num_node,
            'self_link': self.self_link,
            'inward': self.inward,
            'outward': self.outward,
            'neighbor': self.neighbor
        }
        self.graph = Graph(**graph_args)
        self.A = torch.from_numpy(self.graph.A.astype(np.float32))
        self.gcn1 = ConvTemporalGraphical(in_channels=in_channels, out_channels=hidden_channel, kernel_size=self.A.shape[0])
        self.gcn2 = ConvTemporalGraphical(in_channels=hidden_channel, out_channels=out_channel, kernel_size=self.A.shape[0])
        self.data_bn = nn.BatchNorm1d(self.num_node * in_channels)
        self.edge_importance = nn.Parameter(torch.ones(self.A.shape))
        self.d_model = self.out_channel
        # ===============
        self.tgt_tok_emb = TokenEmbedding(num_class, out_channel)
        
        self.positional_encoder = PositionalEncoding(
            d_model=self.d_model, dropout=dropout, max_len=300
        )
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.out = nn.Linear(self.d_model, num_class)

        self.softmax = nn.Softmax(dim=-1)  
    def _gcn_embeddings(self, src):
        A = self.A.cuda(src.get_device())
        ## TODO: Batch normalization
        ## src = [batch_size, seq_len, 34] --> [batch_size, 34, seq_len]
        src = src.permute(0, 2, 1)
        src = self.data_bn(src)
        ## src = [batch_size, 34, seq_len] --> [batch_size, seq_len, 34]
        src = src.permute(0, 2, 1)
        ## TODO: ST-GCN embeddings
        N, T, _ = src.size()
        
        # input src = [batch_size, seq_len, 34] --> [batch_size, 2, seq_len, 17]
        src = src.view(N, T, self.num_node, 2).permute(0, 3, 1, 2)
        src, A = self.gcn1(src, A * self.edge_importance)
        src, A = self.gcn2(src, A * self.edge_importance)

        ## output of agcn = [batch_size, out_channel, seq_len, 17] --> [batch_size, seq_len, out_channel]
        # src = src.mean(dim=3)
        src, _ = torch.max(src, 3)
        src = src.permute(0, 2, 1)
        return src

    def forward(self, src, tgt, tgt_mask, src_pad_mask, tgt_pad_mask):
        # Src size must be (batch_size, src sequence length)
        # Tgt size must be (batch_size, tgt sequence length)
        src = self._gcn_embeddings(src)
        tgt = self.tgt_tok_emb(tgt)
        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)
        
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