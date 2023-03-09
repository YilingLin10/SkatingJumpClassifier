# from new_dataset import IceSkatingDataset
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
import torch.nn.functional as F
from typing import Optional
import numpy as np
import math
from torchcrf import CRF


import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import random
from tqdm import tqdm 
import matplotlib.pyplot as plt

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model

from sklearn.metrics import r2_score,mean_squared_error

import math
import logging
from functools import partial
from collections import OrderedDict
from einops import rearrange, repeat

#from project_spatialformer import PoseTransformer, mpjpe
from functorch import combine_state_for_ensemble, vmap
import time
################################################ 
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PoseTransformer(nn.Module):
    def __init__(self, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,  norm_layer=None):
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio * num_joints   #### temporal embed_dim is num_joints * spatial embedding dim ratio
        out_dim = num_joints * 2     #### output dimension is num_joints * 3

        

        ### spatial patch embedding
        self.Spatial_patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))

        self.pos_drop = nn.Dropout(p=drop_rate)


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.Spatial_blocks = nn.ModuleList([
            Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.Spatial_norm = norm_layer(embed_dim_ratio)

        ####### A easy way to implement weighted mean
        self.weighted_mean = torch.nn.Conv1d(in_channels=num_frame, out_channels=1, kernel_size=1)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim , out_dim),
        )


    def Spatial_forward_features(self, x):
        b, _, f, p = x.shape  ##### b is batch size, f is number of frames, p is number of joints
        x = rearrange(x, 'b c f p  -> (b f) p  c', )

        x = self.Spatial_patch_to_embedding(x)
        x += self.Spatial_pos_embed
        x = self.pos_drop(x)

        for blk in self.Spatial_blocks:
            x = blk(x)

        x = self.Spatial_norm(x)
        x = rearrange(x, '(b f) w c -> b f (w c)', f=f)
        return x



    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        b, _, _, p = x.shape
        ### now x is [batch_size, 2 channels, receptive frames, joint_num], following image data
        x = self.Spatial_forward_features(x)

        #print("1 ", x.shape) #16, 1, 800, [batch/4, input_frame, dim] --> pink block each one of the skeleton 
                                                                      #--> 2 frames input, the output shape is [2, 800] 



        x = self.weighted_mean(x)

        #print("2 ", x.shape) #16, 1, 800, [batch/4, input_frame, dim]

        x = x.view(b, 1, -1) 

        #print("3 ", x.shape) #16, 1, 800, [batch/4, input_frame, dim]

        #x = self.head(x)


        #print("4 ", x.shape) #16, 1, 75

        #x = x.view(b, 1, p, -1)

        #print("5 ", x.shape)  # torch.Size([16, 1, 25, 3])

        # 2  torch.Size([1, 1, 544])
        # 3  torch.Size([1, 1, 544])
        # 4  torch.Size([1, 1, 34])
        # 5  torch.Size([1, 1, 17, 2])

        return x
################################################ 

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

class PoseTransformerEncoderCRF(nn.Module):

    def __init__(self,
                 d_model,
                 nhead, 
                 num_encoder_layers,
                 dim_feedforward,
                 dropout,
                 batch_first,
                 num_class,
                 use_crf,
                 fc_before_encoders,
                 layer_norm_eps = 1e-5,
                 activation = 'relu',
                 norm_first = False,
                 device=None, dtype=None):
        super(PoseTransformerEncoderCRF, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.poseTransformer = PoseTransformer(num_frame=1, 
                                               num_joints=17, 
                                               in_chans=2, 
                                               embed_dim_ratio=32, 
                                               depth=12,
                                               num_heads=8, 
                                               mlp_ratio=2., 
                                               qkv_bias=True, 
                                               qk_scale=None,
                                               drop_path_rate=0.1)
        self.use_crf = use_crf
        self.fc_before_encoders = fc_before_encoders
        self.set_d_model(d_model, fc_before_encoders)
        
        self.pos_encoder = PositionalEncoding(d_model=self.d_model,
                                              dropout=dropout, 
                                              max_len=300)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model,
                                                   nhead =nhead, 
                                                   dim_feedforward = dim_feedforward, 
                                                   dropout = dropout,
                                                   activation = activation, 
                                                   layer_norm_eps = layer_norm_eps, 
                                                   batch_first = batch_first, 
                                                   **factory_kwargs)
        encoder_norm = nn.LayerNorm(normalized_shape=self.d_model, 
                                    eps=layer_norm_eps, 
                                    **factory_kwargs)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self.out = nn.Linear(self.d_model, num_class)
        self.crf = CRF(num_tags=num_class, batch_first=batch_first)
        self.initialize_poseTransformer()

        skeleton_max_frame = 50

        self.models = [self.poseTransformer.to('cuda') for _ in range(skeleton_max_frame)]
        #[m.eval() for m in models]

    
    def initialize_poseTransformer(self):
        checkpoint_path = "/home/r10944006/github/sport_tech_project/ckeckpoints/checkpoint_2/weight_epoch:0.pt"
        self.poseTransformer.load_state_dict(torch.load(checkpoint_path))

        

        
    def set_d_model(self, d_model, fc_before_encoders):
        if fc_before_encoders:
            self.fc = nn.Linear(d_model, 256)
            self.d_model = 256 
        else:
            self.d_model = d_model
    
    def get_embeddings(self, src, mask):
        """
            poseformer ensemble
            input: (batch_size, 300, 17, 2)
            output: (batch_size, 300, 544)
        """
        start = time.time()
        fmodel, params, buffers = combine_state_for_ensemble(self.models)
        [p.requires_grad_() for p in params];

        # # 300 1 1 17 2 
        src = src.squeeze()
        src = src.unsqueeze(1) 
        src = src.unsqueeze(1) 

        #print(src.shape)

        batch_x_1 = src[0:50,:,:,:]
        batch_x_2 = src[50:100,:,:,:]
        batch_x_3 = src[100:150,:,:,:]
        batch_x_4 = src[150:200,:,:,:]
        batch_x_5 = src[200:250,:,:,:]
        batch_x_6 = src[250:300,:,:,:]

        
        output_save = None
        for idx, minibatches in enumerate ([batch_x_1, batch_x_2, batch_x_3, batch_x_4, batch_x_5, batch_x_6]):
            outputs = [self.poseTransformer(minibatch) for self.poseTransformer, minibatch in zip(self.models, minibatches)] # 300 1 1 17 2 
            predictions1_vmap = vmap(fmodel, randomness = "same")(params, buffers, minibatches)
            #assert torch.allclose(predictions1_vmap, torch.stack(outputs), atol=1e-3, rtol=1e-5)

            outputs = torch.squeeze(predictions1_vmap)

            #print("outputs", outputs.shape)

            if idx == 0:
                save_output = outputs
            else :
                save_output = torch.cat([save_output, outputs], dim=0)
        
        src = torch.squeeze(save_output)
        
        src = src.unsqueeze(0) 
        # src: (N, 300, 544)
        end = time.time()
        # print(f"embeddings take: {end-start}")
        return src
    
    def _get_encoder_feature(self, src, mask):
        N, T, _ = src.size()
        src = src.view(N, T, 17, 2) 
        
        ### PoseTransformer embeddings (N, 300, 17, 2) --> (N, 300, 544)
        src = self.get_embeddings(src, mask)
        ### True: not attend, False: attend
        self.src_key_padding_mask = ~mask
        
        if self.fc_before_encoders:
            src = self.fc(src)
            
        # [batch_size, seq_len, 51]
        src = self.pos_encoder(src)

        # output --> [batch_size, seq_len, 51]
        output = self.encoder(src, mask=None, src_key_padding_mask=self.src_key_padding_mask)

        if not self.use_crf:
            # output --> [batch_size * seq_len, 51]
            output = output.contiguous().view(-1, self.d_model)
        # output --> [batch_size, seq_len, 4] or [batch_size * seq_len, 4]
        output = self.out(output)
        return output
    
    def forward(self, src, mask):
        output = self._get_encoder_feature(src, mask)
        
        if self.use_crf:
            ### type: List (for prediction)
            return self.crf.decode(output, mask)
        else:
            return F.log_softmax(output, dim=1)
    
    def loss_fn(self, src, target, mask):
        pred = self._get_encoder_feature(src, mask)
        return -self.crf.forward(pred, target, mask, reduction='mean')
    
if __name__ == '__main__':
    dataset = IceSkatingDataset(dataset="all_jump",
                                split="test",
                                feature_type="raw_skeletons",
                                use_crf=True,
                                add_noise=False)
    dataloader = DataLoader(dataset,batch_size=1,
                        shuffle=False, num_workers=1, collate_fn=dataset.collate_fn)
    
    model = PoseTransformerEncoderCRF(
                    d_model = 544,
                    nhead = 8, 
                    num_encoder_layers = 2,
                    dim_feedforward = 256,
                    dropout = 0.1,
                    batch_first = True,
                    num_class = 4,
                    use_crf = True,
                    fc_before_encoders = True
                ).to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999))

    steps = 0
    for e in range(2):
        for batch_idx, sample in enumerate(dataloader):
            #calculate output
            keypoints, labels = sample['keypoints'].to('cuda'), sample['output'].to('cuda')
            mask = sample['mask'].to('cuda')
            output = model(keypoints, mask)

            #calculate loss
            loss = model.loss_fn(keypoints, labels, mask) 

            #backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            steps += 1