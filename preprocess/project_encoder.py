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

from absl import flags
from absl import app
from read_skeleton import *

flags.DEFINE_string('action', None, 'axel, flip, loop, lutz, old, salchow, toe')
FLAGS = flags.FLAGS

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4, 5, 6"

checkpoint_path = "/home/r10944006/github/sport_tech_project/ckeckpoints/checkpoint_2/weight_epoch:0.pt"
batch_size = 1
device_ids = [0]
seed = 2022
skeleton_max_frame = 75

################# Reproducible ################# 

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

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

def inference(data):
    """
        inference(data)
        - input (1, 300, 17, 2)
        - output (300, 544)
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = PoseTransformer(num_frame=1, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=12,
    num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,drop_path_rate=0.1)
    model.load_state_dict(torch.load(checkpoint_path))
    models = [model.to(device) for _ in range(skeleton_max_frame)]
    [m.eval() for m in models]

    fmodel, params, buffers = combine_state_for_ensemble(models)
    [p.requires_grad_() for p in params];


    input_ids = torch.from_numpy(data)
    inputs_labels = input_ids.detach().clone()  # batch_size, 25, 3


    dataset = torch.utils.data.TensorDataset(torch.Tensor(input_ids), torch.Tensor(inputs_labels))

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers = 8)

    for batch_number,  (batch_x, batch_y) in enumerate(loader):

        batch_x = batch_x.permute(1,0,2,3)

        batch_x = batch_x.unsqueeze(1).to(device)
        batch_y = batch_y.unsqueeze(1).to(device)

        batch_x_1 = batch_x[0:75,:,:,:,:]
        batch_x_2 = batch_x[75:150,:,:,:,:]
        batch_x_3 = batch_x[150:225,:,:,:,:]
        batch_x_4 = batch_x[225:300,:,:,:,:]
        
        output_save = []
        for idx, minibatches in enumerate ([batch_x_1, batch_x_2, batch_x_3, batch_x_4]):

            outputs = [model(minibatch) for model, minibatch in zip(models, minibatches)] # 300 1 1 17 2 


            predictions1_vmap = vmap(fmodel)(params, buffers, minibatches)

            assert torch.allclose(predictions1_vmap, torch.stack(outputs), atol=1e-3, rtol=1e-5)

            outputs = torch.squeeze(predictions1_vmap) # 50 17 2
            batch_y = torch.squeeze(batch_y)

            if idx == 0:
                save_output = outputs
            else :
                save_output = torch.cat([save_output, outputs], dim=0)
        
        return save_output

def pad(array, target_shape):
    return np.pad(
        array,
        [(0, target_shape[i] - array.shape[i]) for i in range(len(array.shape))],
        "constant",
    )
    
def normalize_keypoints(X, w, h):
    """
    divide by [w,h] and map to [-1,1]
    """
    X = np.divide(X, np.array([w, h]))
    return X*2 - [1,1]

def get_video_hw(action, video):
    if (os.path.exists(os.path.join("/home/lin10/projects/SkatingJumpClassifier/20220801/", action, video, "0.jpg"))):
        img = cv2.imread(os.path.join("/home/lin10/projects/SkatingJumpClassifier/20220801/", action, video, "0.jpg"))
    else:
        img = cv2.imread(os.path.join("/home/lin10/projects/SkatingJumpClassifier/20220801/", action, video, "0000.jpg"))
    height, width, _ = img.shape
    return height, width

def main(_argv):
    for split in ["test", "train"]:
        root_dir='/home/lin10/projects/SkatingJumpClassifier/data/{}/alphapose'.format(FLAGS.action)
        with open(f'{root_dir}/{split}_list.txt') as f:
            videos = f.read().splitlines()
        for video in tqdm(videos):
            split_video_name = video.split('_')
            if (len(split_video_name) == 3):
                action = f"{split_video_name[0]}_{split_video_name[1]}"
            else:
                action = split_video_name[0]
            
            alphaposeResults = get_main_skeleton(os.path.join("/home/lin10/projects/SkatingJumpClassifier/20220801/", action, video, "alphapose-results.json"))
            keypoints_list = [np.delete(alphaposeResult[1], 2, axis=1) for alphaposeResult in alphaposeResults]
            valid_len = len(keypoints_list)
            
            ############## convert list of float64 np.array to a single float32 np.array ##############
            keypoints_array = np.stack(keypoints_list)
            
            ############## normalize joint coordinates to [-1, 1] #############
            # get height, width of video
            height, width = get_video_hw(action, video)
            normalized_keypoints = normalize_keypoints(keypoints_array[..., :2], width, height).astype(np.float32)
            
            ############## pad (T, 17, 2) to (300, 17, 2) then unsqueeze to (1, 300, 17, 2) ##############
            normalized_keypoints = pad(normalized_keypoints, (300, 17, 2))
            normalized_keypoints = np.expand_dims(normalized_keypoints, axis=0)
            
            ##### Run inferece function on normalized_keypoints, get embeddings of size (300, 544)
            embeddings = inference(normalized_keypoints)
            ##### unpad output as (T, 544)
            embeddings = embeddings[:valid_len].detach().cpu().numpy()
            ##### save output
            emb_file = os.path.join(f'/home/lin10/projects/SkatingJumpClassifier/20220801/{action}', video, "skeleton_embedding.pkl")
            with open(emb_file, "wb") as f:
                pickle.dump(embeddings, f)

"""
    single video usage: Please set action and video.
"""
# def main(_argv):
#     action = "Loop"
#     video = "Loop_13"
#     alphaposeResults = get_main_skeleton(os.path.join("/home/lin10/projects/SkatingJumpClassifier/20220801/", action, video, "alphapose-results.json"))
#     keypoints_list = [np.delete(alphaposeResult[1], 2, axis=1) for alphaposeResult in alphaposeResults]
#     valid_len = len(keypoints_list)
    
#     ############## convert list of float64 np.array to a single float32 np.array ##############
#     keypoints_array = np.stack(keypoints_list)
    
#     ############## normalize joint coordinates to [-1, 1] #############
#     # get height, width of video
#     height, width = get_video_hw(action, video)
#     normalized_keypoints = normalize_keypoints(keypoints_array[..., :2], width, height).astype(np.float32)
    
#     ############## pad (T, 17, 2) to (300, 17, 2) then unsqueeze to (1, 300, 17, 2) ##############
#     normalized_keypoints = pad(normalized_keypoints, (300, 17, 2))
#     normalized_keypoints = np.expand_dims(normalized_keypoints, axis=0)
    
#     ##### Run inferece function on normalized_keypoints, get embeddings of size (300, 544)
#     embeddings = inference(normalized_keypoints)
#     ##### unpad output as (T, 544)
#     embeddings = embeddings[:valid_len].detach().cpu().numpy()
#     ##### save output
#     emb_file = os.path.join(f'/home/lin10/projects/SkatingJumpClassifier/20220801/{action}', video, "skeleton_embedding.pkl")
#     with open(emb_file, "wb") as f:
#         pickle.dump(embeddings, f)
    
if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass

