from functools import partial
from collections import OrderedDict
import torch
import torch.nn as nn
import math
import copy
from .customized_linear import CustomizedLinear
from einops import rearrange
import random
import numpy as np
import pandas as pd
import torch.nn.functional as F

def drop_set(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output

class Droped(nn.Module):
    def __init__(self, drop_prob=None):
        super(Droped, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_set(x, self.drop_prob, self.training)

class FeatureEmbeding(nn.Module):
    def __init__(self, num_genes, mask, embed_dim=96, fe_bias=True, norm_layer=None):
        super().__init__()
        self.num_genes = num_genes
        self.num_patches = mask.shape[1]
        self.embed_dim = embed_dim
        mask = np.repeat(mask,embed_dim,axis=1)
        self.mask = mask
        self.fe = CustomizedLinear(self.mask)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
    
    def forward(self, x):
        num_cells = x.shape[0]
        x = rearrange(self.fe(x), 'h (w c) -> h c w ', c=self.num_patches)
        x = self.norm(x)
        return x

class Attention(nn.Module):
    def __init__(self,
                 dim, 
                 num_heads=2,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        B, N, C = x.shape
        print(B)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        H = int(N ** 0.5)
        W = H 

        q = q.reshape(B, self.num_heads, H, W, -1)
        k = k.reshape(B, self.num_heads, H, W, -1)
        v = v.reshape(B, self.num_heads, H, W, -1)

        attn_horizontal = (q @ k.transpose(-2, -1)) * self.scale
        attn_horizontal = attn_horizontal.softmax(dim=-1)
        weight_h = attn_horizontal
        attn_horizontal = self.attn_drop(attn_horizontal)
        attn_vertical = (q.permute(0, 1, 3, 2, 4) @ k.permute(0, 1, 3, 2, 4).transpose(-2, -1)) * self.scale
        attn_vertical = attn_vertical.softmax(dim=-1)
        weight_v = attn_vertical
        attn_vertical = self.attn_drop(attn_vertical)
        weights = weight_h + weight_v       
        x_horizontal = (attn_horizontal @ v).reshape(B, N, C)
        x_vertical = (attn_vertical @ v.permute(0, 1, 3, 2, 4)).reshape(B, N, C)
        x = x_horizontal + x_vertical
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, weights


class ComplexGate(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super(ComplexGate, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return torch.sigmoid(self.fc2(self.relu(self.fc1(x)))) * x

class ComplexGatedAttention(Attention):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.,
                 gate_hidden_dim=64):

        super().__init__(dim,

                         num_heads,
                         qkv_bias,
                         qk_scale,
                         attn_drop_ratio,
                         proj_drop_ratio)

        self.q_gate = ComplexGate(dim // num_heads, gate_hidden_dim)
        self.k_gate = ComplexGate(dim // num_heads, gate_hidden_dim)
        self.v_gate = ComplexGate(dim // num_heads, gate_hidden_dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        H = int(N ** 0.5)
        W = H  # Assuming a square image
        q = q.reshape(B, self.num_heads, H, W, -1)
        k = k.reshape(B, self.num_heads, H, W, -1)
        v = v.reshape(B, self.num_heads, H, W, -1)
        
        # Apply gate to q, k and v
        q = self.q_gate(q)
        k = self.k_gate(k)
        v = self.v_gate(v)

        attn_horizontal = (q @ k.transpose(-2, -1)) * self.scale
        attn_horizontal = attn_horizontal.softmax(dim=-1)
        weight_h = attn_horizontal
        attn_horizontal = self.attn_drop(attn_horizontal)
        attn_vertical = (q.permute(0, 1, 3, 2, 4) @ k.permute(0, 1, 3, 2, 4).transpose(-2, -1)) * self.scale
        attn_vertical = attn_vertical.softmax(dim=-1)
        weight_v = attn_vertical
        attn_vertical = self.attn_drop(attn_vertical)
        weights = weight_h + weight_v
        x_horizontal = (attn_horizontal @ v).reshape(B, N, C)
        x_vertical = (attn_vertical @ v.permute(0, 1, 3, 2, 4)).reshape(B, N, C)
        x = x_horizontal + x_vertical
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, weights

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

class Block(nn.Module):

    def __init__(self,
                 dim, 
                 num_heads,
                 mlp_ratio=2.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0., 
                 attn_drop_ratio=0.,
                 drop_set_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        self.drop_set = Droped(drop_set_ratio) if drop_set_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):

        
        h, weights = self.attn(self.norm1(x))
        x = x + self.drop_set(h)
        x = x + self.drop_set(self.mlp(self.norm2(x)))
        return x, weights



def get_weight(att_mat):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    att_mat = torch.stack(att_mat).squeeze(1)
    att_mat = torch.mean(att_mat, dim=2)
    residual_att = torch.eye(att_mat.size(3))
    aug_att_mat = att_mat.to(device) + residual_att.to(device)
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)
    joint_attentions = torch.zeros(aug_att_mat.size()).to(device)
    joint_attentions[0] = aug_att_mat[0]
    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])
    v = joint_attentions[-1]
    v = v[:,0,1:]
    return v


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight.data, gain=1.414)

    def forward(self, x):
        # 图卷积操作：聚合邻居特征
        return F.relu(torch.matmul(torch.matmul(x, self.weight)))


class Transformer(nn.Module):

    def __init__(self, num_genes, mask, fe_bias=True,
                 embed_dim=192, depth=3, num_heads=1, mlp_ratio=1.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_set_ratio=0., embed_layer=FeatureEmbeding, norm_layer=None,
                 act_layer=None):
        super(Transformer, self).__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.feature_embed = embed_layer(num_genes, mask = mask, embed_dim=embed_dim, fe_bias=fe_bias)
        # GCN层
        self.gcn_layer = GCNLayer(embed_dim, embed_dim)  # 输入和输出维度相同
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        dsr = [x.item() for x in torch.linspace(0, drop_set_ratio, depth)]
        self.blocks = nn.ModuleList()
        for i in range(depth):
            layer = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                          drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_set_ratio=dsr[i],
                          norm_layer=norm_layer, act_layer=act_layer)
            self.blocks.append(copy.deepcopy(layer))
        self.norm = norm_layer(embed_dim)
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()
        # self.head = nn.Linear(self.num_features, num_set) if num_set > 0 else nn.Identity()
        # self.head_dist = None
        # if distilled:
        #     self.head_dist = nn.Linear(self.embed_dim, self.num_set) if num_set > 0 else nn.Identity()


        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)
        
    def forward_features(self, x):
        x = self.feature_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1) 
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        attn_weights = []
        tem = x
        for layer_block in self.blocks:
            tem, weights = layer_block(tem)
            attn_weights.append(weights)
        x = self.norm(tem)
        attn_weights = get_weight(attn_weights)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0]),attn_weights 
        else:
            return x[:, 0], x[:, 1],attn_weights


    def forward(self, x):
        latent, _ = self.forward_features(x)
        return latent
        
    # def forward(self, x, normalized_pos_encoding):
    #     latent, attn_weights = self.forward_features(x, normalized_pos_encoding)
    #     if self.head_dist is not None:
    #         latent, latent_dist = self.head(latent[0]), self.head_dist(latent[1])
    #         if self.training and not torch.jit.is_scripting():
    #             return latent, latent_dist
    #         else:
    #             return (latent+latent_dist) / 2
    #     else:
    #         pre = self.head(latent)
    #     return latent, pre, attn_weights

def _init_vit_weights(m):

    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)  

# def creat(num_genes,mask, embed_dim=48,depth=2,num_heads=4,has_logits: bool = True):
#     transformer = Transformer(
#                         num_genes=num_genes,
#                         mask = mask,
#                         embed_dim=embed_dim,
#                         depth=depth,
#                         num_heads=num_heads,
#                         drop_ratio=0.5, attn_drop_ratio=0.5, drop_set_ratio=0.5,
#                         representation_size=embed_dim if has_logits else None)
#
#     return transformer



