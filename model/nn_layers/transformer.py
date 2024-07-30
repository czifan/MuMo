import torch.nn as nn 
import torch 
import torchvision.models as models 
import os 
from torch.nn import functional as F
import numpy as np
from torch.nn import init

def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)

def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)

class FeedForward(nn.Module):
    def __init__(self, dim, out_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
    
class MultiHeadedSelfAttention(nn.Module):
    """Multi-Headed Dot Product Attention"""
    def __init__(self, feat_dim, dim, num_heads, dropout):
        super().__init__()
        self.proj_q = nn.Linear(feat_dim, dim)
        self.proj_k = nn.Linear(feat_dim, dim)
        self.proj_v = nn.Linear(feat_dim, dim)
        self.drop = nn.Dropout(dropout)
        self.n_heads = num_heads
        self.scores = None # for visualization

    def forward(self, x, mask):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [q, k, v])
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        #scores = torch.cosine_similarity(q[:, :, :, None, :], q[:, :, None, :, :], dim=-1)
        if mask is not None:
            mask = mask[:, None, None, :].float()
            scores -= 10000.0 * (1.0 - mask)
        attn_map = F.softmax(scores, dim=-1)
        scores = self.drop(attn_map)
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        h = merge_last(h, 2)
        self.scores = attn_map
        return h
    
class Block(nn.Module):
    """Transformer Block"""
    def __init__(self, dim, num_heads, dropout):
        super().__init__()
        self.attn = MultiHeadedSelfAttention(dim, dim, num_heads, dropout)
        self.attn_norm = nn.LayerNorm(dim)
        self.mlp = nn.Linear(dim, dim)
        self.mlp_norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, feat_x, mask):
        attn_x = self.attn(feat_x, mask)
        attn_x = self.attn_norm(attn_x)
        attn_x = attn_x + feat_x 
        mlp_x  = self.mlp(attn_x)
        mlp_x  = self.mlp_norm(mlp_x)
        mlp_x  = self.drop(mlp_x)
        out_x  = mlp_x + attn_x
        return out_x
    
class SimpleTransformer(nn.Module):
    def __init__(self,
                in_dim,
                num_head,
                dropout,
                num_attn,
                merge_token=False):
        super().__init__()
        self.merge_token = merge_token
        if self.merge_token:
            self.token = nn.Parameter(torch.zeros(1, 1, in_dim).float())
            self.pe_token = nn.Parameter(torch.zeros(1, 1, in_dim).float())
        else:
            self.weight_fc = nn.Linear(in_dim, 1, bias=True)
            self.weight = None

        self.attn_layer_lst = nn.ModuleList([
            Block(in_dim, num_head, dropout) for _ in range(num_attn)
        ])

        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)  # nn.init.constant(m.bias, 0)
        self.apply(_init)
        if self.merge_token:
            nn.init.constant_(self.token, 0)
            nn.init.constant_(self.pe_token, 0)

    def forward(self, x, mask, pe):
        # x: (B, T, C)
        # mask: (B, T)
        if self.merge_token:
            x = torch.cat([self.token.expand(x.shape[0], 1, -1).to(x.device), x], dim=1)
            mask = torch.cat([torch.ones(mask.shape[0], 1).float().to(mask.device), mask], dim=1)
            if pe is not None:
                pe = torch.cat([self.pe_token.expand(pe.shape[0], 1, -1).to(pe.device), pe], dim=1)
        for attn_layer in self.attn_layer_lst:
            if pe is not None:
                x = x + pe
            x = attn_layer(x, mask)
        if self.merge_token:
            return x[:, 0]
        else:
            return x