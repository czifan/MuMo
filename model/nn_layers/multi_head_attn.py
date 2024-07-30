import torch
from torch import nn
import math
import numpy as np
from torch import Tensor
from typing import Optional

'''
This file implements the Multi-head attention
Adapted from OpenNMT-Py
'''


class MultiHeadAttn(torch.nn.Module):
    def __init__(self, input_dim, out_dim, num_heads=8, dropout=0.1, *args, **kwargs):
        super(MultiHeadAttn, self).__init__()
        assert input_dim % num_heads == 0

        self.num_heads = num_heads
        self.dim_per_head = input_dim // num_heads

        self.linear_keys = nn.Linear(input_dim, num_heads * self.dim_per_head)
        self.linear_values = nn.Linear(input_dim, num_heads * self.dim_per_head)
        self.linear_query = nn.Linear(input_dim, num_heads * self.dim_per_head)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.final_linear = nn.Linear(input_dim, out_dim)

        self.scaling_factor = math.sqrt(self.dim_per_head)

    def forward(self, key, value, query, need_attn=False):
        '''
        :param key: A tensor of shape  [B x N_b x N_w x d] or [B x N_b x d]
        :param value: A tensor of shape  [B x N_b x N_w x d] or [B x N_b x d]
        :param query: A tensor of shape  [B x N_b x N_w x d] or [B x N_b x d]
        :param need_attn: Need attention weights or not
        :return: Tuple containing output and mean attention scores across all heads (optional)
            Output size is [B x N_b x N_w x d'] or [B x N_b x d']
            Attention score size is [B x N_b*N_w x N_b*N_w] or [B x N_b x N_b]
        '''
        dim_size = key.size()
        reshape=False
        if key.dim() == 4:
            # [B x N_b x N_w x d] --> [B x N_b*N_w x d]
            key = key.view(dim_size[0], -1, dim_size[3])
            value = key.view(dim_size[0], -1, dim_size[3])
            query = key.view(dim_size[0], -1, dim_size[3])
            reshape = True

        batch_size = key.size(0)

        dim_per_head = self.dim_per_head
        head_count = self.num_heads

        key = self.linear_keys(key)
        value = self.linear_values(value)
        query = self.linear_query(query)

        query = query / self.scaling_factor

        # [B x N_b*N_w x d] --> [B x N_b*N_w x h x d_h] --> [B x h x N_b*N_w x d_h]
        key = (
            key.contiguous()
            .view(batch_size, -1, head_count, dim_per_head)
            .transpose(1, 2)
        )

        value = (
            value.contiguous()
            .view(batch_size, -1, head_count, dim_per_head)
            .transpose(1, 2)
        )

        query = (
            query.contiguous()
            .view(batch_size, -1, head_count, dim_per_head)
            .transpose(1, 2)
        )

        # compute attention scores
        # [B x h x N_b*N_w x d_h] x [B x h x d_h x N_b*N_w] --> [B x h x N_b*N_w x N_b*N_w]
        scores = torch.matmul(query, key.transpose(2, 3)).float()

        attn = self.softmax(scores).to(query.dtype)
        drop_attn = self.dropout(attn)

        # [B x h x N_b*N_w x N_b*N_w] x [B x h x N_b*N_w x d_h] --> [B x h x N_b*N_w x d_h]
        context = torch.matmul(drop_attn, value)

        # [B x h x N_b*N_w x d_h] --> [B x N_b*N_w x h x d_h] -->  [B x N_b*N_w x h*d_h]
        context = (
            context.transpose(1, 2)
            .contiguous().view(batch_size, -1, head_count * dim_per_head)
        )

        output = self.final_linear(context)

        attn_scores: Tensor[Optional] = None
        if need_attn:
            attn_scores = torch.mean(scores, dim=1)

        if reshape:
            # [B x N_b*N_w x d] --> [B x N_b x N_w x d]
            output = output.contiguous().view(dim_size[0], dim_size[1], dim_size[2], -1).contiguous()

        return output, attn_scores