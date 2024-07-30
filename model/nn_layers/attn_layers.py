import torch
from torch import nn
from model.nn_layers.ffn import FFN
from model.nn_layers.multi_head_attn import MultiHeadAttn


'''
Adapted from OpenNMT-Py
'''

class SelfAttention(nn.Module):
    '''
    This class implements the transformer block with multi-head attention and Feed forward network
    '''
    def __init__(self, in_dim, num_heads=8, p=0.1, *args, **kwargs):
        super(SelfAttention, self).__init__()
        self.self_attn = MultiHeadAttn(input_dim=in_dim, out_dim=in_dim, num_heads=num_heads)

        self.ffn = FFN(in_dim, scale=4, p=p, expansion=True)

        self.layer_norm_1 = nn.LayerNorm(in_dim, eps=1e-6)
        self.drop = nn.Dropout(p=p)

    def forward(self, x, need_attn=False):
        '''
        :param x: Input (bags or words)
        :param need_attn: Need attention weights or not
        :return: returns the self attention output and attention weights (optional)
        '''
        x_norm = self.layer_norm_1(x)

        context, attn = self.self_attn(x_norm, x_norm, x_norm, need_attn=need_attn)

        out = self.drop(context) + x
        return self.ffn(out), attn

class MultiWaySelfAttention(nn.Module):
    '''
    This class implements the transformer block with multi-head attention and Feed forward network
    '''
    def __init__(self, in_dim, num_heads=8, p=0.1, num_way=4, *args, **kwargs):
        super(MultiWaySelfAttention, self).__init__()
        self.self_attn = MultiHeadAttn(input_dim=in_dim, out_dim=in_dim, num_heads=num_heads)

        self.way_cls_fn = nn.Linear(in_dim, num_way)
        self.ffn_lst = nn.ModuleList([FFN(in_dim, scale=4, p=p, expansion=True) for _ in range(num_way)])

        self.layer_norm_1 = nn.LayerNorm(in_dim, eps=1e-6)
        self.drop = nn.Dropout(p=p)

        self.num_way = num_way

    def forward(self, x, w, need_attn=False):
        '''
        :param x: Input (bags or words)
        :param need_attn: Need attention weights or not
        :return: returns the self attention output and attention weights (optional)
        '''
        # x: (B, N, C)
        # w: (B, N)
        x_norm = self.layer_norm_1(x)

        context, attn = self.self_attn(x_norm, x_norm, x_norm, need_attn=need_attn)

        out = self.drop(context) + x

        ffn_out = None
        for ci in range(self.num_way):
            ci_ffn_out = self.ffn_lst[ci](out)
            ci_mask = (w == ci).float().unsqueeze(dim=-1)
            if ffn_out is None:
                ffn_out = ci_ffn_out
            else:
                ffn_out = ffn_out * (1.0 - ci_mask) + ci_ffn_out * ci_mask

        # if len(x.shape) == 3: # (B, N, C)
        # ffn_out = []
        # for b in range(out.shape[0]):
        #     b_ffn_out = []
        #     for n in range(out.shape[1]):
        #         b_ffn_out.append(self.ffn_lst[int(w[b, n].item())](out[b:b+1,n:n+1]))
        #     b_ffn_out = torch.cat(b_ffn_out, dim=1)
        #     ffn_out.append(b_ffn_out)
        # ffn_out = torch.cat(ffn_out, dim=0)
        # elif len(x.shape) == 4: # (B, N1, N2, C)
        #     ffn_out = []
        #     for b in range(out.shape[0]):
        #         b_ffn_out = []
        #         for n1 in range(out.shape[1]):
        #             b_n1_ffn_out = []
        #             for n2 in range(out.shape[2]):
        #                 b_n1_ffn_out.append(self.ffn_lst[int(w[b, n1, n2].item())](out[b:b+1,n1:n1+1,n2:n2+1]))
        #             b_n1_ffn_out = torch.cat(b_n1_ffn_out, dim=2)
        #             b_ffn_out.append(b_n1_ffn_out)
        #         b_ffn_out = torch.cat(b_ffn_out, dim=1)
        #         ffn_out.append(b_ffn_out)
        #     ffn_out = torch.cat(ffn_out, dim=0)
        return ffn_out, self.way_cls_fn(out), attn


class ContextualAttention(torch.nn.Module):
    '''
        This class implements the contextual attention.
        For example, we used this class to compute bag-to-bag attention where
        one set of bag is directly from CNN, while the other set of bag is obtained after self-attention
    '''
    def __init__(self, in_dim, num_heads=8, p=0.1, *args, **kwargs):
        super(ContextualAttention, self).__init__()
        self.self_attn = MultiHeadAttn(input_dim=in_dim, out_dim=in_dim, num_heads=num_heads)

        self.context_norm = nn.LayerNorm(in_dim)
        self.context_attn = MultiHeadAttn(input_dim=in_dim, out_dim=in_dim, num_heads=num_heads)
        self.ffn = FFN(in_dim, scale=4, p=p, expansion=True)

        self.input_norm = nn.LayerNorm(in_dim, eps=1e-6)
        self.query_norm = nn.LayerNorm(in_dim, eps=1e-6)
        self.drop = nn.Dropout(p=p)

    def forward(self, input, context, need_attn=False):
        '''
        :param input: Tensor of shape (B x N_b x N_w x CNN_DIM) or (B x N_b x CNN_DIM)
        :param context: Tensor of shape (B x N_b x N_w x hist_dim) or (B x N_b x hist_dim)
        :return:
        '''

        # Self attention on Input features
        input_norm = self.input_norm(input)
        query, _ = self.self_attn(input_norm, input_norm, input_norm, need_attn=need_attn)
        query = self.drop(query) + input
        query_norm = self.query_norm(query)

        # Contextual attention
        context_norm = self.context_norm(context)
        mid, contextual_attn = self.context_attn(context_norm, context_norm, query_norm, need_attn= need_attn)
        output = self.ffn(self.drop(mid) + input)

        return output, contextual_attn

class MultiWayContextualAttention(torch.nn.Module):
    '''
        This class implements the contextual attention.
        For example, we used this class to compute bag-to-bag attention where
        one set of bag is directly from CNN, while the other set of bag is obtained after self-attention
    '''
    def __init__(self, in_dim, num_heads=8, p=0.1, num_way=6, *args, **kwargs):
        super(MultiWayContextualAttention, self).__init__()
        self.self_attn = MultiHeadAttn(input_dim=in_dim, out_dim=in_dim, num_heads=num_heads)

        self.context_norm = nn.LayerNorm(in_dim)
        self.context_attn = MultiHeadAttn(input_dim=in_dim, out_dim=in_dim, num_heads=num_heads)
        
        self.way_cls_fn = nn.Linear(in_dim, num_way)
        self.ffn_lst = nn.ModuleList([FFN(in_dim, scale=4, p=p, expansion=True) for _ in range(num_way)])

        self.input_norm = nn.LayerNorm(in_dim, eps=1e-6)
        self.query_norm = nn.LayerNorm(in_dim, eps=1e-6)
        self.drop = nn.Dropout(p=p)

        self.num_way = num_way

    def forward(self, input, context, w, need_attn=False):
        '''
        :param input: Tensor of shape (B x N_b x N_w x CNN_DIM) or (B x N_b x CNN_DIM)
        :param context: Tensor of shape (B x N_b x N_w x hist_dim) or (B x N_b x hist_dim)
        :return:
        '''

        # Self attention on Input features
        input_norm = self.input_norm(input)
        query, _ = self.self_attn(input_norm, input_norm, input_norm, need_attn=need_attn)
        query = self.drop(query) + input
        query_norm = self.query_norm(query)

        # Contextual attention
        context_norm = self.context_norm(context)
        mid, contextual_attn = self.context_attn(context_norm, context_norm, query_norm, need_attn= need_attn)
        
        #output = self.ffn(self.drop(mid) + input)
        tmp = self.drop(mid) + input

        output = None
        for ci in range(self.num_way):
            ci_output = self.ffn_lst[ci](tmp) # (B, N, C)
            ci_mask = (w == ci).float().unsqueeze(dim=-1) # (B, N, 1)
            if output is None:
                output = ci_output
            else:
                output = output * (1.0 - ci_mask) + ci_output * ci_mask

        # output = []
        # for b in range(tmp.shape[0]):
        #     b_output = []
        #     for n in range(tmp.shape[1]):
        #         b_output.append(self.ffn_lst[int(w[b, n].item())](tmp[b:b+1,n:n+1]))
        #     b_output = torch.cat(b_output, dim=1)
        #     output.append(b_output)
        # output = torch.cat(output, dim=0)

        return output, self.way_cls_fn(tmp), contextual_attn
