import dgl.function as fn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    Graph Transformer Layer
    
"""

"""
    Util functions
"""


def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: (edges.src[src_field] * edges.dst[dst_field]).sum(-1, keepdim=True)}

    return func


def scaled_exp(field, scale_constant):
    def func(edges):
        # clamp for softmax numerical stability
        return {field: torch.exp((edges.data[field] / scale_constant).clamp(-5, 5))}

    return func


"""
    Single Attention Head
"""


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, head_dim, num_heads, use_bias):
        super().__init__()

        self.head_dim = head_dim
        self.num_heads = num_heads

        if use_bias:
            self.Q = nn.Linear(in_dim, head_dim * num_heads, bias=True)
            self.K = nn.Linear(in_dim, head_dim * num_heads, bias=True)
            self.V = nn.Linear(in_dim, head_dim * num_heads, bias=True)
        else:
            self.Q = nn.Linear(in_dim, head_dim * num_heads, bias=False)
            self.K = nn.Linear(in_dim, head_dim * num_heads, bias=False)
            self.V = nn.Linear(in_dim, head_dim * num_heads, bias=False)

    def forward(
            self,
            h: torch.Tensor  # [n, d_in]
    ):
        sequence_length = h.size()[0]
        Q_h = self.Q(h).view(sequence_length, self.num_heads, self.head_dim).transpose(0, 1)
        K_h = self.K(h).view(sequence_length, self.num_heads, self.head_dim).transpose(0, 1)
        V_h = self.V(h).view(sequence_length, self.num_heads, self.head_dim).transpose(0, 1)  # [num_heads, seq_len, d_h]
        attn_scores = torch.matmul(Q_h, K_h.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [num_heads, seq_len, seq_len]
        attn_probs = F.softmax(attn_scores, dim=-1)  # [num_heads, seq_len, seq_len]
        attn_output = torch.matmul(attn_probs, V_h)  # [num_heads, seq_len, d_h]
        attn_output = attn_output.transpose(0, 1).contiguous().view(sequence_length, self.num_heads * self.head_dim)  # [seq_len, num_heads*d_h]

        return attn_output


class GraphTransformerLayer(nn.Module):
    """
        Param: 
    """

    def __init__(self, in_dim, out_dim, num_heads, dropout=0.0, layer_norm=False, batch_norm=True, residual=True, use_bias=False):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm

        self.attention = MultiHeadAttentionLayer(in_dim, out_dim // num_heads, num_heads, use_bias)

        self.O = nn.Linear(out_dim, out_dim)

        if self.layer_norm:
            self.layer_norm1 = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm1 = nn.BatchNorm1d(out_dim)

        # FFN
        self.FFN_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_layer2 = nn.Linear(out_dim * 2, out_dim)

        if self.layer_norm:
            self.layer_norm2 = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm2 = nn.BatchNorm1d(out_dim)

    def forward(self, h):
        h_in1 = h  # for first residual connection

        # multi-head attention out
        attn_out = self.attention(h)
        h = attn_out.view(-1, self.out_channels)

        h = F.dropout(h, self.dropout, training=self.training)

        h = self.O(h)

        if self.residual:
            h = h_in1 + h  # residual connection

        if self.layer_norm:
            h = self.layer_norm1(h)

        if self.batch_norm:
            h = self.batch_norm1(h)

        h_in2 = h  # for second residual connection

        # FFN
        h = self.FFN_layer1(h)
        h = F.gelu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_layer2(h)

        if self.residual:
            h = h_in2 + h  # residual connection

        if self.layer_norm:
            h = self.layer_norm2(h)

        if self.batch_norm:
            h = self.batch_norm2(h)

        return h

    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(self.__class__.__name__,
                                                                                   self.in_channels,
                                                                                   self.out_channels, self.num_heads, self.residual)
