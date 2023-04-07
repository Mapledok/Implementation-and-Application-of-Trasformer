"""
@Time : 2023/4/7 14:12
@Author : 十三
@Email : mapledok@outlook.com
@File : MutiHeadAttention.py
@Project : Transformer
"""
import math
import torch
import torch.nn as nn
from EncoderDecoderStacks import clones


def attention(query, key, value, mask=None, dropout=None):
    """
    Compute 'Scaled Dot Product Attention'
    """

    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_model, dropout=0.1):
        """
        Take in model size and number of heads.
        """

        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        # Assume d_v = d_k
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """
        Implements Figure 2
        """

        if mask is not None:
            mask = mask.unsqueeze(1)
        batches = query.size(0)

        # Do all the linear projection in batch from d_model => num_heads x d_k
        query, key, value = [
            # [batch size, seq len, n heads, head dim]
            lin(x).view(batches, -1, self.num_heads, self.d_k).transpose(1, 2)
            # Transpose dimension 1 and 2, this puts the dimensions of each head together
            for lin, x in zip(self.linears, (query, key, value))
        ]
        # zip(self.linears, (query, key, value) =
        # [
        #     (self.linears[0], query),
        #     (self.linears[1], key),
        #     (self.linears[2], value)
        # ]

        # Apply attention on all the projected vectors in batch.
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        # 'Concat' using a view and apply a final linear
        x = (
            x.transpose(1, 2)
            # Denote that the memory of the tensor x is continuous, which can improve the computational efficiency
            .contiguous()
            # [batch size, seq len, d_model]
            .view(batches, -1, self.num_heads * self.d_k)
        )
        del query, key, value
        return self.linears[-1](x)
