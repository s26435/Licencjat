import torch
from torch import nn
from torch.nn import functional as F
import math


class SelfAttention(nn.Module):
    def __init__(
        self, n_heads: int, d_embed: int, in_bias: bool = True, out_bias: bool = False
    ):
        super(SelfAttention, self).__init__()
        self.in_p = nn.Linear(d_embed, 3 * d_embed, bias=in_bias)
        self.out_p = nn.Linear(d_embed, d_embed, bias=out_bias)

        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x: torch.Tensor, casal_mask: bool = False) -> torch.Tensor:
        """
        x: (batch size, seqence length, dim)
        """

        in_shape = x.shape
        b, seq_len, d_embd = in_shape
        i_shape = (b, seq_len, self.n_heads, self.d_head)

        # (batch size, seqence length, dim) ->  (batch size, seqence length, 3*dim) -> 3 * (batch size, seqence length, dim)
        q, k, v = self.in_p(x).chunk(3, dim=-1)

        # (batch size, seqence length, dim) -> (batch size, seqence length, n_heads, dim/n_heads) -> (batch size, n_heads, seqence length, dim/n_heads)
        q = q.view(i_shape).transpose(1, 2)
        k = k.view(i_shape).transpose(1, 2)
        v = v.view(i_shape).transpose(1, 2)

        weight = q @ k.transpose(-1, -2)

        if casal_mask:
            mask = torch.ones_like(weight, dtype=torch.bool).triu()
            weight.masked_fill_(mask, -torch.inf)

        # (batch size, n_heads, seqence length, seqence length) -> (batch size, n_heads, seqence length, dim/n_heads)
        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)
        output = weight @ v

        # (batch size, n_heads, seqence length, dim/n_heads) -> (batch size, seqence length, n_heads,  dim/n_heads)
        return self.out_p(output).transpose(1, 2).reshape(in_shape)
