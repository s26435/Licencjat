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

    def forward(self, x: torch.Tensor, causal_mask: bool = False) -> torch.Tensor:
        """
        x:      (batch size, seqence length, dim)
        return: (batch size, seqence length, n_heads,  dim/n_heads)
        """
        B, L, D = x.shape
        in_shape = x.shape
        b, seq_len, d_embd = in_shape
        assert d_embd == self.n_heads * self.d_head

        i_shape = (b, seq_len, self.n_heads, self.d_head)

        # (batch size, seqence length, dim) ->  (batch size, seqence length, 3*dim) -> 3 * (batch size, seqence length, dim)
        q, k, v = self.in_p(x).chunk(3, dim=-1)

        # (batch size, seqence length, dim) -> (batch size, seqence length, n_heads, dim/n_heads) -> (batch size, n_heads, seqence length, dim/n_heads)
        q = q.view(i_shape).transpose(1, 2)
        k = k.view(i_shape).transpose(1, 2)
        v = v.view(i_shape).transpose(1, 2)

        scores = q @ k.transpose(-1, -2) / math.sqrt(self.d_head)

        if causal_mask:
            causal = torch.ones(L, L, device=x.device, dtype=torch.bool).triu(diagonal=1)
            scores = scores.masked_fill(causal[None, None, :, :], -1e9)


        scores = scores - scores.amax(dim=-1, keepdim=True)
        # (batch size, n_heads, seqence length, seqence length) -> (batch size, n_heads, seqence length, dim/n_heads)
        weight = F.softmax(scores, dim=-1)
        output = weight @ v

        # (batch size, n_heads, seqence length, dim/n_heads) -> (batch size, seqence length, n_heads,  dim/n_heads)
        return self.out_p(output.transpose(1, 2).reshape(in_shape))


class CrossAttention(nn.Module):
    def __init__(
        self,
        n_heads: int,
        embedding_dim: int,
        cross_dim: int,
        in_bias: bool = True,
        out_bias: bool = True,
    ):
        super(CrossAttention, self).__init__()
        self.q_proj = nn.Linear(embedding_dim, embedding_dim, bias=in_bias)
        self.k_proj = nn.Linear(cross_dim, embedding_dim, bias=in_bias)
        self.v_proj = nn.Linear(cross_dim, embedding_dim, bias=in_bias)

        self.out_proj = nn.Linear(embedding_dim, embedding_dim, bias=out_bias)
        self.n_heads = n_heads
        self.d_head = embedding_dim // n_heads

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        x  (latent): (batch size, sequence length  Q, dim q)
        y (context): (batch size, sequence lenght KV, dim kv)
        return:
        """

        input_shape = x.shape
        B, Q_len, _ = x.shape
        B2, KV_len, _ = y.shape
        assert B == B2

        Q = self.q_proj(x).view(B, Q_len, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(y).view(B, KV_len, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(y).view(B, KV_len, self.n_heads, self.d_head).transpose(1, 2)

        att = F.softmax(Q @ K.transpose(-1, -2) / math.sqrt(self.d_head), dim=-1) @ V

        return self.out_proj(att.transpose(1, 2).reshape(input_shape))
