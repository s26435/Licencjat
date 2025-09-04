import torch 
from torch import nn 
from torch.nn import functional as F

from .attention import SelfAttention

class CLIPEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embedding_size: int, n_tokens: int):
        super(CLIPEmbedding, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_size)
        self.positional_embedding = nn.Parameter(torch.zeros(n_tokens, embedding_size))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: (batch size, sequence length)
        returns: (batch size, sequence lenght, dim)
        """

        x = self.token_embedding(tokens)
        x += self.positional_embedding 
        return x


class CLIPLayer(nn.Module):
    def __init__(self, n_head: int, embed_size: int):
        super(CLIPLayer, self).__init__()
        self.layernorm1 = nn.LayerNorm(embed_size)
        self.att = SelfAttention(n_head, embed_size)
        self.layernorm2 = nn.LayerNorm(embed_size)

        self.dense1 = nn.Linear(embed_size, 4 * embed_size)
        self.dense2 = nn.Linear(4 * embed_size, embed_size)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch size, sequence length, dim)
        return: 
        """
        residue = x 
        x = self.layernorm1(x)
        x = self.att(x, causal_mask= True)
        x+=residue
        residue = x
        x = self.layernorm2(x)
        x = self.dense1(x)
        x = self.dense2(x)
        # Quick GILU activation funcion
        x = x * torch.sigmoid(1.702 * x)
        x += residue
        return x





class CLIP(nn.Module):
    def __init__(self):
        super(CLIP, self).__init__()
        self.embedding = CLIPEmbedding(49408, 768, 77)

        self.layers = nn.ModuleList([
            CLIPLayer(12, 768) for i in range(12)
        ])

        self.layernorm = nn.LayerNorm(768)

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        """
        tokens: (batch size, sequence lenght)
        returns: (batch size, sequence length, dim)
        """
        tokens = tokens.type(torch.long)

        # (bs, sq_l) -> (bs, sq_l, dim)
        state = self.embedding(tokens)

        for layer in self.layers:
            state = layer(state)

        # (bs, sq_l, dim)
        return self.layernorm(state)
