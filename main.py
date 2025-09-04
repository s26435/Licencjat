import torch
from torch import nn
from torch.nn import functional as F
from torchinfo import summary
from src import VAE_Encoder, VAE_Decoder

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Training on ", device)


x = torch.randn((1, 3, 64, 64, 64)).to(device=device)
noise = torch.randn((1, 4,  8, 8, 8)).to(device=device)

print(x.shape, noise.shape)

decoder = VAE_Decoder().to(device=device)
encoder = VAE_Encoder().to(device=device)

y = encoder(x, noise)
o = decoder(y)

print(o.shape)