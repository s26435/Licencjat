import torch
from torchinfo import summary # noqa: F401
from src import VAE_Encoder, VAE_Decoder
from src import CLIP

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Training on ", device)


x = torch.randn((1, 3, 64, 64, 64)).to(device=device)
noise = torch.randn((1, 4,  8, 8, 8)).to(device=device)
prompt = torch.randint(0, 49408, (1, 77), dtype=torch.long, device=device)


print(x.shape, noise.shape)

decoder = VAE_Decoder().to(device=device)
encoder = VAE_Encoder().to(device=device)
clip = CLIP().to(device=device)
y = encoder(x, noise)
o = decoder(y)

emb = clip(prompt)

print(o.shape, emb.shape)