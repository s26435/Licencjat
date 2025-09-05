import torch
from torchinfo import summary # noqa: F401
from src import VAE_Encoder, VAE_Decoder
from src import CLIP, Diffusion, report_model, human_bytes, model_size_bytes

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Training on ", device)


x = torch.randn((1, 3, 64, 64, 64)).to(device=device)
noise = torch.randn((1, 4,  8, 8, 8)).to(device=device)
prompt = torch.randint(0, 49408, (1, 77), dtype=torch.long, device=device)
time = torch.rand((1, 320), device=device)

print(x.shape, noise.shape)

decoder = VAE_Decoder().to(device=device)
encoder = VAE_Encoder().to(device=device)
clip = CLIP().to(device=device)
diff = Diffusion(768).to(device=device)

emb = clip(prompt)
o = diff(noise, emb, time)

print(o.shape, emb.shape)

report_model("VAE_Encoder", encoder)
report_model("VAE_Decoder", decoder)
report_model("CLIP",        clip)
report_model("Diffusion",   diff)
total_bytes = sum(model_size_bytes(m)[2] for m in [encoder, decoder, clip, diff])
print(f"====================\nSUMA rozmiarów modeli: {human_bytes(total_bytes)}")