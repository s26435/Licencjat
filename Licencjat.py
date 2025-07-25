#!/usr/bin/env python
# coding: utf-8

import torch
print(torch.__version__)
import pytorch_lightning as pl
print(pl.__version__)

from torch import nn as nn
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

def moving_average(x, w=20):
    return np.convolve(x, np.ones(w), 'valid') / w


class Encoder(nn.Module):
    def __init__(self, dim=4):
        super(Encoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(512, 1024, 3, 2, 1), 
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(1024, dim)
        self.fc_log = nn.Linear(1024, dim)

    def forward(self, x):
        h = self.conv(x).view(x.size(0), -1)
        # print(h.shape)
        mu = self.fc_mu(h)
        log = self.fc_log(h)

        std = torch.exp(0.5 * log)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z, mu, log

class Decoder(nn.Module):
    def __init__(self, dim=4):
        super(Decoder, self).__init__()

        self.fc = nn.Linear(dim, 1024*4*4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 3, 4, 2, 1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        h = self.fc(x).view(x.size(0), 1024, 4, 4)
        return self.deconv(h)



class CrossAttention(nn.Module):
    def __init__(self, dim, context_dim):
        super(CrossAttention, self).__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(context_dim, dim)
        self.v = nn.Linear(context_dim, dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x, context):
        q = self.q(x)

        k = self.k(context)
        v = self.v(context)

        att = torch.softmax(q @ k.transpose(-1, -2)/(q.size(-1)**0.5), dim=-1)
        return self.out(att @ v)


a = CrossAttention(10, 10)
fwd, context = torch.randn(1, 10), torch.randn(1,10)
print(a(fwd, context))

class UNetBlock(nn.Module):
    def __init__(self, dim, context_dim):
        super(UNetBlock, self).__init__()

        self.conv = nn.Conv2d(dim, dim, 3, padding=1)
        self.norm = nn.GroupNorm(8, dim)
        self.att = CrossAttention(dim, context_dim)

    def forward(self, x, context):
        B, C, H, W = x.shape

        h = F.silu(self.norm(self.conv(x)))
        h_flat = h.view(B, C, -1).permute(0,2,1)
        h_att = self.att(h_flat, context).permute(0,2,1).view(B, C, H, W)
        return x + h_att


unetblock = UNetBlock(32, 10)
fwd, context = torch.randn(1, 32, 32, 32), torch.randn(1,10)

print(unetblock(fwd, context))


class UNet(nn.Module):
    def __init__(self, in_channels=4, base_channels=64, context_dim =786, num_blocks=4):
        super(UNet, self).__init__()
        self.init = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.blocks = nn.ModuleList([UNetBlock(base_channels, context_dim) for _ in range(num_blocks)])
        self.final = nn.Conv2d(base_channels, in_channels, 3, padding=1)

        self.time_mlp = nn.Sequential(
            nn.Linear(1, base_channels),
            nn.SiLU(), 
            nn.Linear(base_channels, base_channels))

    def forward(self, x, t, context):

        x = self.init(x)
        t = t.view(-1, 1).float().to(x.device)
        t_emb = self.time_mlp(t).view(x.shape[0], -1, 1, 1)
        for block in self.blocks:
            x = block(x + t_emb, context)

        return self.final(x)

def apply_cfg(unet, z_t, t, context_cond, context_uncond, cfg_scale=7.5):
    device = next(unet.parameters()).device

    z_t = z_t.to(device)
    t = t.to(device)
    context_cond = context_cond.to(device)
    context_uncond = context_uncond.to(device)

    pred_uncond = unet(z_t, t, context_uncond)
    pred_cond = unet(z_t, t, context_cond)
    return pred_uncond + cfg_scale * (pred_cond - pred_uncond)



def linear_beta_shedule(timesteps):
    return torch.linspace(1e-4, 0.02, timesteps)

class DiffusionShedulder:
    def __init__(self, timesteps= 1000):
        self.timesteps = timesteps
        self.betas = linear_beta_shedule(timesteps)

        self.alphas = 1.0 - self.betas

        self.alpha_hat = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x_start, t, noise):
        device = x_start.device
        # print(device)

        B, C, H, W = x_start.shape
        alpha_hat_t = self.alpha_hat[t].to(device).view(B, 1, 1, 1)
        one_minus = (1 - self.alpha_hat[t]).to(device).view(B, 1, 1, 1)

        return alpha_hat_t * x_start + one_minus * noise


class DummyTextEncoder(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.token_to_id = {word: i for i, word in enumerate(vocab)}
        self.embed = nn.Embedding(len(vocab), 768)

    def forward(self, prompts):
        device = self.embed.weight.device
        ids = torch.tensor([self.token_to_id[p] for p in prompts], device=device)
        return self.embed(ids).unsqueeze(1) 

class StableDiffusion(pl.LightningModule):
    def __init__(self, context_dropout=0.1, vae_steps=100, vae_dim=1024, vae_epochs = 2):
        super().__init__()
        self.encoder = Encoder(dim=vae_dim)
        self.decoder = Decoder(dim=vae_dim)
        self.unet = UNet(in_channels=vae_dim, context_dim=768, num_blocks=10)
        self.text_encoder = DummyTextEncoder(["", "zero", "one", "two", "three", "four", 
                                              "five", "six", "seven", "eight", "nine"])
        self.scheduler = DiffusionShedulder()
        self.context_dropout = context_dropout
        self.loss_fn = nn.MSELoss()
        self.loss_history = []
        self.vae_epochs = vae_epochs
        self.vae_dim = vae_dim
        self._pretrain_vae(steps=vae_steps)

    def _pretrain_vae(self, steps=100):
        print("Pretraining VAE...")
        vae_optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=1e-3, weight_decay=1e-4
        )

        dataset = MNISTWithPrompts(train=True)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        self.encoder.to(self.device)
        self.decoder.to(self.device)

        epochs = self.vae_epochs
        history = []
        for epoch in range(epochs):
            bar = tqdm(enumerate(loader), total=100)
            for i, (x, _) in bar:
                x = x.to(self.device)
                if i >= steps:
                    break

                z, mu, logvar = self.encoder(x)
                recon = self.decoder(z)

                recon_loss = nn.MSELoss()(recon, x)
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + 1e-4 * kl_loss
                history.append(loss)
                bar.set_postfix(loss=loss)
                vae_optimizer.zero_grad()
                loss.backward()
                vae_optimizer.step()



            with torch.no_grad():
                x_sample = x[0].unsqueeze(0)
                recon_sample = self.decoder(self.encoder(x_sample)[0])

                fig, axs = plt.subplots(1, 2, figsize=(6, 3))
                axs[0].imshow(np.transpose(x_sample.squeeze().cpu().numpy(), (1, 2, 0)))
                axs[0].set_title("Oryginał")
                axs[0].axis('off')

                axs[1].imshow(np.transpose(recon_sample.squeeze().cpu().numpy(), (1, 2, 0)))
                axs[1].set_title("Rekonstrukcja")
                axs[1].axis('off')

                plt.suptitle(f"Epoka {epoch+1}")
                plt.show()

        history = [h.cpu().detach() for h in history]

        plt.plot(moving_average(history, 10))
        plt.title("Smoothed Loss Curve")
        plt.xlabel("Iteracja")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.show()

    def training_step(self, batch, batch_idx):
        x, prompt = batch
        batch_size = x.size(0)

        z0, mu, logvar = self.encoder(x)

        t = torch.randint(0, self.scheduler.timesteps, (batch_size,))
        z0 = z0.view(batch_size, self.vae_dim, 1, 1)
        noise = torch.randn_like(z0)
        z_t = self.scheduler.q_sample(z0, t, noise)
        prompt_embed = self.text_encoder(prompt).to(self.device)
        null_embed = torch.zeros_like(prompt_embed)

        mask = torch.rand(batch_size, device=prompt_embed.device) < self.context_dropout
        context = torch.where(mask.view(-1, 1, 1), null_embed, prompt_embed)

        pred_noise = self.unet(z_t, t, context)
        loss = self.loss_fn(pred_noise, noise)
        self.loss_history.append(loss)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-6, weight_decay=1e-4)

LABEL_TO_WORD = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
    5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"
}
class MNISTWithPrompts(Dataset):
    def __init__(self, train=True):
        self.dataset = datasets.MNIST(
            root="./data", train=train, download=True,
            transform=transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor()
            ])
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        prompt = LABEL_TO_WORD[label]
        return image.repeat(3, 1, 1), prompt


model = StableDiffusion(context_dropout=0.1, vae_steps=100)

from pytorch_lightning.callbacks import EarlyStopping
early_stop_callback = EarlyStopping(
    monitor="train_loss", 
    patience=3,
    verbose=True,
    mode="min" 
)

train_loader = DataLoader(
    MNISTWithPrompts(train=True),
    batch_size=64,
    shuffle=True,
    num_workers=0 
)

trainer = pl.Trainer(
    max_epochs=1000,
    accelerator="gpu",
    devices=1,
    callbacks=[early_stop_callback]
)

trainer.fit(model, train_loader)


@torch.no_grad()
def generate(model, prompt: str, steps=1000, cfg_scale=7.5, z_dim=1024, device='cuda'):
    model = model.to(device)
    z = torch.randn((1, z_dim), device=device)
    text_embed = model.text_encoder([prompt]).to(device)   
    null_embed = model.text_encoder([""]).to(device) 
    z_t = apply_cfg(unet, z_t, t=torch.full((B,), t, device=device), 
                    context_cond=context_cond,
                    context_uncond=context_uncond,
                    cfg_scale=5.0)

    img = model.decoder(z).clamp(0, 1)
    return img



p = "nine"
img = generate(model, prompt=p, cfg_scale=12, steps=1000, device="cuda")
plt.imshow(TF.to_pil_image(img.squeeze(0).cpu()))
plt.title(f"Generated: {p}")
plt.axis("off")
plt.show()


# In[ ]:


history = [h.cpu().detach() for h in model.loss_history]

plt.plot(moving_average(history, 20))
plt.title("Smoothed Loss Curve")
plt.xlabel("Iteracja")
plt.ylabel("Loss")
plt.grid(True)
plt.show()


# In[ ]:




