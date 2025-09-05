from typing import Optional
import os
import math

import torch
from torch.nn import functional as F

import pytorch_lightning as pl

from .configs import LitConfig
from .decoder import VAE_Decoder
from .encoder import VAE_Encoder
from .clip import CLIP
from .diffusion import Diffusion
from .ddpm import timestep_embedding, DDPMSampler

try:
    from torchvision.utils import save_image

    _HAS_TV = True
except Exception:
    _HAS_TV = False


class DiffusionSystem(pl.LightningModule):
    def __init__(self, cfg: LitConfig):
        super().__init__()
        self.save_hyperparameters(cfg.__dict__)
        self.cfg = cfg
        self.encoder = VAE_Encoder()
        self.decoder = VAE_Decoder()
        self.clip = CLIP() if cfg.use_clip else None
        self.diffusion = Diffusion(context_dim=cfg.context_dim)

        if self.clip and cfg.freeze_clip:
            for p in self.clip.parameters():
                p.requires_grad = False
        if cfg.freeze_decoder:
            for p in self.decoder.parameters():
                p.requires_grad = False

        self.register_buffer("betas", torch.empty(0), persistent=False)
        self.register_buffer("alphas", torch.empty(0), persistent=False)
        self.register_buffer("alphas_bar", torch.empty(0), persistent=False)
        os.makedirs(self.cfg.outdir, exist_ok=True)
        self.val_step_count = 0

    def setup(self, stage: Optional[str] = None) -> None:
        if self.cfg.beta_schedule == "sqrt_linear":
            betas = (
                torch.linspace(1e-4**0.5, 2e-2**0.5, self.cfg.T, dtype=torch.float32)
                ** 2
            )
        elif self.cfg.beta_schedule == "linear":
            betas = torch.linspace(1e-4, 2e-2, self.cfg.T, dtype=torch.float32)
        elif self.cfg.beta_schedule == "cosine":
            s = 0.008
            steps = torch.arange(
                self.cfg.T + 1, dtype=torch.float64, device=self.device
            )
            f = torch.cos(((steps / self.cfg.T + s) / (1 + s)) * math.pi / 2) ** 2
            alphas_bar = f / f[0]
            betas = (
                (1 - (alphas_bar[1:] / alphas_bar[:-1]))
                .clamp(1e-6, 0.9999)
                .to(torch.float32)
            )
        else:
            raise ValueError(self.cfg.beta_schedule)

        betas = betas.to(self.device)
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)
        self._load_vae_weights()

    def _encode_with_noise(self, x: torch.Tensor) -> torch.Tensor:
        B, _, D, H, W = x.shape
        zD, zH, zW = (
            D // self.cfg.downscale,
            H // self.cfg.downscale,
            W // self.cfg.downscale,
        )
        noise_enc = torch.randn(
            (B, self.cfg.latent_channels, zD, zH, zW),
            device=x.device,
            dtype=x.dtype,
        )

        try:
            out = self.encoder(x, noise_enc)
        except TypeError:
            out = self.encoder(x)

        if isinstance(out, tuple):
            if len(out) == 3:
                z, mu, logvar = out
            elif len(out) == 2:
                mu, logvar = out
                z = mu + noise_enc * torch.exp(0.5 * logvar)
            elif len(out) == 1:
                z = out[0]
            else:
                z = out[0]
        else:
            z = out

        return z

    def _check_finite(self, name: str, t: torch.Tensor):
        if not torch.isfinite(t).all():
            with torch.no_grad():
                m = t.float()
                print(
                    f"[NaNGuard] {name}: "
                    f"shape={tuple(m.shape)} min={m.min().item():.3e} max={m.max().item():.3e} "
                    f"mean={m.mean().item():.3e} std={m.std().item():.3e}"
                )
            raise RuntimeError(f"Detected non-finite values in {name}")

    def _forward_loss(self, x: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        B = x.size(0)

        # encode
        z0 = self._encode_with_noise(x) * self.cfg.latent_scale

        # noise
        t = torch.randint(0, self.cfg.T, (B,), device=self.device, dtype=torch.long)
        eps = torch.randn_like(z0)
        a_bar_t = self.alphas_bar[t].view(B, 1, 1, 1, 1)
        zt = torch.sqrt(a_bar_t) * z0 + torch.sqrt(1 - a_bar_t) * eps

        # context
        if self.cfg.use_clip:
            context = self.clip(tokens)  # (B,77,context_dim)
            assert context.size(-1) == self.cfg.context_dim, (
                f"CLIP zwrócił dim={context.size(-1)}, oczekiwano {self.cfg.context_dim}"
            )
        else:
            context = torch.zeros(B, 1, self.cfg.context_dim, device=self.device)

        self._check_finite("context", context)

        t_vec = timestep_embedding(t, self.cfg.time_dim)
        self._check_finite("t_vec", t_vec)

        eps_hat = self.diffusion(zt, context, t_vec)
        self._check_finite("eps_hat", eps_hat)

        loss = F.mse_loss(eps_hat, eps)
        self._check_finite("loss", loss)

        return F.mse_loss(eps_hat, eps)

    # ---- Lightning hooks ----
    def training_step(self, batch, batch_idx):
        x, tokens = batch
        loss = self._forward_loss(x, tokens)
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=x.size(0),
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, tokens = batch
        loss = self._forward_loss(x, tokens)
        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=x.size(0),
        )

        # sampling podglądu TYLKO na pierwszej paczce epoki
        if (
            self.current_epoch + 1
        ) % self.cfg.sample_every_n_epochs == 0 and batch_idx == 0:
            self._do_val_sampling(tokens[: self.cfg.sample_bs])
        return loss

    def _load_vae_weights(self):
        if not self.cfg.vae_weights:
            self.print("[Diff] pomijam ładowanie VAE (brak ścieżki).")
            return
        ckpt = torch.load(self.cfg.vae_weights, map_location=self.device)
        enc_sd = ckpt.get("encoder")
        dec_sd = ckpt.get("decoder")
        if enc_sd is None or dec_sd is None:
            raise RuntimeError(
                f"[Diff] plik {self.cfg.vae_weights} nie zawiera kluczy 'encoder'/'decoder'."
            )

        missing, unexpected = self.encoder.load_state_dict(enc_sd, strict=False)
        if missing:
            self.print(
                f"[Diff][enc] missing: {missing[:6]}{'...' if len(missing) > 6 else ''}"
            )
        if unexpected:
            self.print(
                f"[Diff][enc] unexpected: {unexpected[:6]}{'...' if len(unexpected) > 6 else ''}"
            )

        missing, unexpected = self.decoder.load_state_dict(dec_sd, strict=False)
        if missing:
            self.print(
                f"[Diff][dec] missing: {missing[:6]}{'...' if len(missing) > 6 else ''}"
            )
        if unexpected:
            self.print(
                f"[Diff][dec] unexpected: {unexpected[:6]}{'...' if len(unexpected) > 6 else ''}"
            )

        if "decoder_scale" in ckpt:
            try:
                self.decoder.scale = float(ckpt["decoder_scale"])
                self.print(f"[Diff] ustawiono decoder.scale = {self.decoder.scale}")
            except Exception:
                pass

        if self.cfg.freeze_vae:
            for p in self.encoder.parameters():
                p.requires_grad = False
            for p in self.decoder.parameters():
                p.requires_grad = False
            self.encoder.eval()
            self.decoder.eval()
            self.print("[Diff] VAE załadowany i zamrożony.")

    @torch.no_grad()
    def _do_val_sampling(self, tokens: torch.Tensor):
        self.diffusion.eval()
        self.encoder.eval()

        sampler = DDPMSampler(
            num_training_steps=self.cfg.T,
            schedule=self.cfg.beta_schedule,
            beta_start=float(self.betas[0].item()),
            beta_end=float(self.betas[-1].item()),
        )
        sampler.set_device(self.device)

        B = tokens.size(0)
        if self.cfg.use_clip:
            context = self.clip(tokens)
        else:
            context = torch.zeros(B, 1, self.cfg.context_dim, device=self.device)

        D = self.cfg.sample_size
        latent_shape = (B, 4, D // 8, D // 8, D // 8)

        z0 = sampler.sample(
            model=self.diffusion,
            context=context,
            latent_shape=latent_shape,
            device=self.device,
            time_dim=self.cfg.time_dim,
            generator=torch.Generator(device=self.device).manual_seed(0),
            guidance_scale=self.cfg.guidance_scale,
            uncond_context=None,
            return_all=False,
        )

        x_hat = self.decoder(z0 / max(self.cfg.latent_scale, 1e-8))
        if _HAS_TV:
            mid = D // 2
            slice_img = (x_hat[:, :, mid].clamp(-1, 1) + 1) / 2.0
            save_path = os.path.join(
                self.cfg.outdir, f"sample_ep{self.current_epoch:03d}.png"
            )
            save_image(slice_img, save_path)
            self.print(f"[sample] zapisano podgląd: {save_path}")
        else:
            self.print("[sample] torchvision nieobecny – pomijam zapis PNG.")

    def configure_optimizers(self):
        learnable = list(self.diffusion.parameters()) + list(self.encoder.parameters())
        opt = torch.optim.AdamW(
            learnable, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )

        if self.cfg.warmup_steps and self.cfg.warmup_steps > 0:

            def lr_lambda(step):
                if step < self.cfg.warmup_steps:
                    return float(step + 1) / float(self.cfg.warmup_steps)
                return 1.0

            sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
            return {
                "optimizer": opt,
                "lr_scheduler": {
                    "scheduler": sched,
                    "interval": "step",
                    "name": "warmup",
                },
            }
        return opt
