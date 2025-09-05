import os
import torch
from torch.nn import functional as F
import pytorch_lightning as pl
from .decoder import VAE_Decoder
from .encoder import VAE_Encoder
from .configs import VAEConfig

try:
    from torchvision.utils import save_image
    _HAS_TV = True
except Exception:
    _HAS_TV = False


class VAESystem(pl.LightningModule):
    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.save_hyperparameters(cfg.__dict__)
        self.cfg = cfg

        self.encoder = VAE_Encoder()
        self.decoder = VAE_Decoder()

        os.makedirs(self.cfg.outdir, exist_ok=True)

    def _encode(self, x: torch.Tensor):
        B, _, D, H, W = x.shape
        zD, zH, zW = D // self.cfg.downscale, H // self.cfg.downscale, W // self.cfg.downscale
        noise = torch.randn(B, self.cfg.latent_channels, zD, zH, zW, device=x.device, dtype=x.dtype)
        out = None
        try:
            out = self.encoder(x, noise)
        except TypeError:
            out = self.encoder(x)

        z = mu = logvar = None
        if isinstance(out, tuple):
            if len(out) == 3:
                z, mu, logvar = out
            elif len(out) == 2:
                mu, logvar = out
                z = mu + noise * torch.exp(0.5 * logvar)
            elif len(out) == 1:
                z = out[0]
            else:
                z = out[0]
        else:
            z = out

        return z, mu, logvar

    def _current_beta(self) -> float:
        if self.cfg.beta_kl_warmup_steps <= 0:
            return self.cfg.beta_kl_max
        s = min(1.0, float(self.global_step) / float(self.cfg.beta_kl_warmup_steps))
        return float(self.cfg.beta_kl_max * s)

    def _kl_terms(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # KL per-element: (B, C, D, H, W)
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        if self.cfg.use_free_bits and self.cfg.kl_free_nats > 0:
            kl = torch.clamp(kl, min=self.cfg.kl_free_nats)
        return kl

    def _loss(self, x: torch.Tensor):
        z, mu, logvar = self._encode(x)
        x_hat = self.decoder(z * self.decoder.scale)

        rec = F.l1_loss(x_hat, x)  # albo MSE
        if mu is not None and logvar is not None:
            kl_map = self._kl_terms(mu, logvar)       # (B,C,D,H,W)
            kl = kl_map.mean()                         # skalar
            beta = self._current_beta()
        else:
            kl = torch.tensor(0.0, device=x.device, dtype=x.dtype); beta = 0.0

        loss = rec + beta * kl

        # logging pomocniczy – zobaczysz, czy KL „żyje”
        with torch.no_grad():
            self.log("debug/beta", beta, on_step=True, prog_bar=False)
            if mu is not None:
                self.log("debug/mu_std", mu.float().std(), on_step=True)
                self.log("debug/logvar_mean", logvar.float().mean(), on_step=True)
        return loss, rec.detach(), kl.detach()

    def _get_x(self, batch):
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch
        if not torch.is_tensor(x):
            raise TypeError(f"Oczekiwałem tensora, dostałem {type(x)}")
        return x.to(self.device)

    def training_step(self, batch, batch_idx):
        x = self._get_x(batch)
        loss, rec, kl = self._loss(x)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=x.size(0))
        self.log("train/rec",  rec,  on_step=True, on_epoch=True, batch_size=x.size(0))
        self.log("train/kl",   kl,   on_step=True, on_epoch=True, batch_size=x.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        x = self._get_x(batch)
        loss, rec, kl = self._loss(x)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, batch_size=x.size(0))
        self.log("val/rec",  rec,  on_epoch=True, batch_size=x.size(0))
        self.log("val/kl",   kl,   on_epoch=True, batch_size=x.size(0))
        if (self.current_epoch + 1) % self.cfg.sample_every_n_epochs == 0 and batch_idx == 0:
            self._save_preview(x)
        return loss

    @torch.no_grad()
    def _save_preview(self, x: torch.Tensor):
        """
        Zapisuje porównanie [x, x_hat] jako PNG (środkowy slice po osi Z).
        """
        self.encoder.eval()
        self.decoder.eval()
        z, _, _ = self._encode(x)
        x_hat = self.decoder(z * self.decoder.scale)

        if not _HAS_TV:
            return

        D = x.shape[2]
        mid = D // 2
        def to_img(t):
            return (t[:, :, mid].clamp(-1, 1) + 1) / 2.0

        grid = torch.cat([to_img(x), to_img(x_hat)], dim=0)
        save_path = os.path.join(self.cfg.outdir, f"vae_ep{self.current_epoch:03d}.png")
        save_image(grid, save_path, nrow=x.size(0), padding=0, normalize=True, value_range=(0, 100))
        self.print(f"[VAE] zapisano podgląd: {save_path}")

    def configure_optimizers(self):
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        opt = torch.optim.AdamW(params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        return opt

    def on_after_backward(self):
        if self.cfg.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(list(self.encoder.parameters()) + list(self.decoder.parameters()),self.cfg.grad_clip)
    
    def export_vae(self, path: str):
        export = {
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "decoder_scale": float(getattr(self.decoder, "scale", 1.0)),
            "arch": {
                "latent_channels": getattr(self.cfg, "latent_channels", 4),
                "downscale": getattr(self.cfg, "downscale", 8),
            },
        }
        torch.save(export, path)
        self.print(f"[VAE] zapisano encoder+decoder do: {path}")

    def on_fit_end(self):
        out_path = os.path.join(self.cfg.outdir, "vae_export.pt")
        self.export_vae(out_path)
