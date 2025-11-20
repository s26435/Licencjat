import os
import torch
from pathlib import Path

from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from typing import Optional
import numpy as np

import pytorch_lightning as pl

from .decoder import VAE_Decoder
from .encoder import VAE_Encoder
from .configs import VAEConfig

class GridOnlyWrapper(Dataset):
    def __init__(
        self,
        base_dataset: Dataset,
        replace_nan: bool = True,
        replace_minus_one: bool = False,
        fill_value: float = 0.0,
        dtype: torch.dtype = torch.float32,
    ):
        self.ds = base_dataset
        self.replace_nan = replace_nan
        self.replace_m1 = replace_minus_one
        self.fill_value = float(fill_value)
        self.dtype = dtype

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> torch.Tensor:
        item = self.ds[idx]
        g = item["grid"]  # torch.Tensor [P,G,K,3] lub numpy

        if isinstance(g, np.ndarray):
            g = torch.from_numpy(g)

        g = g.to(dtype=self.dtype)  # [P,G,K,3]

        if self.replace_nan and torch.is_floating_point(g):
            g = torch.nan_to_num(g, nan=self.fill_value)

        if self.replace_m1:
            g = torch.where(torch.isclose(g, torch.tensor(-1.0, dtype=g.dtype)), 
                            torch.tensor(self.fill_value, dtype=g.dtype), g)

        # [P,G,KT,3] -> [3,P,G,KT]
        x = g.permute(3, 0, 1, 2).contiguous()
        return x


class GridOnlyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        base_dataset: Dataset,
        batch_size: int = 2,
        num_workers: int = 0,
        val_split: float = 0.1,
        shuffle: bool = True,
        replace_nan: bool = True,
        replace_minus_one: bool = False,
        fill_value: float = 0.0,
        dtype: torch.dtype = torch.float32,
        max_samples: Optional[int] = None,
    ):
        super().__init__()
        self.base = base_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.shuffle = shuffle

        self.wrap_kwargs = dict(
            replace_nan=replace_nan,
            replace_minus_one=replace_minus_one,
            fill_value=fill_value,
            dtype=dtype,
        )
        self.max_samples = max_samples

        self.train_set: Optional[Dataset] = None
        self.val_set: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        ds = self.base
        if self.max_samples is not None and self.max_samples < len(ds):
            idx = torch.arange(len(ds))[: self.max_samples].tolist()
            ds = torch.utils.data.Subset(ds, idx)

        n = len(ds)
        n_val = max(1, int(round(n * self.val_split)))
        n_train = n - n_val

        tr_base, va_base = random_split(
            ds, [n_train, n_val], generator=torch.Generator().manual_seed(0)
        )
        self.train_set = GridOnlyWrapper(tr_base, **self.wrap_kwargs)
        self.val_set = GridOnlyWrapper(va_base, **self.wrap_kwargs)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

class VAE(pl.LightningModule):
    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.save_hyperparameters(cfg.__dict__)
        self.cfg = cfg

        self.encoder = VAE_Encoder()
        self.decoder = VAE_Decoder()

        os.makedirs(self.cfg.outdir, exist_ok=True)

    def _encode(self, x: torch.Tensor):
        B, _, D, H, W = x.shape
        zD, zH, zW = (
            D // self.cfg.downscale,
            H // self.cfg.downscale,
            W // self.cfg.downscale,
        )
        noise = torch.randn(
            B, self.cfg.latent_channels, zD, zH, zW, device=x.device, dtype=x.dtype
        )
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
            kl_map = self._kl_terms(mu, logvar)  # (B,C,D,H,W)
            kl = kl_map.mean()  # skalar
            beta = self._current_beta()
        else:
            kl = torch.tensor(0.0, device=x.device, dtype=x.dtype)
            beta = 0.0

        loss = rec + beta * kl

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
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=x.size(0),
        )
        self.log("train/rec", rec, on_step=True, on_epoch=True, batch_size=x.size(0))
        self.log("train/kl", kl, on_step=True, on_epoch=True, batch_size=x.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        x = self._get_x(batch)
        loss, rec, kl = self._loss(x)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, batch_size=x.size(0))
        self.log("val/rec", rec, on_epoch=True, batch_size=x.size(0))
        self.log("val/kl", kl, on_epoch=True, batch_size=x.size(0))
        if (
            self.current_epoch + 1
        ) % self.cfg.sample_every_n_epochs == 0 and batch_idx == 0:
            self._save_preview(x)
        return loss

    def configure_optimizers(self):
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        opt = torch.optim.AdamW(
            params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )
        return opt

    def on_after_backward(self):
        if self.cfg.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.decoder.parameters()),
                self.cfg.grad_clip,
            )

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

    def load_vae_from_file(self, vae_path: Path):
        dic = torch.load(vae_path, map_location=self.device)
        self.encoder.load_state_dict(
            state_dict=dic["encoder"],
            strict=True,
        )
        self.decoder.load_state_dict(
            state_dict=dic["decoder"],
            strict=True,
        )