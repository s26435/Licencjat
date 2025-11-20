import pytorch_lightning as pl
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


from src.modules.vae import VAE
from src.modules.context import ContextEncoder
from src.modules.configs import VAEConfig
from src.modules.diffusion import Diffusion
from src.modules.ddpm import DDPMSampler, timestep_embedding
from typing import Any, Optional, Union

def SD_loss(eps_pred: torch.Tensor, eps_true: torch.Tensor, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(eps_pred, eps_true) + F.mse_loss(y_pred, y_true)

class StableDiffusion(pl.LightningModule):
    def __init__(
        self,
        vae_path: str | Path,
        time_dim: int = 320,
        context_dim: int = 128,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        num_train_steps: int = 1000,
        p_uncond: float = 0.1,
        guidance_scale: float = 1.0,
        ctx_encoder_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(
            {
                "time_dim": time_dim,
                "context_dim": context_dim,
                "lr": lr,
                "weight_decay": weight_decay,
                "num_train_steps": num_train_steps,
                "p_uncond": p_uncond,
                "guidance_scale": guidance_scale,
                "ctx_encoder_kwargs": ctx_encoder_kwargs or {},
            }
        )

        vae_cfg = VAEConfig()
        self.vae = VAE(vae_cfg)
        self.vae.load_vae_from_file(Path(vae_path))
        for p in self.vae.parameters():
            p.requires_grad = False
        self.vae.eval()
        self.ctx_enc = ContextEncoder(
            in_dim=3,
            out_dim=context_dim,
            hidden=256,
            dropout=0.0,
        )
        latent_ch = getattr(vae_cfg, "latent_channels", getattr(vae_cfg, "latent", 4))
        self.model = Diffusion(
            context_dim=context_dim,
            time_embedding_size=time_dim,
            latent_space_size=latent_ch,
        )
        self.null_context = nn.Parameter(torch.zeros(1, 1, context_dim))

        self.sampler = DDPMSampler(num_training_steps=num_train_steps)

    @staticmethod
    def _pick_x_from_batch(batch):
        if isinstance(batch, dict):
            for k in ("x", "image", "images"):
                if k in batch:
                    return batch[k]
            return next(iter(batch.values()))
        if isinstance(batch, (list, tuple)):
            return batch[0]
        return batch

    @staticmethod
    def _pick_context_from_batch(batch):
        if isinstance(batch, dict):
            for k in ("context", "cond", "ctx"):
                if k in batch:
                    return batch[k]
        if isinstance(batch, (list, tuple)) and len(batch) > 1:
            return batch[1]
        return None

    @torch.no_grad()
    def _encode_to_latents(self, x: torch.Tensor) -> torch.Tensor:
        out = self.vae.encoder(x)
        if isinstance(out, (tuple, list)) and len(out) >= 1:
            z = out[0]
        else:
            z = out
        return z

    def _encode_context(
        self,
        ctx_raw: torch.Tensor | None,
        batch_size: int | None = 1,
    ) -> torch.Tensor:
        if ctx_raw is None:
            if batch_size is None:
                raise ValueError(
                    "Dla ctx_raw=None musisz podać batch_size do _encode_context."
                )
            return self.null_context.expand(batch_size, 1, self.hparams.context_dim)

        ctx_raw = ctx_raw.to(self.device)
        ctx_emb = self.ctx_enc(ctx_raw)
        return ctx_emb

    def _maybe_drop_context(
        self, ctx_emb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b = ctx_emb.size(0)
        if self.hparams.p_uncond <= 0:
            return ctx_emb, torch.zeros(b, dtype=torch.bool, device=ctx_emb.device)
        drop = torch.rand(b, device=ctx_emb.device) < self.hparams.p_uncond
        if drop.any():
            null = self.null_context.expand(b, 1, self.hparams.context_dim)
            ctx_emb = torch.where(drop.view(b, 1, 1), null, ctx_emb)
        return ctx_emb, drop

    def training_step(self, batch, batch_idx: int):
        x = self._pick_x_from_batch(batch).to(self.device)
        ctx_raw = self._pick_context_from_batch(batch)

        if "cell_params" not in batch:
            raise RuntimeError("Batch nie zawiera 'cell_params', nie mam targetu dla y.")
        cell = batch["cell_params"].to(self.device)

        with torch.no_grad():
            z = self._encode_to_latents(x)

        b = z.size(0)
        t = torch.randint(0, self.sampler.T, (b,), device=self.device, dtype=torch.long)
        eps = torch.randn_like(z)

        alpha_bar = self.sampler.alphas_bar.to(self.device)[t].view(
            b, *([1] * (z.dim() - 1))
        )
        z_noisy = torch.sqrt(alpha_bar) * z + torch.sqrt(1.0 - alpha_bar) * eps

        ctx_emb = self._encode_context(ctx_raw)
        ctx_emb, drop_mask = self._maybe_drop_context(ctx_emb)

        t_emb = timestep_embedding(t, self.hparams.time_dim).to(self.device)
        eps_pred, y = self.model(z_noisy, ctx_emb, t_emb)

        loss = SD_loss(eps_pred, eps, y, cell)

        self.log_dict(
            {
                "train/loss": loss,
                "train/uncond_ratio": drop_mask.float().mean(),
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=b,
        )
        return loss

    def validation_step(self, batch, batch_idx: int):
        x = self._pick_x_from_batch(batch).to(self.device)
        ctx_raw = self._pick_context_from_batch(batch)

        if "cell_params" not in batch:
            raise RuntimeError("Batch nie zawiera 'cell_params', nie mam targetu dla y (val).")
        
        cell = batch["cell_params"].to(self.device)
        with torch.no_grad():
            z = self._encode_to_latents(x)

        b = z.size(0)
        t = torch.randint(0, self.sampler.T, (b,), device=self.device, dtype=torch.long)
        eps = torch.randn_like(z)

        alpha_bar = self.sampler.alphas_bar.to(self.device)[t].view(
            b, *([1] * (z.dim() - 1))
        )
        z_noisy = torch.sqrt(alpha_bar) * z + torch.sqrt(1.0 - alpha_bar) * eps

        ctx_emb = self._encode_context(ctx_raw)

        t_emb = timestep_embedding(t, self.hparams.time_dim).to(self.device)
        eps_pred, y = self.model(z_noisy, ctx_emb, t_emb)

        loss = SD_loss(eps_pred, eps, y, cell)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, batch_size=b)
        return loss

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        latent_shape_hwz: Tuple[int, ...],
        ctx_raw: torch.Tensor | None = None,
        ctx_raw_uncond: torch.Tensor | None = None,
        guidance_scale: float | None = None,
    ) -> torch.Tensor:
        device = self.device
        Cz = getattr(self.vae.encoder, "latent_size", None)
        if Cz is None:
            Cz = getattr(self.hparams, "latent_channels", None)
        if Cz is None:
            Cz = getattr(
                getattr(self, "vae", object), "cfg", VAEConfig()
            ).latent_channels

        z_shape = (batch_size, Cz, *latent_shape_hwz)

        cond_ctx = self._encode_context(ctx_raw) if ctx_raw is not None else None

        if ctx_raw_uncond is not None:
            uncond_ctx = self._encode_context(ctx_raw_uncond)
        else:
            uncond_ctx = self.null_context.expand(
                batch_size, 1, self.hparams.context_dim
            )

        g = self.hparams.guidance_scale if guidance_scale is None else guidance_scale

        latents = self.sampler.sample(
            model=self.model,
            context=cond_ctx if cond_ctx is not None else uncond_ctx,
            latent_shape=z_shape,
            device=device,
            time_dim=self.hparams.time_dim,
            guidance_scale=g,
            uncond_context=uncond_ctx if cond_ctx is not None else None,
        )

        scale = getattr(self.vae.decoder, "scale", 1.0)
        x = self.vae.decoder(latents * scale)
        return x

    def configure_optimizers(self):
        return torch.optim.AdamW(
            list(self.model.parameters()) + list(self.ctx_enc.parameters()),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

    def on_fit_start(self):
        if hasattr(self.sampler, "set_device"):
            self.sampler.set_device(self.device)

    def save(
        self,
        path: Union[str, Path],
        *,
        save_vae: bool = False,
        vae_path_hint: Optional[Union[str, Path]] = None,
        extra: Optional[dict] = None,
    ) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "format": "stable_diffusion_4d_pt1",
            "hparams": dict(self.hparams),
            "state_dicts": {
                "ctx_enc": self.ctx_enc.state_dict(),
                "diffusion": self.model.state_dict(),
                "null_context": self.null_context.detach().cpu(),
            },
            "vae": None,
            "vae_path_hint": str(vae_path_hint) if vae_path_hint is not None else None,
            "extra": extra or {},
        }

        if save_vae:
            payload["vae"] = {
                "encoder": self.vae.encoder.state_dict(),
                "decoder": self.vae.decoder.state_dict(),
                "decoder_scale": getattr(self.vae.decoder, "scale", 1.0),
            }

        torch.save(payload, path)
        return path

    @classmethod
    def load(
        cls,
        ckpt_path: Union[str, Path],
        *,
        map_location: Union[str, torch.device] = "cpu",
        vae_path: Optional[Union[str, Path]] = None,
        strict: bool = True,
        override_hparams: Optional[dict] = None,
    ) -> "StableDiffusion":
        ckpt_path = Path(ckpt_path)
        blob = torch.load(ckpt_path, map_location=map_location)

        if blob.get("format") != "stable_diffusion_4d_pt1":
            raise ValueError("Nieznany format checkpointu (expected stable_diffusion_4d_pt1)")

        hp = blob["hparams"] or {}
        if override_hparams:
            hp.update(override_hparams)

        has_vae_weights = blob.get("vae") is not None
        vae_path_hint = blob.get("vae_path_hint", None)
        chosen_vae_path = vae_path or vae_path_hint

        if not has_vae_weights and chosen_vae_path is None:
            raise ValueError(
                "Checkpoint nie zawiera wag VAE i nie podano `vae_path`. "
                "Podaj ścieżkę do VAE lub zapisz checkpoint z `save_vae=True`."
            )

        sd = cls(
            vae_path=chosen_vae_path if chosen_vae_path is not None else "",
            time_dim=hp.get("time_dim", 320),
            context_dim=hp.get("context_dim", 128),
            lr=hp.get("lr", 1e-4),
            weight_decay=hp.get("weight_decay", 1e-4),
            num_train_steps=hp.get("num_train_steps", 1000),
            p_uncond=hp.get("p_uncond", 0.1),
            guidance_scale=hp.get("guidance_scale", 1.0),
            ctx_encoder_kwargs=hp.get("ctx_encoder_kwargs", {}),
        )
        sd.to(map_location)

        sd.model.load_state_dict(blob["state_dicts"]["diffusion"], strict=strict)
        sd.ctx_enc.load_state_dict(blob["state_dicts"]["ctx_enc"], strict=strict)
        null_ctx = blob["state_dicts"].get("null_context", None)
        if null_ctx is not None:
            with torch.no_grad():
                sd.null_context.copy_(null_ctx.to(map_location))

        if has_vae_weights:
            vae_pack = blob["vae"]
            sd.vae.encoder.load_state_dict(vae_pack["encoder"], strict=True)
            sd.vae.decoder.load_state_dict(vae_pack["decoder"], strict=True)
            if "decoder_scale" in vae_pack and hasattr(sd.vae.decoder, "scale"):
                sd.vae.decoder.scale = vae_pack["decoder_scale"]
            for p in sd.vae.parameters():
                p.requires_grad = False
            sd.vae.eval()
        else:
            pass

        return sd

    @torch.no_grad()
    def generate_unimat_and_cell(
        self,
        batch_size: int,
        latent_shape_hwz: Tuple[int, ...],
        ctx_raw: torch.Tensor | None = None,
        ctx_raw_uncond: torch.Tensor | None = None,
        guidance_scale: float | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        x = self.sample(
            batch_size=batch_size,
            latent_shape_hwz=latent_shape_hwz,
            ctx_raw=ctx_raw,
            ctx_raw_uncond=ctx_raw_uncond,
            guidance_scale=guidance_scale,
        )

        z = self._encode_to_latents(x)
        b = z.size(0)
        device = self.device

        t = torch.zeros(b, dtype=torch.long, device=device)
        t_emb = timestep_embedding(t, self.hparams.time_dim).to(device)

        ctx_emb = self._encode_context(ctx_raw)

        eps_pred, y_pred = self.model(z, ctx_emb, t_emb)

        grid = x.permute(0, 2, 3, 4, 1).contiguous()

        return grid, y_pred