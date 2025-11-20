from src.modules.vae import VAE, GridOnlyDataModule
from src.modules.configs import VAEConfig
from src.unimat.dataset import UniMatTorchDataset, GridCtxDataModule
from src.stable_diffusion import StableDiffusion

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
from pathlib import Path
from typing import Optional
import torch

import gc

def train_vae(ds_path: Path, epochs: int) -> VAE:
    cfg = VAEConfig()
    dataset = GridOnlyDataModule(UniMatTorchDataset(ds_path), num_workers=4, batch_size=8)
    model = VAE(cfg)
    lrmon = LearningRateMonitor(logging_interval="step")
    logger = CSVLogger(cfg.outdir, name="logs")
    trainer = Trainer(
        max_epochs=epochs,
        precision="bf16-mixed",
        accelerator="auto",
        devices="auto",
        callbacks=[lrmon],
        logger=logger,
        deterministic=False,
        num_sanity_val_steps=0,
        limit_val_batches=0, 
    )

    trainer.fit(model, datamodule=dataset)
    return model


def train_stable_diffusion(
    ds_path: Path,
    vae_path: Path,
    epochs: int,
    time_dim: int = 320,
    context_dim: int = 128,
    num_train_steps: int = 1000,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    p_uncond: float = 0.1,
    guidance_scale: float = 1.0,
) -> StableDiffusion:

    datamodule = GridCtxDataModule(ds_path, batch_size=2, num_workers=4)

    sd = StableDiffusion(
        vae_path=vae_path,
        time_dim=time_dim,
        context_dim=context_dim,
        lr=lr,
        weight_decay=weight_decay,
        num_train_steps=num_train_steps,
        p_uncond=p_uncond,
        guidance_scale=guidance_scale,
    )
    sd.vae.decoder = None

    lrmon = LearningRateMonitor(logging_interval="step")
    logger = CSVLogger(save_dir=str(Path(VAEConfig().outdir)), name="sd_logs")
    trainer = Trainer(
        max_epochs=epochs,
        precision="bf16-mixed",
        accelerator="auto",
        devices="auto",
        accumulate_grad_batches=4,
        num_sanity_val_steps=0,
        limit_val_batches=0, 
        callbacks=[lrmon],
        logger=logger,
        deterministic=False,
    )

    trainer.fit(sd, datamodule=datamodule)
    return sd




def _export_vae_weights(vae: VAE, export_path: Path) -> Path:
    export_path.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(vae, "save_vae_to_file") and callable(getattr(vae, "save_vae_to_file")):
        vae.save_vae_to_file(export_path)
        return export_path
    if hasattr(vae, "export") and callable(getattr(vae, "export")):
        vae.export(export_path)
        return export_path

    payload = {
        "encoder": vae.encoder.state_dict(),
        "decoder": vae.decoder.state_dict(),
    }
    cfg = getattr(vae, "cfg", None)
    if cfg is not None:
        try:
            payload["cfg"] = getattr(cfg, "__dict__", dict(cfg))
        except Exception:
            pass

    torch.save(payload, export_path)
    return export_path

def train_full_workflow(
    ds_path: Path,
    vae_epochs: int,
    sd_epochs: int,
    *,
    vae_export_path: Optional[Path] = None,
    time_dim: int = 320,
    context_dim: int = 128,
    num_train_steps: int = 1000,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    p_uncond: float = 0.1,
    guidance_scale: float = 1.0,
) -> tuple[VAE, StableDiffusion, Path]:

    vae_model = train_vae(ds_path=ds_path, epochs=vae_epochs)
    if vae_export_path is None:
        vae_export_path = Path(VAEConfig().outdir) / "vae_export.pt"
    vae_export_path = _export_vae_weights(vae_model, vae_export_path)

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect() 

    gc.collect()
    torch.cuda.reset_peak_memory_stats() 
    sd_model = train_stable_diffusion(
        ds_path=ds_path,
        vae_path=vae_export_path,
        epochs=sd_epochs,
        time_dim=time_dim,
        context_dim=context_dim,
        num_train_steps=num_train_steps,
        lr=lr,
        weight_decay=weight_decay,
        p_uncond=p_uncond,
        guidance_scale=guidance_scale,
    )

    sd_model.save(Path(vae_export_path).parent / "sd_export.pt")

    return vae_model, sd_model, vae_export_path