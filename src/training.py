import pytorch_lightning as pl
import torch
from .utils import VolumetricDataModule
from .stable_diffusion import DiffusionSystem
from .configs import VAEConfig, LitConfig
from .vae import VAESystem

def train_new_vae():
    pl.seed_everything(42, workers=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    cfg = VAEConfig(
        latent_channels=4,
        downscale=8,
        beta_kl=0.25,
        rec_loss="l1",
        lr=1e-4,
        weight_decay=1e-5,
        outdir="checkpoints_vae",
        sample_every_n_epochs=1,
        sample_size=64,
        grad_clip=1.0,
    )

    system = VAESystem(cfg)
    dm = VolumetricDataModule(batch_size=1, num_workers=4, size=cfg.sample_size)

    ckpt = pl.callbacks.ModelCheckpoint(
        dirpath=cfg.outdir,
        filename="vae-{epoch:03d}-{val_loss:.4f}",
        save_top_k=3, monitor="val/loss", mode="min", save_last=True
    )
    lrmon = pl.callbacks.LearningRateMonitor(logging_interval="step")
    logger = pl.loggers.CSVLogger("logs", name="pl_vae3d")

    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto",
        precision="32-true",        # na start trzymaj FP32; potem można dać "16-mixed"
        num_sanity_val_steps=0,
        gradient_clip_val=cfg.grad_clip,
        log_every_n_steps=20,
        callbacks=[ckpt, lrmon],
        logger=logger,
    )

    trainer.fit(system, dm)


def train_new_diffusion():
    pl.seed_everything(42, workers=True)

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    cfg = LitConfig(
        T=1000,
        beta_schedule="sqrt_linear",
        time_dim=64,
        context_dim=768,
        latent_scale=0.18215,
        lr=1e-5,
        weight_decay=1e-4,
        warmup_steps=2000,
        use_clip=True,
        freeze_clip=True,
        freeze_decoder=True,
        sample_every_n_epochs=1,
        sample_bs=1,
        sample_size=64,
        guidance_scale=1.0,
        outdir="checkpoints",
    )

    system = DiffusionSystem(cfg)
    dm = VolumetricDataModule(batch_size=2, num_workers=4, size=64)

    ckpt = pl.callbacks.ModelCheckpoint(
        dirpath=cfg.outdir,
        filename="diff3d-{epoch:03d}-{val_loss:.4f}",
        save_top_k=3,
        monitor="val/loss",
        mode="min",
        save_last=True,
    )

    lrmon = pl.callbacks.LearningRateMonitor(logging_interval="step")
    logger = pl.loggers.CSVLogger("logs", name="pl_diffusion3d")

    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto",
        precision="16-mixed",
        num_sanity_val_steps=0,
        gradient_clip_val=1.0,
        accumulate_grad_batches=1,
        log_every_n_steps=50,
        callbacks=[ckpt, lrmon],
        logger=logger,
    )

    trainer.fit(system, dm)