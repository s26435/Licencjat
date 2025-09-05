from dataclasses import dataclass
from typing import Optional

@dataclass
class VAEConfig:
    latent_channels: int = 4
    downscale: int = 8
    beta_kl: float = 0.25
    rec_loss: str = "l1"
    lr: float = 1e-4
    weight_decay: float = 1e-5
    outdir: str = "checkpoints_vae"
    sample_every_n_epochs: int = 1
    sample_size: int = 64
    grad_clip: float = 1.0
    beta_kl_max: float = 0.25
    beta_kl_warmup_steps: int = 10_000
    use_free_bits: bool = True
    kl_free_nats: float = 0.5 

@dataclass
class LitConfig:
    T: int = 1000
    beta_schedule: str = "sqrt_linear"
    time_dim: int = 320
    context_dim: int = 768
    latent_scale: float = 1.0
    latent_channels: int = 4 
    downscale: int = 8 
    lr: float = 1e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 0
    use_clip: bool = True
    freeze_clip: bool = True
    freeze_decoder: bool = True
    sample_every_n_epochs: int = 1
    sample_bs: int = 1
    sample_size: int = 64
    guidance_scale: float = 1.0
    outdir: str = "checkpoints"
    vae_weights: Optional[str] = "checkpoints_vae/vae_export.pt"
    freeze_vae: bool = True