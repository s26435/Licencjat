from dataclasses import dataclass

@dataclass
class VAEConfig:
    latent_channels: int = 4
    downscale: int = 8
    beta_kl: float = 0.25
    rec_loss: str = "l1"
    lr: float = 1e-4
    weight_decay: float = 1e-5
    outdir: str = "checkpoints"
    sample_every_n_epochs: int = 1
    sample_size: int = 64
    grad_clip: float = 1.0
    beta_kl_max: float = 0.25
    beta_kl_warmup_steps: int = 10_000
    use_free_bits: bool = True
    kl_free_nats: float = 0.5 

