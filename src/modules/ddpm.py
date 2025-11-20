import torch
import math
from torch import nn
from torch.nn import functional as F


def timestep_embedding(
    timesteps: torch.Tensor, dim: int, max_period: int = 10000
) -> torch.Tensor:
    if timesteps.ndim == 2 and timesteps.shape[1] == 1:
        timesteps = timesteps[:, 0]

    timesteps = timesteps.to(torch.float32)
    half = dim // 2
    device = timesteps.device
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(0, half, device=device, dtype=torch.float32)
        / half
    )
    args = timesteps[:, None] * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class DDPMSampler:
    def __init__(
        self,
        num_training_steps: int = 1000,
        beta_start: float = 0.0085,
        beta_end: float = 0.012,
        schedule: str = "sqrt_linear",
    ):
        self.T = int(num_training_steps)
        self.schedule = schedule
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)

        if schedule == "sqrt_linear":
            betas = (
                torch.linspace(
                    self.beta_start**0.5,
                    self.beta_end**0.5,
                    self.T,
                    dtype=torch.float32,
                )
                ** 2
            )
        elif schedule == "linear":
            betas = torch.linspace(
                self.beta_start, self.beta_end, self.T, dtype=torch.float32
            )
        elif schedule == "cosine":
            s = 0.008
            steps = torch.arange(self.T + 1, dtype=torch.float64)
            f = torch.cos(((steps / self.T + s) / (1 + s)) * math.pi / 2) ** 2
            alphas_bar = f / f[0]
            betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
            betas = betas.clamp(1e-6, 0.9999).to(torch.float32)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        self.betas_cpu = betas
        self._device = torch.device("cpu")
        self._set_cached_tensors(self.betas_cpu, self._device)

    def _set_cached_tensors(self, betas: torch.Tensor, device: torch.device):
        self.betas = betas.to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)
        self.alphas_bar_prev = torch.cat(
            [
                torch.ones(1, device=device, dtype=self.alphas_bar.dtype),
                self.alphas_bar[:-1],
            ],
            dim=0,
        )
        self.timesteps = torch.arange(
            self.T - 1, -1, -1, device=device, dtype=torch.long
        )
        self.posterior_var = (
            self.betas * (1.0 - self.alphas_bar_prev) / (1.0 - self.alphas_bar)
        )
        self.posterior_log_var_clipped = torch.log(self.posterior_var.clamp(min=1e-20))
        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_bar_prev)) / (
            1.0 - self.alphas_bar
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_bar_prev) * torch.sqrt(self.alphas)
        ) / (1.0 - self.alphas_bar)

    def set_device(self, device: torch.device | str):
        device = torch.device(device)
        if device != self._device:
            self._device = device
            self._set_cached_tensors(self.betas_cpu, device)

    @torch.no_grad()
    def step(
        self,
        eps_pred: torch.Tensor,
        t: int,
        x_t: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        assert 0 <= t < self.T
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alphas_bar[t]

        mean = (
            x_t - ((1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t)) * eps_pred
        ) / torch.sqrt(alpha_t)

        if t == 0:
            return mean

        noise = torch.randn(x_t.shape, dtype=x_t.dtype, device=x_t.device, generator=generator)

        var = self.posterior_var[t]
        return mean + torch.sqrt(var) * noise

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        context: torch.Tensor,
        latent_shape: tuple,
        device: torch.device | str,
        time_dim: int = 320,
        generator: torch.Generator | None = None,
        guidance_scale: float = 1.0,
        uncond_context: torch.Tensor | None = None,
        return_all: bool = False,
    ) -> torch.Tensor | list[torch.Tensor]:
        self.set_device(device)
        device = torch.device(device)

        B = latent_shape[0]
        x = torch.randn(latent_shape, device=device, generator=generator)

        traj = [x.clone()] if return_all else None
        for t in self.timesteps.tolist():
            t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
            t_emb = timestep_embedding(t_tensor, time_dim)

            if uncond_context is not None and guidance_scale != 1.0:
                x_in = torch.cat([x, x], dim=0)
                c_in = torch.cat([uncond_context, context], dim=0)
                t_in = torch.cat([t_emb, t_emb], dim=0)

                eps, _ = model(x_in, c_in, t_in)
                eps_uncond, eps_cond = eps.chunk(2, dim=0)
                eps_pred = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            else:
                eps_pred, _ = model(x, context, t_emb)

            x = self.step(eps_pred, t, x, generator=generator)

            if return_all:
                traj.append(x.clone())

        return traj if return_all else x
