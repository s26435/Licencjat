"""
4.09.2025 - Jan Wolski
VAE Encoder for 4D stable diffusion
"""

import torch
from torch import nn
from torch.nn import functional as F
from .decoder import VAE_Attention_Block, VAE_Residual_Block
from .utils import safe_num_groups


class VAE_Encoder_Block(nn.Sequential):
    def __init__(self, in_channels, out_channels, downsample: bool = True):
        super().__init__(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=2 if downsample else 1,
                padding=1,
            ),
            VAE_Residual_Block(out_channels, out_channels),
            VAE_Residual_Block(out_channels, out_channels),
        )


class VAE_Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        latent_size: int = 4,
        num_groups: int = 32,
        conv_layers_channels: list[int] = [128, 256, 512],
        variance_clamp=30,
        scale_const=0.18215,
    ):
        """
        (batch_size, in_channels, x, y, z) -> (batch_size, 2*latent_size, x/8. y/8, z/8)
        TODO dodać walidację wejścia
        """

        super(VAE_Encoder, self).__init__()

        self.in_channels = in_channels
        self.latent_size = latent_size
        self.variance_clamp = variance_clamp
        self.scale = scale_const

        layers = []
        layers.append(VAE_Encoder_Block(in_channels, conv_layers_channels[0]))

        for i in range(len(conv_layers_channels) - 1):
            layers.append(
                VAE_Encoder_Block(conv_layers_channels[i], conv_layers_channels[i + 1])
            )

        layers.extend(
            [
                VAE_Residual_Block(conv_layers_channels[-1], conv_layers_channels[-1]),
                VAE_Residual_Block(conv_layers_channels[-1], conv_layers_channels[-1]),
                VAE_Residual_Block(conv_layers_channels[-1], conv_layers_channels[-1]),
                VAE_Attention_Block(conv_layers_channels[-1]),
                VAE_Residual_Block(conv_layers_channels[-1], conv_layers_channels[-1]),
                nn.GroupNorm(
                    safe_num_groups(conv_layers_channels[-1], num_groups),
                    conv_layers_channels[-1],
                ),
                nn.SiLU(),
                nn.Conv3d(
                    conv_layers_channels[-1], latent_size * 2, kernel_size=1, padding=0
                ),
            ]
        )

        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        x:     (batch_size, in_channels,   x,   y,   z)
        noise: (batch_size, latent_size, x/8, y/8, z/8)
        TODO dodać walidacje wejścia i noise
        """

        for module in self.layers:
            if getattr(module, "stride", None) == (2, 2, 2):
                # asymethric padding
                x = F.pad(x, (1, 0, 1, 0, 1, 0))

            x = module(x)

        # mean, std: (batch_size, out_channels, x/8, y/8, z/8)
        mean, log_var = torch.chunk(x, 2, dim=1)
        std = torch.exp(
            0.5 * torch.clamp(log_var, -self.variance_clamp, self.variance_clamp)
        )

        return (mean + std * noise) * self.scale
