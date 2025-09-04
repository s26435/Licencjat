"""
4.09.2025 - Jan Wolski

"""

import torch
from torch import nn
from torch.nn import functional as F
from .attention import SelfAttention
from .utils import safe_num_groups


class VAE_Attention_Block(nn.Module):
    def __init__(self, channels: int):
        super(VAE_Attention_Block, self).__init__()

        self.groupnorm = nn.GroupNorm(safe_num_groups(channels, 32), channels)
        self.att = SelfAttention(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residue = x

        b, c, w, y, z = x.shape
        x = self.groupnorm(x)
        # (batch, channels, x, y, z) -> (batch, channels, x * y * z)
        x = x.view(b, c, w * y * z)

        # (batch, channels, x, y * z) -> (batch, x * y * z, channels)
        x = x.transpose(-1, -2)

        x = self.att(x)

        #  (batch, x, y * z, channels) -> (batch, channels, x, y * z)
        x = x.transpose(-1, -2)

        # (batch, channels, x * y * z) -> (batch, channels, x, y, z)
        x = x.view(b, c, w, y, z)

        return x + residue


class VAE_Residual_Block(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int
    ):
        super(VAE_Residual_Block, self).__init__()


        self.groupnorm1 = nn.GroupNorm(safe_num_groups(in_channels, 32), in_channels)
        self.conv1 = nn.Conv3d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1
        )

        self.groupnorm2 = nn.GroupNorm(safe_num_groups(out_channels, 32), out_channels)
        self.conv2 = nn.Conv3d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                padding=0,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residue = x

        x = F.silu(self.groupnorm1(x))
        x = self.conv1(x)

        x = F.silu(self.groupnorm2(x))
        x = self.conv2(x)

        return x + self.residual_layer(residue)


class VAE_Decoder_Block(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
                nn.Upsample(scale_factor=2),
                nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1),
                VAE_Residual_Block(in_channels, out_channels),
                VAE_Residual_Block(out_channels, out_channels),
                VAE_Residual_Block(out_channels, out_channels),
        )


class VAE_Decoder(nn.Module):
    def __init__(
        self,
        latent_size: int = 4,
        out_channels: int = 3,
        cov_layer_channels: list[int] = [512, 256, 128],
        scale_const = 0.18215
    ):
        super(VAE_Decoder, self).__init__()

        layers = [
            nn.Conv3d(latent_size, latent_size, kernel_size=3, padding=1),
            nn.Conv3d(latent_size, cov_layer_channels[0], kernel_size=3, padding=1),
            VAE_Residual_Block(cov_layer_channels[0], cov_layer_channels[0]),
            VAE_Attention_Block(cov_layer_channels[0]),
            VAE_Residual_Block(cov_layer_channels[0], cov_layer_channels[0]),
            VAE_Residual_Block(cov_layer_channels[0], cov_layer_channels[0]),
            VAE_Residual_Block(cov_layer_channels[0], cov_layer_channels[0]),
            VAE_Residual_Block(cov_layer_channels[0], cov_layer_channels[0]),
            VAE_Decoder_Block(cov_layer_channels[0], cov_layer_channels[0])
        ]

        layers.extend(
            [
                VAE_Decoder_Block(cov_layer_channels[i], cov_layer_channels[i + 1])
                for i in range(len(cov_layer_channels) - 1)
            ]
        )

        layers.extend(
            [
                nn.GroupNorm(
                    safe_num_groups(cov_layer_channels[-1], 32), cov_layer_channels[-1]
                ),
                nn.SiLU(),
                nn.Conv3d(
                    cov_layer_channels[-1], out_channels, kernel_size=3, padding=1
                ),
            ]
        )
        self.scale = scale_const
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch size, latent size, x, y, z)
        """
        x /= self.scale
        for module in self.layers:
            x = module(x)
        return x