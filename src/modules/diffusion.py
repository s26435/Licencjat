import torch
from torch import nn
from torch.nn import functional as F

from .attention import SelfAttention, CrossAttention
from .utils import safe_num_groups


class TimeEmbedding(nn.Module):
    def __init__(self, embeding_size: int):
        super(TimeEmbedding, self).__init__()
        self.dense1 = nn.Linear(embeding_size, 4 * embeding_size)
        self.dense2 = nn.Linear(4 * embeding_size, 4 * embeding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (1, embeding size)
        return: (1, 4*embeding size)
        """
        x = F.silu(self.dense1(x))
        return self.dense2(x)


class SwitchSequential(nn.Sequential):
    def forward(
        self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor
    ) -> torch.Tensor:
        """
        x:
        return:
        """
        for layer in self:
            if isinstance(layer, UNET_Attention_Block):
                x = layer(x, context)
            elif isinstance(layer, UNET_Residual_Block):
                x = layer(x, time)
            else:
                x = layer(x)

        return x


class UNET_Residual_Block(nn.Module):
    def __init__(
        self, input_channels: int, output_channels: int, time_embedding_size: int
    ):
        super(UNET_Residual_Block, self).__init__()
        self.groupnorm1 = nn.GroupNorm(
            safe_num_groups(input_channels, 32), input_channels
        )
        self.conv1 = nn.Conv3d(
            input_channels, output_channels, kernel_size=3, padding=1
        )
        self.dense = nn.Linear(time_embedding_size * 4, output_channels)

        self.groupnorm2 = nn.GroupNorm(
            safe_num_groups(output_channels, 32), output_channels
        )
        self.conv2 = nn.Conv3d(
            output_channels, output_channels, kernel_size=3, padding=1
        )

        self.residual_layer = (
            nn.Identity()
            if input_channels == output_channels
            else nn.Conv3d(input_channels, output_channels, kernel_size=1, padding=0)
        )

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """
        x: (batch size, in channels, x, y, z)
        time: (1, time embedding size * 4)
        return:
        """
        residue = x
        x = self.conv1(F.silu(self.groupnorm1(x)))
        time = (
            self.dense(F.silu(time)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        )  # (B, C_out, 1,1,1)
        merged = x + time
        merged = self.conv2(F.silu(self.groupnorm2(merged)))
        return merged + self.residual_layer(residue)


class UNET_Output_Layer(nn.Module):
    def __init__(self, time_embedding_size: int, latent_size: int):
        super(UNET_Output_Layer, self).__init__()
        self.groupnorm = nn.GroupNorm(
            safe_num_groups(time_embedding_size, 32), time_embedding_size
        )
        self.conv = nn.Conv3d(
            time_embedding_size, latent_size, kernel_size=3, padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch size, time embedding size, x, y, z)
        return: (batch size, latent space size, x, y, z)
        """
        x = F.silu(self.groupnorm(x))
        return self.conv(x)


class UNET_Attention_Block(nn.Module):
    def __init__(self, n_heads: int, embedding_size: int, dim_context: int):
        super(UNET_Attention_Block, self).__init__()
        num_channels = n_heads * embedding_size
        self.groupnorm = nn.GroupNorm(
            safe_num_groups(num_channels, 32), num_channels, eps=1e-6
        )
        self.conv_in = nn.Conv3d(num_channels, num_channels, kernel_size=1, padding=0)
        self.layernorm1 = nn.LayerNorm(num_channels)
        self.att1 = SelfAttention(n_heads, num_channels, in_bias=False)
        self.layernorm2 = nn.LayerNorm(num_channels)
        self.att2 = CrossAttention(n_heads, num_channels, dim_context, in_bias=False)
        self.layernorm3 = nn.LayerNorm(num_channels)
        self.dense1 = nn.Linear(num_channels, 2 * 4 * num_channels)
        self.dense2 = nn.Linear(4 * num_channels, num_channels)

        self.conv_out = nn.Conv3d(num_channels, num_channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        x: (batch sieze, latent_size, x, y, z)
        context: (batch, sequence length, dim)
        return: (batch sieze, latent_size, x, y, z)
        """
        long_residue = x
        x = self.conv_in(self.groupnorm(x))
        b, c, w, h, d = x.shape
        # (batch sieze, latent_size, x, y, z) -> (batch sieze, latent_size, x * y* z)
        x = x.view((b, c, w * h * d))
        # (batch sieze, latent_size, x * y* z) -> (batch sieze, x * y* z, latent_size)
        x = x.transpose(-1, -2)

        short_residue = x
        x = self.att1(self.layernorm1(x)) + short_residue
        short_residue = x

        x = self.layernorm2(x)
        x = self.att2(x, context) + short_residue

        short_residue = x
        x = self.layernorm3(x)
        x, gate = self.dense1(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)
        x = self.dense2(x) + short_residue

        x = x.transpose(-1, -2).view((b, c, w, h, d))

        return self.conv_out(x) + long_residue


class UNET_Upsample(nn.Module):
    def __init__(self, channels: int):
        super(UNET_Upsample, self).__init__()
        self.conv = nn.Conv3d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch size, channels, x, y, z)
        return: (batch size, channels, x*2, y*2, z*2)
        """
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class UNET(nn.Module):
    @staticmethod
    def _resize_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if x.shape[2:] == ref.shape[2:]:
            return x
        return F.interpolate(x, size=ref.shape[2:], mode="nearest")

    def __init__(
        self, context_dim: int, latent_size: int, T: int, H: int = 8, A: int = 40
    ):
        super(UNET, self).__init__()
        self.context_dim = context_dim
        self.T = T
        self.H = H
        self.A = A

        # ---- encoder jak miałeś ----
        self.encoder = nn.ModuleList(
            [
                SwitchSequential(nn.Conv3d(latent_size, T, kernel_size=3, padding=1)),
                SwitchSequential(
                    UNET_Residual_Block(T, T, T),
                    UNET_Attention_Block(H, A, context_dim),
                ),
                SwitchSequential(
                    UNET_Residual_Block(T, T, T),
                    UNET_Attention_Block(H, A, context_dim),
                ),
                SwitchSequential(
                    nn.Conv3d(T, T, kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0)
                ),
                SwitchSequential(
                    UNET_Residual_Block(T, 2 * T, T),
                    UNET_Attention_Block(H, 2 * A, context_dim),
                ),
                SwitchSequential(
                    UNET_Residual_Block(2 * T, 2 * T, T),
                    UNET_Attention_Block(H, 2 * A, context_dim),
                ),
                SwitchSequential(
                    nn.Conv3d(
                        2 * T, 2 * T, kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0
                    )
                ),
                SwitchSequential(
                    UNET_Residual_Block(2 * T, 2 * T, T),
                    UNET_Attention_Block(H, 2 * A, context_dim),
                ),
                SwitchSequential(
                    UNET_Residual_Block(2 * T, 2 * T, T),
                    UNET_Attention_Block(H, 2 * A, context_dim),
                ),
            ]
        )

        self.bottle_neck = SwitchSequential(
            UNET_Residual_Block(2 * T, 2 * T, T),
            UNET_Attention_Block(H, 2 * A, context_dim),
            UNET_Residual_Block(2 * T, 2 * T, T),
        )

        # ---- decoder jak u Ciebie ----
        self.decoder = nn.ModuleList(
            [
                SwitchSequential(
                    UNET_Residual_Block(4 * T, 2 * T, T),
                    UNET_Attention_Block(H, 2 * A, context_dim),
                ),
                SwitchSequential(
                    UNET_Residual_Block(4 * T, 2 * T, T),
                    UNET_Attention_Block(H, 2 * A, context_dim),
                    UNET_Upsample(2 * T),
                ),

                SwitchSequential(
                    UNET_Residual_Block(4 * T, T, T),
                    UNET_Attention_Block(H, A, context_dim),
                ),
                SwitchSequential(
                    UNET_Residual_Block(3 * T, T, T),
                    UNET_Attention_Block(H, A, context_dim),
                    UNET_Upsample(T),
                ),

                SwitchSequential(
                    UNET_Residual_Block(2 * T, T, T),
                    UNET_Attention_Block(H, A, context_dim),
                ),
                SwitchSequential(
                    UNET_Residual_Block(2 * T, T, T),
                    UNET_Attention_Block(H, A, context_dim),
                ),
            ]
        )

        # ---- dodatkowe projekcje X -> sekwencja o wymiarze 2*T ----
        self.proj_low  = nn.Conv3d(2 * T, 2 * T, kernel_size=1)  # po pierwszym upsamplu
        self.proj_mid  = nn.Conv3d(T,     2 * T, kernel_size=1)  # po drugim upsamplu
        self.proj_top  = nn.Conv3d(T,     2 * T, kernel_size=1)  # finalna skala (opcjonalnie)

        # ---- HEAD Y: atencja z kontekstem i z X podczas upsamplowania ----
        self.bottom_enocoder = nn.ModuleList(
            [
                # start z feature map po bottlenecku: (B, 2*T, D', H', W')
                nn.Conv3d(2 * T, 2 * T, kernel_size=1, stride=1, padding=0),
                nn.Conv3d(2 * T, 2 * T, kernel_size=1, stride=1, padding=0),

                # Y <-> context (tekst)
                CrossAttention(H, 2 * T, context_dim),

                # Y <-> X_low
                CrossAttention(H, 2 * T, 2 * T),

                # Y <-> X_mid
                CrossAttention(H, 2 * T, 2 * T),

                # Y <-> X_top
                CrossAttention(H, 2 * T, 2 * T),

                # na koniec wektor (B, 6)
                nn.Linear(2 * T, 6),
            ]
        )

    def _fmap_to_seq(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, D, H, W) -> (B, L, C)
        """
        B, C, D, H, W = x.shape
        return x.view(B, C, D * H * W).transpose(1, 2)


    def forward(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor):
        B, C, D, H_, W_ = x.shape

        x = self.encoder[0](x, context, time)
        x = self.encoder[1](x, context, time)
        x = self.encoder[2](x, context, time)
        skip_top = x

        x = self.encoder[3](x, context, time)
        x = self.encoder[4](x, context, time)
        x = self.encoder[5](x, context, time)
        skip_mid = x

        x = self.encoder[6](x, context, time)
        x = self.encoder[7](x, context, time)
        x = self.encoder[8](x, context, time)
        skip_low = x  

        x = self.bottle_neck(x, context, time)

        y = self.bottom_enocoder[0](x)
        y = self.bottom_enocoder[1](y)
        y = self._fmap_to_seq(y)

        y = self.bottom_enocoder[2](y, context)

        x = self._resize_like(x, skip_low)
        x = torch.cat([x, skip_low], dim=1)
        x = self.decoder[0](x, context, time)

        x = self._resize_like(x, skip_low)
        x = torch.cat([x, skip_low], dim=1)
        x = self.decoder[1](x, context, time)  # zawiera Upsample -> wyższa H,W

        # Y patrzy na X po pierwszym upsamplu
        x_low = self.proj_low(x)                # (B, 2*T, D1, H1, W1)
        x_low_seq = self._fmap_to_seq(x_low)    # (B, L_low, 2*T)
        y = self.bottom_enocoder[3](y, x_low_seq)

        # skala "mid"
        x = self._resize_like(x, skip_mid)
        x = torch.cat([x, skip_mid], dim=1)     # (B, 4*T, ...)
        x = self.decoder[2](x, context, time)   # (B, T, ...)

        x = self._resize_like(x, skip_mid)
        x = torch.cat([x, skip_mid], dim=1)     # (B, 3*T, ...)
        x = self.decoder[3](x, context, time)   # Upsample

        # Y patrzy na X po drugim upsamplu
        x_mid = self.proj_mid(x)                # (B, 2*T, D2, H2, W2)
        x_mid_seq = self._fmap_to_seq(x_mid)    # (B, L_mid, 2*T)
        y = self.bottom_enocoder[4](y, x_mid_seq)

        # skala "top"
        x = self._resize_like(x, skip_top)
        x = torch.cat([x, skip_top], dim=1)     # (B, 2*T, ...)
        x = self.decoder[4](x, context, time)   # (B, T, ...)

        x = self._resize_like(x, skip_top)
        x = torch.cat([x, skip_top], dim=1)     # (B, 2*T, ...)
        x = self.decoder[5](x, context, time)   # (B, T, D_out, H_out, W_out)

        # Y patrzy na finalne X (opcjonalne, ale czemu nie)
        x_top = self.proj_top(x)                # (B, 2*T, D3, H3, W3)
        x_top_seq = self._fmap_to_seq(x_top)    # (B, L_top, 2*T)
        y = self.bottom_enocoder[5](y, x_top_seq)

        y = y.mean(dim=1)
        y = self.bottom_enocoder[6](y) 
        return x, y


class Diffusion(nn.Module):
    def __init__(
        self,
        context_dim: int,
        time_embedding_size: int = 320,
        latent_space_size: int = 4,
    ):
        super(Diffusion, self).__init__()
        self.time_embedding = TimeEmbedding(time_embedding_size)
        self.unet = UNET(context_dim, latent_space_size, time_embedding_size)
        self.final = UNET_Output_Layer(time_embedding_size, latent_space_size)

    def forward(
        self, latent: torch.Tensor, context: torch.Tensor, time: torch.Tensor
    ) -> torch.Tensor:
        """
        latent: (batch size, latent size, x/8, y/8, z/8)
        context: (natch size, sequence length, dim)
        time: (1, time_embedding_size)
        return:
        """
        time = self.time_embedding(time)

        output, y = self.unet(latent, context, time)
        return self.final(output), y
