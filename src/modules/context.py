
import torch 
import torch.nn as nn

class ContextEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int = 3,
        out_dim: int = 128,
        hidden: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        complex_in_dim = in_dim * 2

        self.net = nn.Sequential(
            nn.Linear(complex_in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

        self.null_token = nn.Parameter(torch.zeros(1, 1, out_dim))

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        nan_mask = torch.isnan(x_raw)
        x = torch.cat([torch.nan_to_num(x_raw, nan=0.0), nan_mask.to(x_raw.dtype)], dim=-1)
        tok = self.net(x)
        weights = (~nan_mask).any(dim=-1).float().unsqueeze(-1)
        denom_raw = weights.sum(dim=1, keepdim=True)
        pooled = (tok * weights).sum(dim=1, keepdim=True) / denom_raw.clamp_min(1.0)
        empty = (denom_raw.squeeze(-1).squeeze(-1) == 0)

        if empty.any():
            pooled = torch.where(
                empty.view(-1, 1, 1),
                self.null_token.expand_as(pooled),
                pooled,
            )
        return pooled