
import torch 
import torch.nn as nn

class ContextEncoder(nn.Module):
    def __init__(self, in_dim: int = 3, out_dim: int = 128, hidden: int = 256, dropout: float = 0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

        self.null_token = nn.Parameter(torch.zeros(1, 1, out_dim))

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        valid = ~torch.isnan(x_raw).any(dim=-1)
        x = torch.nan_to_num(x_raw, nan=0.0)

        tok = self.net(x)

        weights = valid.float().unsqueeze(-1)
        denom = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (tok * weights).sum(dim=1, keepdim=True) / denom 

        empty = (denom.squeeze(-1).squeeze(-1) == 0)
        if empty.any():
            pooled = torch.where(
                empty.view(-1, 1, 1), self.null_token.expand_as(pooled), pooled
            )

        return pooled