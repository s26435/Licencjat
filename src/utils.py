import torch


def safe_num_groups(ch, preferred=32):
    if ch % preferred == 0:
        return preferred
    for g in (16, 8, 4, 2, 1):
        if ch % g == 0:
            return g
    return 1


class QuickGELU(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)
