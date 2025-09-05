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

def human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    i = 0
    n = float(n)
    while n >= 1024 and i < len(units) - 1:
        n /= 1024.0
        i += 1
    return f"{n:.2f} {units[i]}"

def model_size_bytes(model: torch.nn.Module) -> tuple[int, int, int]:
    param_bytes  = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    return param_bytes, buffer_bytes, param_bytes + buffer_bytes

def count_params(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def guess_dtype(model: torch.nn.Module):
    first = next(model.parameters(), None)
    return str(first.dtype) if first is not None else "n/a"

def report_model(name: str, model: torch.nn.Module):
    total, trainable = count_params(model)
    p_b, b_b, t_b = model_size_bytes(model)
    print(
        f"{name:>14}: "
        f"params={total:,} (trainable={trainable:,}) | "
        f"size={human_bytes(p_b)} + buffers={human_bytes(b_b)} = {human_bytes(t_b)} | "
        f"dtype={guess_dtype(model)}"
    )