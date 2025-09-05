import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from typing import Optional


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
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
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


def models_summary():
    from .encoder import VAE_Encoder
    from .decoder import VAE_Decoder
    from .clip import CLIP
    from .diffusion import Diffusion

    device = "cuda" if torch.cuda.is_available() else "cpu"
    decoder = VAE_Decoder().to(device=device)
    encoder = VAE_Encoder().to(device=device)
    clip = CLIP().to(device=device)
    diff = Diffusion(768).to(device=device)

    report_model("VAE_Encoder", encoder)
    report_model("VAE_Decoder", decoder)
    report_model("CLIP", clip)
    report_model("Diffusion", diff)
    total_bytes = sum(model_size_bytes(m)[2] for m in [encoder, decoder, clip, diff])
    print(f"====================\nSUMA rozmiarów modeli: {human_bytes(total_bytes)}")


class Dummy3DDataset(Dataset):
    def __init__(self, n=1024, size=64, tok_len=77, channels=3, vmin=0.0, vmax=100.0):
        self.n = n
        self.size = size
        self.tok_len = tok_len
        self.channels = channels
        self.vmin = float(vmin)
        self.vmax = float(vmax)

        # precompute bazowy gradient (1, D, H, W)
        lin = torch.linspace(0.0, 1.0, steps=size)
        # siatka o indeksowaniu ij: Z, Y, X
        z, y, x = torch.meshgrid(lin, lin, lin, indexing="ij")
        grad01 = (x + y + z) / 3.0  # 0..1 po przekątnej
        base = self.vmin + grad01 * (self.vmax - self.vmin)
        self.base = base.unsqueeze(0).to(torch.float32)  # (1, S, S, S)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # powiel przez kanały: (C, S, S, S)
        x = self.base.repeat(self.channels, 1, 1, 1).clone()
        tokens = torch.randint(0, 49408, (self.tok_len,), dtype=torch.long)
        return x, tokens


class VolumetricDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 2, num_workers: int = 4, size: int = 64):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.size = size

    def setup(self, stage: Optional[str] = None):
        self.train_ds = Dummy3DDataset(n=2048, size=self.size)
        self.val_ds = Dummy3DDataset(n=64, size=self.size)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
