import torch
import numpy as np
from typing import Union
from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifWriter
from pathlib import Path
from src.unimat.globals import PERIOD_GROUP_TO_SYM
from src.stable_diffusion import StableDiffusion

def unimat_to_structure(
    grid: Union[np.ndarray, torch.Tensor],
    cell_params: Union[np.ndarray, torch.Tensor],
    pad_value: float = -1.0,
) -> Structure:
    if isinstance(grid, torch.Tensor):
        grid_np = grid.detach().cpu().numpy()
    else:
        grid_np = np.asarray(grid, dtype=np.float32)

    if isinstance(cell_params, torch.Tensor):
        cell_np = cell_params.detach().cpu().numpy()
    else:
        cell_np = np.asarray(cell_params, dtype=np.float32)

    if grid_np.ndim != 4 or grid_np.shape[-1] != 3:
        raise ValueError(f"grid expected shape [P,G,K,3], got {tuple(grid_np.shape)}")
    if cell_np.shape[-1] != 6:
        raise ValueError(f"cell_params expected shape [6], got {tuple(cell_np.shape)}")

    P, G, K, _ = grid_np.shape

    a, b, c, alpha, beta, gamma = map(float, cell_np.tolist())
    lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma)

    species: list[str] = []
    frac_coords: list[list[float]] = []

    for pi in range(P):
        period = pi + 1
        for gi in range(G):
            group = gi + 1
            key = (period, group)
            if key not in PERIOD_GROUP_TO_SYM:
                continue
            sym = PERIOD_GROUP_TO_SYM[key]

            for ki in range(K):
                xyz = grid_np[pi, gi, ki, :]
                if np.allclose(xyz, pad_value, atol=1e-5):
                    continue
                if np.any(np.isnan(xyz)):
                    continue

                species.append(sym)
                frac_coords.append(xyz.tolist())

    if not species:
        raise ValueError("Brak atomów z gridu (wszystko pad_value / NaN).")

    struct = Structure(lattice, species, frac_coords, coords_are_cartesian=False)
    return struct


def unimat_to_cif_file(
    grid: Union[np.ndarray, torch.Tensor],
    cell_params: Union[np.ndarray, torch.Tensor],
    out_path: Union[str, Path],
    pad_value: float = -1.0,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    struct = unimat_to_structure(grid, cell_params, pad_value=pad_value)
    writer = CifWriter(struct)
    writer.write_file(str(out_path))
    return out_path





def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sd = StableDiffusion.load(
        ckpt_path="checkpoints/sd_export.pt",
        vae_path="checkpoints/vae_export.pt",
        map_location=device,
    ).to(device)
    sd.eval()
    latent_shape_hwz = (8, 8, 4)

    # [B, L, 3]
    ctx_raw = None

    grid, cell = sd.generate_unimat_and_cell(
        batch_size=1,
        latent_shape_hwz=latent_shape_hwz,
        ctx_raw=ctx_raw,
        ctx_raw_uncond=None,
        guidance_scale=7.5,
    )
    grid0 = grid[0]
    cell0 = cell[0]

    out_cif = unimat_to_cif_file(grid0, cell0, "outputs/generated_001.cif")
    print("Zapisano CIF do:", out_cif)

if __name__ == "__main__":
    main()
