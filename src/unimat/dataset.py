from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any 
import json
import warnings
import dotenv
from datetime import datetime

import numpy as np
import pandas as pd

from mp_api.client import MPRester
from emmet.core.summary import HasProps
from pymatgen.io.cif import CifWriter
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

import torch
from torch.utils.data import Dataset, DataLoader, random_split

import pytorch_lightning as pl

from .globals import (
    MAIN_DS_DIR,
    CIF_SUBDIR,
    CSV_NAME,
    TORCH_OUT,
    LIMIT,
    MAX_GROUPS,
    MAX_PERIODS,
    MAX_ATOMS_PER_ELEMENT,
    SYM_TO_PERIOD_GROUP,
)

from .ds_utils import (
    _ensure_dir,
    _ensure_structure,
    normalize_label_file_only,
    _get_bs_rows,
    _build_label_vocab,
    _parse_ev_triplet,
)

warnings.filterwarnings("ignore", "", FutureWarning)
dotenv.load_dotenv()


def _download_cif_by_id(
    mpr: MPRester,
    mpid: str,
    folder: Path,
    symprec: float = 1e-2,
    angle_tol: float = 5.0,
    force_primitive: bool = True,
):
    raw = mpr.get_structure_by_material_id(mpid)
    structure = _ensure_structure(raw)
    sga = SpacegroupAnalyzer(structure, symprec=symprec, angle_tolerance=angle_tol)
    prim = sga.get_primitive_standard_structure() if force_primitive else structure
    if prim is None or prim.num_sites >= structure.num_sites:
        prim = structure
    folder.mkdir(parents=True, exist_ok=True)
    out_path = folder / f"{mpid}.cif"
    CifWriter(prim, symprec=symprec).write_file(out_path)


def _read_cif_as_structure(
    cif_path: Path,
    force_primitive: bool = True,
    symprec: float = 1e-2,
    angle_tol: float = 5.0,
) -> Structure:
    s = Structure.from_file(str(cif_path))
    if force_primitive:
        sga = SpacegroupAnalyzer(s, symprec=symprec, angle_tolerance=angle_tol)
        prim = sga.get_primitive_standard_structure()
        if prim is not None and prim.num_sites <= s.num_sites:
            return prim
    return s


def _structure_to_atoms_and_cell(
    structure: Structure,
) -> Tuple[List[Tuple[str, List[float]]], Dict[str, float]]:
    atoms: List[Tuple[str, List[float]]] = []
    frac = structure.frac_coords
    for i, site in enumerate(structure.sites):
        sym = site.species_string.split()[0]
        atoms.append((sym, [float(frac[i, 0]), float(frac[i, 1]), float(frac[i, 2])]))
    lat = structure.lattice
    cell = {
        "a": float(lat.a),
        "b": float(lat.b),
        "c": float(lat.c),
        "alpha": float(lat.alpha),
        "beta": float(lat.beta),
        "gamma": float(lat.gamma),
    }
    return atoms, cell


def _build_unimat_grid(
    atoms: List[Tuple[str, List[float]]],
    max_atoms_per_element: int = 8,
    pad_value: float = -1,
) -> Tuple[np.ndarray, Dict[Tuple[int, int], int]]:
    grid = np.full(
        (MAX_PERIODS, MAX_GROUPS, max_atoms_per_element, 3), pad_value, dtype=np.float32
    )
    counts: Dict[Tuple[int, int], int] = {}
    bucketed: Dict[Tuple[int, int], List[List[float]]] = {}

    for sym, frac in atoms:
        if sym not in SYM_TO_PERIOD_GROUP:
            continue
        p, g = SYM_TO_PERIOD_GROUP[sym]
        bucketed.setdefault((p, g), []).append(frac)

    for (p, g), lst in bucketed.items():
        chosen = lst[:max_atoms_per_element]
        counts[(p, g)] = len(chosen)
        for i, xyz in enumerate(chosen):
            grid[p - 1, g - 1, i, :] = np.asarray(xyz, dtype=np.float32)

    return grid, counts


class UniMatTorchDataset(Dataset):
    def __init__(self, dataset_root: str, ids: Optional[List[str]] = None):
        self.root = Path(dataset_root)
        idx = json.load(open(self.root / "index.json", "r", encoding="utf-8"))
        self.ids = ids or idx["ids"]
        self.ctx_meta = idx.get("context", {})
        self.labels = self.ctx_meta.get("labels", [])
        self.num_labels = len(self.labels)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i: int):
        mpid = self.ids[i]
        npz_path = self.root / "samples" / f"{mpid}.npz"
        with np.load(npz_path, allow_pickle=False) as data:
            grid = torch.from_numpy(data["grid"]).float()  # [P,G,K,3]
            cell = torch.from_numpy(data["cell_params"]).float()  # [6]
            context = torch.from_numpy(data["context"]).float()  # [L,3]
        return {"id": mpid, "grid": grid, "cell_params": cell, "context": context}

def _to_ch_first(grid: torch.Tensor) -> torch.Tensor:
    if grid.dim() != 4 or grid.size(-1) != 3:
        raise ValueError(f"grid expected shape [P,G,K,3], got {tuple(grid.shape)}")
    return grid.permute(3, 0, 1, 2).contiguous()


def _pad_context(batch_ctx: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    B = len(batch_ctx)
    L_max = max(int(c.size(0)) for c in batch_ctx) if B > 0 else 0
    if L_max == 0:
        return torch.zeros(B, 1, 3), torch.zeros(B, 1, dtype=torch.bool)

    ctx_padded = torch.zeros(B, L_max, batch_ctx[0].size(-1), dtype=batch_ctx[0].dtype)
    mask = torch.zeros(B, L_max, dtype=torch.bool)
    for i, c in enumerate(batch_ctx):
        L = c.size(0)
        ctx_padded[i, :L] = c
        mask[i, :L] = True
    return ctx_padded, mask

class GridCtxDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_root: str | Path,
        batch_size: int = 4,
        num_workers: int = 4,
        val_split: float = 0.1,
        test_split: float = 0.0,
        seed: int = 42,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        drop_last: bool = False,
        return_ctx_mask: bool = True,
    ):
        super().__init__()
        self.dataset_root = str(dataset_root)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = float(val_split)
        self.test_split = float(test_split)
        self.seed = seed
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.drop_last = drop_last
        self.return_ctx_mask = return_ctx_mask

        self.ds_full: Optional[UniMatTorchDataset] = None
        self.ds_train = self.ds_val = self.ds_test = None

    def setup(self, stage: Optional[str] = None):
        if self.ds_full is not None:
            return
        self.ds_full = UniMatTorchDataset(self.dataset_root)

        n = len(self.ds_full)
        n_test = int(round(n * self.test_split))
        n_val = int(round((n - n_test) * self.val_split))
        n_train = n - n_val - n_test
        g = torch.Generator().manual_seed(self.seed)
        self.ds_train, self.ds_val, self.ds_test = random_split(self.ds_full, [n_train, n_val, n_test], generator=g)

    def _collate_with_context(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        ids = [s["id"] for s in samples]
        x = torch.stack([_to_ch_first(s["grid"]) for s in samples], dim=0)      # [B, 3, P, G, K]
        cell = torch.stack([s["cell_params"] for s in samples], dim=0)          # [B, 6]
        ctx_list = [s["context"] for s in samples]                               # L_i x 3
        ctx_padded, mask = _pad_context(ctx_list)                                # [B, Lmax, 3], [B, Lmax]
        batch = {"id": ids, "x": x, "cell_params": cell, "context": ctx_padded}
        if self.return_ctx_mask:
            batch["context_mask"] = mask
        return batch

    def train_dataloader(self):
        return DataLoader(
            self.ds_train, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers, drop_last=self.drop_last,
            collate_fn=self._collate_with_context,
        )

    def val_dataloader(self):
        if len(self.ds_val) == 0:
            return None
        return DataLoader(
            self.ds_val, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers, drop_last=False,
            collate_fn=self._collate_with_context,
        )

    def test_dataloader(self):
        if len(self.ds_test) == 0:
            return None
        return DataLoader(
            self.ds_test, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers, drop_last=False,
            collate_fn=self._collate_with_context,
        )


def _collate_fixed(batch: List[Dict[str, torch.Tensor]]):
    ids = [b["id"] for b in batch]
    grid = torch.stack([b["grid"] for b in batch], 0)  # [B,P,G,K,3]
    cell = torch.stack([b["cell_params"] for b in batch], 0)  # [B,6]
    ctx = torch.stack([b["context"] for b in batch], 0)  # [B,L,3]
    return {"id": ids, "grid": grid, "cell_params": cell, "context": ctx}

def all_in_one(
    main_ds_dir: str = MAIN_DS_DIR,
    cif_subdir: str = CIF_SUBDIR,
    csv_name: str = CSV_NAME,
    torch_out: str = TORCH_OUT,
    limit: Optional[int] = LIMIT,
    max_atoms_per_element: int = MAX_ATOMS_PER_ELEMENT,
):
    start = datetime.now()
    cif_out = _ensure_dir(Path(main_ds_dir) / cif_subdir)

    print("1) Pobieram band structures i CIF-y...")
    whole = pd.DataFrame(
        {
            "material_id": pd.Series(dtype="str"),
            "label": pd.Series(dtype="str"),
            "ev": pd.Series(dtype="object"),
        }
    )
    with MPRester() as mpr:
        docs = mpr.materials.summary.search(
            has_props=[HasProps.bandstructure], fields=["material_id"]
        )
        mpids = [str(doc.material_id) for doc in docs]
        total = limit if limit is not None else len(mpids)
        for i, mpid in enumerate(mpids, start=1):
            if limit is not None and i > limit:
                break
            print(f"[{i}/{total}] {mpid}")
            try:
                df_one = _get_bs_rows(mpr, mpid)
                whole = pd.concat([whole, df_one], axis=0, ignore_index=True)
            except Exception as e:
                print(f"[WARN] BS {mpid}: {e}")
            try:
                _download_cif_by_id(mpr, mpid, cif_out)
            except Exception as e:
                print(f"[WARN] CIF {mpid}: {e}")

    whole["label"] = whole["label"].map(normalize_label_file_only)
    csv_out = Path(main_ds_dir) / csv_name
    whole.to_csv(csv_out, index=False)

    print("2) Buduję UNIMAT grid i context po labelach...")
    out_root = _ensure_dir(Path(main_ds_dir) / torch_out)
    samp_dir = _ensure_dir(out_root / "samples")
    meta_dir = _ensure_dir(out_root / "meta")

    label_vocab = _build_label_vocab(whole)
    by_mpid = {k: v for k, v in whole.groupby("material_id")}

    cif_paths = sorted([p for p in cif_out.glob("*.cif") if p.is_file()])
    ids: List[str] = [p.stem for p in cif_paths]
    saved: List[str] = []

    for mpid, path in zip(ids, cif_paths):
        try:
            s = _read_cif_as_structure(
                path, force_primitive=True, symprec=1e-2, angle_tol=5.0
            )
            atoms, cell = _structure_to_atoms_and_cell(s)
            grid, _ = _build_unimat_grid(
                atoms, max_atoms_per_element=max_atoms_per_element
            )
            cell_vec = np.array(
                [
                    cell["a"],
                    cell["b"],
                    cell["c"],
                    cell["alpha"],
                    cell["beta"],
                    cell["gamma"],
                ],
                dtype=np.float32,
            )

            ctx = np.full((len(label_vocab), 3), np.nan, dtype=np.float32)
            if mpid in by_mpid:
                sub = by_mpid[mpid][["label", "ev"]].dropna(subset=["label"])
                sub = sub.groupby("label", as_index=False).first()
                val_map = {
                    row["label"]: _parse_ev_triplet(row["ev"])
                    for _, row in sub.iterrows()
                }
                for i_lab, lab in enumerate(label_vocab):
                    if lab in val_map:
                        ctx[i_lab, :] = val_map[lab]

            np.savez_compressed(
                samp_dir / f"{mpid}.npz", grid=grid, cell_params=cell_vec, context=ctx
            )

            formula = s.composition.reduced_formula
            density = s.density
            sg = SpacegroupAnalyzer(
                s, symprec=1e-2, angle_tolerance=5.0
            ).get_space_group_symbol()
            meta_text = f"material_id={mpid}\nformula={formula}\ndensity={density:.6f}\nspacegroup={sg}"
            (meta_dir / f"{mpid}.txt").write_text(meta_text, encoding="utf-8")

            saved.append(mpid)
        except Exception as e:
            print(f"[WARN] BUILD {mpid}: {e}")

    (out_root / "index.json").write_text(
        json.dumps(
            {
                "ids": saved,
                "max_atoms_per_element": max_atoms_per_element,
                "context": {
                    "type": "band_labels_triplet",
                    "per_label": 3,
                    "labels": label_vocab,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    whole.to_csv(out_root / "data.csv", index=False)
    print(f"Zapisano {len(saved)} próbek do {out_root}")

    print("3) Smoke test Dataset/DataLoader...")
    ds = UniMatTorchDataset(str(out_root))
    if len(ds) == 0:
        print("[WARN] Brak próbek w datasetcie.")
    else:
        sample = ds[0]
        print("ID:", sample["id"])
        print(
            "grid:",
            tuple(sample["grid"].shape),
            "cell:",
            tuple(sample["cell_params"].shape),
            "context:",
            tuple(sample["context"].shape),
        )
        dl = DataLoader(
            ds, batch_size=2, shuffle=True, num_workers=0, collate_fn=_collate_fixed
        )
        batch = next(iter(dl))
        print(
            "Batch shapes:",
            batch["grid"].shape,
            batch["cell_params"].shape,
            batch["context"].shape,
        )

    print("Gotowe w", datetime.now() - start)


if __name__ == "__main__":
    all_in_one()
