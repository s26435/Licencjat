from __future__ import annotations

import json
import os

from mp_api.client import MPRester

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import enum

import numpy as np
import torch
from torch.utils.data import Dataset

_ELEM_TABLE: List[Tuple[str, int, int]] = [
    ("H",1,1), ("He",1,18),
    ("Li",2,1), ("Be",2,2), ("B",2,13), ("C",2,14), ("N",2,15), ("O",2,16), ("F",2,17), ("Ne",2,18),
    ("Na",3,1), ("Mg",3,2), ("Al",3,13), ("Si",3,14), ("P",3,15), ("S",3,16), ("Cl",3,17), ("Ar",3,18),
    ("K",4,1), ("Ca",4,2), ("Sc",4,3), ("Ti",4,4), ("V",4,5), ("Cr",4,6), ("Mn",4,7), ("Fe",4,8),
    ("Co",4,9), ("Ni",4,10), ("Cu",4,11), ("Zn",4,12), ("Ga",4,13), ("Ge",4,14), ("As",4,15), ("Se",4,16),
    ("Br",4,17), ("Kr",4,18),
    ("Rb",5,1), ("Sr",5,2), ("Y",5,3), ("Zr",5,4), ("Nb",5,5), ("Mo",5,6), ("Tc",5,7), ("Ru",5,8),
    ("Rh",5,9), ("Pd",5,10), ("Ag",5,11), ("Cd",5,12), ("In",5,13), ("Sn",5,14), ("Sb",5,15), ("Te",5,16),
    ("I",5,17), ("Xe",5,18),
    ("Cs",6,1), ("Ba",6,2),
    ("La",6,3), ("Ce",6,3), ("Pr",6,3), ("Nd",6,3), ("Pm",6,3), ("Sm",6,3), ("Eu",6,3), ("Gd",6,3),
    ("Tb",6,3), ("Dy",6,3), ("Ho",6,3), ("Er",6,3), ("Tm",6,3), ("Yb",6,3), ("Lu",6,3),
    ("Hf",6,4), ("Ta",6,5), ("W",6,6), ("Re",6,7), ("Os",6,8), ("Ir",6,9), ("Pt",6,10), ("Au",6,11),
    ("Hg",6,12), ("Tl",6,13), ("Pb",6,14), ("Bi",6,15), ("Po",6,16), ("At",6,17), ("Rn",6,18),
    ("Fr",7,1), ("Ra",7,2),
    ("Ac",7,3), ("Th",7,3), ("Pa",7,3), ("U",7,3), ("Np",7,3), ("Pu",7,3), ("Am",7,3), ("Cm",7,3),
    ("Bk",7,3), ("Cf",7,3), ("Es",7,3), ("Fm",7,3), ("Md",7,3), ("No",7,3), ("Lr",7,3),
    ("Rf",7,4), ("Db",7,5), ("Sg",7,6), ("Bh",7,7), ("Hs",7,8), ("Mt",7,9), ("Ds",7,10), ("Rg",7,11),
    ("Cn",7,12), ("Nh",7,13), ("Fl",7,14), ("Mc",7,15), ("Lv",7,16), ("Ts",7,17), ("Og",7,18),
]
SYM_TO_PERIOD_GROUP: Dict[str, Tuple[int, int]] = {s: (p, g) for s, p, g in _ELEM_TABLE}
MAX_PERIODS = 9 
MAX_GROUPS  = 18

def _ensure_dir(path: os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def _default_fields() -> List[str]:
    return [
        "material_id", "formula_pretty", "structure", "density",
        "energy_above_hull",
        "formation_energy_per_atom", "band_gap",
        "is_magnetic", "total_magnetization", "task_ids", "nsites",
        "symmetry", "deprecated", "warnings",
    ]

def _pydantic_to_dict(obj):
    if hasattr(obj, "model_dump"):  # pydantic v2
        return obj.model_dump()
    if hasattr(obj, "dict"):        # pydantic v1
        return obj.dict()
    return obj

def _json_sanitize(obj):
    # Pymatgen Structure / Site itp.
    if hasattr(obj, "as_dict"):
        try:
            return _json_sanitize(obj.as_dict())
        except Exception:
            pass
    # Pydantic BaseModel
    if hasattr(obj, "model_dump") or hasattr(obj, "dict"):
        return _json_sanitize(_pydantic_to_dict(obj))
    # Enum
    if isinstance(obj, enum.Enum):
        return _json_sanitize(obj.value)
    # numpy scalary/arraye
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # kolekcje
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_sanitize(v) for v in obj]
    # Path
    if isinstance(obj, Path):
        return str(obj)
    return obj

def _translate_query_for_mpr(query: Optional[dict]) -> dict:
    q = query or {}
    out = {}
    if "elements" in q:
        el = q["elements"]
        if isinstance(el, dict) and "$in" in el:
            out["elements"] = list(el["$in"])
        elif isinstance(el, (list, tuple, set)):
            out["elements"] = list(el)
    for k in ("material_ids", "is_stable", "exclude_elements", "nelements", "num_elements"):
        if k in q:
            out[k] = q[k]
    return out



def download_crystals_to_disk(
    out_dir: str,
    api_key: str,
    query: Optional[dict] = None,
    fields: Optional[List[str]] = None,
    limit: int = 500,
    batch_size: int = 100,  # ignorowane przez mp-api, ale zostawiamy dla kompatybilności
) -> List[str]:
    """
    Wersja korzystająca z mp-api (MPRester). Pobiera SummaryDoc z 'structure'
    i zapisuje:
      - raw/<mpid>.json  (słownik + structure jako MSON: as_dict())
      - meta/<mpid>.txt  (k=v)
    """
    out = _ensure_dir(out_dir)
    raw_dir = _ensure_dir(out / "raw")
    meta_dir = _ensure_dir(out / "meta")

    fields = fields or _default_fields()
    if "structure" not in fields:
        fields = list(fields) + ["structure"]

    saved_ids: List[str] = []
    q_args = _translate_query_for_mpr(query)

    with MPRester(api_key) as mpr:
        docs_iter = mpr.materials.summary.search(fields=fields, **q_args)
        for doc in docs_iter:
            if len(saved_ids) >= limit:
                break

            d = _pydantic_to_dict(doc)
            mpid = str(d.get("material_id"))
            if not mpid:
                continue

            if hasattr(doc, "structure") and doc.structure is not None:
                d["structure"] = doc.structure.as_dict()

            d_clean = _json_sanitize(d)

            with open(raw_dir / f"{mpid}.json", "w", encoding="utf-8") as f:
                json.dump(d_clean, f, ensure_ascii=False, indent=2)

            meta_lines = []
            for k, v in d_clean.items():
                if k == "structure":
                    continue
                if isinstance(v, (dict, list)):
                    v = json.dumps(v, ensure_ascii=False)
                meta_lines.append(f"{k}={v}")
            with open(meta_dir / f"{mpid}.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(meta_lines))

            saved_ids.append(mpid)

    with open(out / "index.json", "w", encoding="utf-8") as f:
        json.dump({"ids": saved_ids}, f, indent=2)
    return saved_ids

@dataclass
class UniMatSample:
    material_id: str
    grid: np.ndarray 
    cell_params: np.ndarray
    meta_path: Path
    extras: Dict[str, str]

def _extract_structure_from_doc(doc: dict) -> Tuple[List[Tuple[str, List[float]]], Dict[str, float]]:
    s = doc["structure"]
    lat = s["lattice"]
    cell_params = {
        "a": float(lat["a"]),
        "b": float(lat["b"]),
        "c": float(lat["c"]),
        "alpha": float(lat["alpha"]),
        "beta": float(lat["beta"]),
        "gamma": float(lat["gamma"]),
    }
    atoms: List[Tuple[str, List[float]]] = []
    for site in s["sites"]:
        sp = site["species"][0]["element"]
        if "abc" in site:
            frac = list(map(float, site["abc"]))
        elif "frac_coords" in site:
            frac = list(map(float, site["frac_coords"]))
        else:
            frac = list(map(float, site["xyz"]))
            raise ValueError("Structure site lacks fractional coords; please request 'structure' with fractional coords.")
        atoms.append((sp, frac))
    return atoms, cell_params

def _build_unimat_grid(
    atoms: List[Tuple[str, List[float]]],
    max_atoms_per_element: int = 8,
    pad_value: float = np.nan,
) -> Tuple[np.ndarray, Dict[Tuple[int, int], int]]:
    grid = np.full((MAX_PERIODS, MAX_GROUPS, max_atoms_per_element, 3), pad_value, dtype=np.float32)
    counts: Dict[Tuple[int, int], int] = {}
    bucketed: Dict[Tuple[int, int], List[List[float]]] = {}

    for sym, frac in atoms:
        if sym not in SYM_TO_PERIOD_GROUP:
            continue
        p, g = SYM_TO_PERIOD_GROUP[sym]
        key = (p, g)
        bucketed.setdefault(key, []).append(frac)

    for key, lst in bucketed.items():
        chosen = lst[:max_atoms_per_element]
        counts[key] = len(chosen)
        p, g = key
        for i, xyz in enumerate(chosen):
            grid[p-1, g-1, i, :] = np.asarray(xyz, dtype=np.float32)

    return grid, counts

def convert_raw_to_unimat_dataset(
    raw_root: str,
    out_root: str,
    max_atoms_per_element: int = 8,
) -> List[str]:
    raw_root = str(raw_root)
    out = _ensure_dir(out_root)
    samp_dir = _ensure_dir(Path(out) / "samples")
    out_meta = _ensure_dir(Path(out) / "meta")

    index_path = Path(raw_root) / "index.json"
    if not Path(index_path).exists():
        raise FileNotFoundError(f"Brakuje {index_path}; najpierw wywołaj download_crystals_to_disk()")

    with open(index_path, "r", encoding="utf-8") as f:
        ids = json.load(f)["ids"]

    saved: List[str] = []
    for mpid in ids:
        raw_doc = json.load(open(Path(raw_root) / "raw" / f"{mpid}.json", "r", encoding="utf-8"))
        atoms, cell = _extract_structure_from_doc(raw_doc)
        grid, counts = _build_unimat_grid(atoms, max_atoms_per_element=max_atoms_per_element)

        cell_vec = np.array([cell["a"], cell["b"], cell["c"], cell["alpha"], cell["beta"], cell["gamma"]], dtype=np.float32)
        np.savez_compressed(samp_dir / f"{mpid}.npz", grid=grid, cell_params=cell_vec)
        in_meta = Path(raw_root) / "meta" / f"{mpid}.txt"
        if in_meta.exists():
            txt = open(in_meta, "r", encoding="utf-8").read()
        else:
            useful = {k: raw_doc.get(k) for k in ("material_id", "formula_pretty", "density", "band_gap", "e_above_hull")}
            txt = "\n".join(f"{k}={json.dumps(v, ensure_ascii=False) if isinstance(v,(dict,list)) else v}" for k,v in useful.items())
        with open(out_meta / f"{mpid}.txt", "w", encoding="utf-8") as f:
            f.write(txt)

        saved.append(mpid)

    with open(Path(out) / "index.json", "w", encoding="utf-8") as f:
        json.dump({"ids": saved, "max_atoms_per_element": max_atoms_per_element}, f, indent=2)
    return saved

class UniMatTorchDataset(Dataset):
    def __init__(self, dataset_root: str, ids: Optional[List[str]] = None):
        self.root = Path(dataset_root)
        idx = json.load(open(self.root / "index.json", "r", encoding="utf-8"))
        self.ids = ids or idx["ids"]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i: int):
        mpid = self.ids[i]
        npz_path = self.root / "samples" / f"{mpid}.npz"
        meta_path = self.root / "meta" / f"{mpid}.txt"

        with np.load(npz_path) as data:
            grid = torch.from_numpy(data["grid"]).float()          # [P,G,K,3]
            cell = torch.from_numpy(data["cell_params"]).float()   # [6]

        meta_text = ""
        if meta_path.exists():
            meta_text = meta_path.read_text(encoding="utf-8")

        return {"id": mpid, "grid": grid, "cell_params": cell, "meta_text": meta_text}


def build_and_save_unimat_dataset(
    api_key: str,
    tmp_raw_dir: str,
    final_dataset_dir: str,
    query: Optional[dict] = None,
    limit: int = 500,
    batch_size: int = 100,
    max_atoms_per_element: int = 8,
    fields: Optional[List[str]] = None,
) -> List[str]:
    ids = download_crystals_to_disk(
        out_dir=tmp_raw_dir,
        api_key=api_key,
        query=query,
        fields=fields,
        limit=limit,
        batch_size=batch_size,
    )
    convert_raw_to_unimat_dataset(
        raw_root=tmp_raw_dir,
        out_root=final_dataset_dir,
        max_atoms_per_element=max_atoms_per_element,
    )
    return ids

def load_unimat_torch_dataset(dataset_dir: str, ids: Optional[List[str]] = None) -> Dataset:
    return UniMatTorchDataset(dataset_dir, ids=ids)

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    API_KEY = os.getenv("API_KEY")

    ids = build_and_save_unimat_dataset(
        api_key=API_KEY,
        tmp_raw_dir="./mp_raw",
        final_dataset_dir="./mp_unimat_ds",
        query={"elements": ["Na", "Cl"]},
        limit=200,
        max_atoms_per_element=8,
    )
    ds = load_unimat_torch_dataset("./mp_unimat_ds")
    sample = ds[0]
    print(sample["id"], sample["grid"].shape, sample["cell_params"])

