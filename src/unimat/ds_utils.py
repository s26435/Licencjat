import os
from pathlib import Path
import re
from typing import Optional, List, Any
import ast
import math

import numpy as np
import pandas as pd

import mp_api
from mp_api.client import MPRester
from pymatgen.electronic_structure.plotter import BSPlotter
from emmet.core.electronic_structure import BSPathType
from pymatgen.core import Structure


def _ensure_dir(path: os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def normalize_label_file_only(s: str) -> str:
    s = str(s).strip()
    s = s.replace("$", "")
    s = s.replace(r"\Gamma", "gamma")
    s = s.replace(r"\Sigma_1", "sigma_1")
    s = s.replace(r"\Sigma", "sigma")
    s = s.lower()
    s = re.sub(r"[^a-z0-9_]+", "", s)
    return s


def interp_at_x(x0, x_list, y_list) -> Optional[float]:
    i = np.searchsorted(x_list, x0) - 1
    if i < 0 or i >= len(x_list) - 1:
        return None
    x1, x2 = x_list[i], x_list[i + 1]
    if not (x1 <= x0 <= x2):
        return None
    t = (x0 - x1) / (x2 - x1) if x2 != x1 else 0.0
    return (1 - t) * y_list[i] + t * y_list[i + 1]


_FLOAT_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def _parse_ev_triplet(ev_cell) -> np.ndarray:
    """Z kolumny 'ev' wyciąga [p1,p2,n1] -> float32. Braki = NaN."""
    if ev_cell is None or (isinstance(ev_cell, float) and math.isnan(ev_cell)):
        return np.array([np.nan, np.nan, np.nan], dtype=np.float32)

    if isinstance(ev_cell, (list, tuple, np.ndarray)):
        arr = np.array(ev_cell, dtype=float).reshape(-1)
    elif isinstance(ev_cell, str):
        try:
            obj = ast.literal_eval(ev_cell)
            if isinstance(obj, (list, tuple, np.ndarray)):
                arr = np.array(obj, dtype=float).reshape(-1)
            else:
                nums = [float(x) for x in _FLOAT_RE.findall(ev_cell)]
                arr = np.array(nums, dtype=float).reshape(-1)
        except Exception:
            nums = [float(x) for x in _FLOAT_RE.findall(ev_cell)]
            arr = np.array(nums, dtype=float).reshape(-1)
    else:
        try:
            arr = np.array(ev_cell, dtype=float).reshape(-1)
        except Exception:
            return np.array([np.nan, np.nan, np.nan], dtype=np.float32)

    out = np.full(3, np.nan, dtype=np.float32)
    out[: min(3, arr.size)] = arr[: min(3, arr.size)]
    return out


def _build_label_vocab(df: pd.DataFrame) -> List[str]:
    return sorted(df["label"].dropna().unique().tolist())


def _get_bs_rows(mpr: MPRester, material_id: str) -> pd.DataFrame:
    bs = None
    try:
        bs = mpr.get_bandstructure_by_material_id(
            material_id=material_id,
            line_mode=True,
            path_type=BSPathType.setyawan_curtarolo,
        )
    except mp_api.client.core.client.MPRestError as e:
        print(str(e))
        bs = None
    except Exception as e:
        print(f"\x1b[31m{str(e)}\x1b[0m")
        bs = None

    if bs is None:
        return pd.DataFrame({"material_id": [], "label": [], "ev": []})

    plotter = BSPlotter(bs)
    ticks = plotter.get_ticks()
    tick_x = list(map(float, ticks["distance"]))
    tick_labels_raw = list(map(str, ticks["label"]))

    tick_pairs = []
    for x, lab in zip(tick_x, tick_labels_raw):
        for sub in lab.split("$\\mid$"):
            tick_pairs.append((x, sub))

    data = plotter.bs_plot_data()
    energy_dict = data["energy"]
    spin_keys = list(energy_dict.keys())
    dist_branches = data["distances"]

    rows = []
    tol = 1e-10
    for x0, lab in tick_pairs:
        for b_idx, x_branch in enumerate(dist_branches):
            xb = np.array(x_branch, dtype=float)
            if x0 < xb.min() - tol or x0 > xb.max() + tol:
                continue
            for spin in spin_keys:
                Eb = np.array(energy_dict[spin][b_idx], dtype=float)
                for band_idx in range(Eb.shape[0]):
                    yb = Eb[band_idx, :]
                    yx = interp_at_x(x0, xb, yb)
                    if yx is not None:
                        rows.append(
                            {"material_id": material_id, "label": lab, "energy": yx}
                        )

    if not rows:
        return pd.DataFrame({"material_id": [], "label": [], "ev": []})

    df_cross = (
        pd.DataFrame(rows).sort_values(["label", "energy"]).reset_index(drop=True)
    )

    def _pick_rows_s_l(g: pd.Series) -> np.ndarray:
        pos = np.sort(g[g > 0].to_numpy())[:2]
        neg = g[g < 0]
        neg = np.array([neg.max()]) if not neg.empty else np.array([np.nan])
        return np.concatenate([pos, neg]).astype(np.float32)

    out = (
        df_cross.groupby("label", group_keys=True)["energy"]
        .apply(_pick_rows_s_l)
        .rename("ev")
        .reset_index()
    )
    out["material_id"] = material_id
    out = out[["material_id", "label", "ev"]]
    return out


def _ensure_structure(obj: Any) -> Structure:
    if isinstance(obj, Structure):
        return obj
    if isinstance(obj, dict):
        if "structure" in obj and isinstance(obj["structure"], (dict, Structure)):
            inner = obj["structure"]
            return inner if isinstance(inner, Structure) else Structure.from_dict(inner)
        return Structure.from_dict(obj)
    raise TypeError(f"Nie potrafię skonwertować typu {type(obj)} do Structure")
