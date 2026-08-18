"""
Choose the slab: how thick, and should off-direction voxels be pruned?

The conventional 5 mm ALPS spheres sit only 12 mm apart, so each contains
transition-zone tissue belonging to the other tract. That pulls the two
estimated tract directions toward each other and makes them appear 67 degrees
apart when the anatomical labels put them near 84. Estimating the directions
from a larger region fixes this, but two choices then have to be made on
evidence rather than convenience.

  band     The whole JHU tract label restricted to an axial band about the ALPS
           level. Wider bands supply more voxels but average along a curving
           tract, which drags the estimate away from the local direction.

  prune    Voxels far from the region's own mean direction are tract edge,
           partial volume or crossing fibre. Discarding them and re-estimating
           should sharpen the direction, at the cost of voxels. Pruning is
           relative to the region's own mean, so it assumes nothing about which
           way the tract runs.

Measurement always stays in the conventional 5 mm sphere, so the reported
quantity is unchanged and remains comparable with prior work. Only the axes
move. The criteria are the inter-fibre angle, which should approach 90 degrees
as contamination is removed, and within-participant reliability.

Usage:
    python slab_tuning.py --limit 160
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from data_paths import winpath

import atomic_io  # noqa: F401  writes become atomic on import

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from estimator_variants import directional_diffusivity, variance_components
from direction_estimators import weights_for, principal, align, X, Y, Z
from alps_common import parse_age

HERE = Path(__file__).resolve().parent
DIFF = HERE.parent.parent / "diffusion"
OUT = winpath("Q:/dti_output")

BANDS = [6, 8, 10, 12, 16]
PRUNES = [None, 45, 30, 20]
FA_MIN = 0.2


def direction(v1, w, prune_deg):
    """Dyadic principal direction, optionally re-estimated after pruning."""
    d = principal(v1, w)
    if prune_deg is not None:
        keep = np.abs(v1 @ d) >= np.cos(np.radians(prune_deg))
        if keep.sum() >= 20:
            d = principal(v1[keep], w[keep])
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=160)
    args = ap.parse_args()

    import nibabel as nib

    src = pd.read_csv(DIFF / "HCP" / "hcpa_alps_spheres_5mm.csv")
    src = src[src.status == "ok"].copy()
    src["Age"] = parse_age(src["Age"])
    counts = src.Subject_ID.value_counts()
    src = src[src.Subject_ID.isin(counts[counts >= 2].index)]
    src = src.sort_values(["Subject_ID", "Visit"]).head(args.limit)
    print(f"{len(src)} sessions, {src.Subject_ID.nunique()} participants\n")

    rows = []
    for i, r in enumerate(src.itertuples(), 1):
        sd = OUT / r.DTI_Session_ID / "processed"
        lab_p = sd / "atlas" / "jhu_labels_registered.nii.gz"
        sph_p = sd / "atlas" / "sphere_roi" / "sphere_roi_combined.nii.gz"
        ev_p = sd / "tensor_eigenvalues_b1500.nii.gz"
        vc_p = sd / "tensor_eigenvectors_b1500.nii.gz"
        if not all(p.exists() for p in (lab_p, sph_p, ev_p, vc_p)):
            continue
        try:
            limg = nib.load(str(lab_p)); lab = limg.get_fdata().astype(int)
            sph = nib.load(str(sph_p)).get_fdata().astype(int)
            ev = nib.load(str(ev_p)).get_fdata()
            vc = nib.load(str(vc_p)).get_fdata()
        except Exception:
            continue

        md = ev.mean(-1)
        nu = np.sqrt(((ev - md[..., None]) ** 2).sum(-1))
        de = np.sqrt((ev ** 2).sum(-1))
        fa = np.clip(np.sqrt(1.5) * np.divide(nu, de, out=np.zeros_like(nu),
                                              where=de != 0), 0, 1)
        ii, jj, kk = np.indices(lab.shape)
        A = limg.affine
        zw = A[2, 0] * ii + A[2, 1] * jj + A[2, 2] * kk + A[2, 3]
        # Hemisphere is taken from the world x coordinate, not the voxel index.
        # These volumes have a negative x scale, so voxel index < mid is world
        # x > 0, which is the RIGHT hemisphere. Pairing it with the _L labels
        # measured one hemisphere while estimating directions from the other.
        xc = (limg.affine[0, 0] * ii + limg.affine[0, 1] * jj
              + limg.affine[0, 2] * kk + limg.affine[0, 3])
        z0 = float(np.median(zw[sph > 0])) if (sph > 0).any() else 0.0

        def pack(mask):
            v = vc[mask][:, :, 0]
            n = np.linalg.norm(v, axis=1, keepdims=True); n[n == 0] = 1
            return {"v1": v / n, "fa": fa[mask], "evals": ev[mask], "evecs": vc[mask]}

        rec = {"Subject_ID": r.Subject_ID, "Visit": r.Visit, "Age": r.Age}
        got = True
        for hemi, side, scr, slf in (("L", xc < 0, 26, 42), ("R", xc > 0, 25, 41)):
            mp_s = (sph == 1) & side & (fa >= FA_MIN)
            ma_s = (sph == 2) & side & (fa >= FA_MIN)
            if mp_s.sum() < 4 or ma_s.sum() < 4:
                got = False
                break
            P, Asp = pack(mp_s), pack(ma_s)
            for band in BANDS:
                inband = np.abs(zw - z0) <= band
                mp = (lab == scr) & inband & (fa >= FA_MIN)
                ma = (lab == slf) & inband & (fa >= FA_MIN)
                if mp.sum() < 30 or ma.sum() < 30:
                    continue
                DP, DA = pack(mp), pack(ma)
                for pr in PRUNES:
                    vp = align(direction(DP["v1"], weights_for("cl", DP), pr), Z)
                    va = align(direction(DA["v1"], weights_for("cl", DA), pr), Y)
                    p = np.cross(vp, va); p /= max(np.linalg.norm(p), 1e-12)
                    op = np.cross(p, vp); op /= max(np.linalg.norm(op), 1e-12)
                    oa = np.cross(p, va); oa /= max(np.linalg.norm(oa), 1e-12)
                    val = ((directional_diffusivity(P["evals"], P["evecs"], p)
                            + directional_diffusivity(Asp["evals"], Asp["evecs"], p))
                           / (directional_diffusivity(P["evals"], P["evecs"], op)
                              + directional_diffusivity(Asp["evals"], Asp["evecs"], oa)))
                    tag = f"{band}_{pr if pr else 0}"
                    rec.setdefault(f"rf_{tag}", []).append(val)
                    rec.setdefault(f"ang_{tag}", []).append(
                        np.degrees(np.arccos(np.clip(abs(vp @ va), 0, 1))))
                    rec.setdefault(f"n_{tag}", []).append(min(mp.sum(), ma.sum()))
        if not got:
            continue
        for k in list(rec):
            if isinstance(rec[k], list):
                rec[k] = float(np.mean(rec[k]))
        rows.append(rec)
        if i % 40 == 0:
            print(f"  {i}/{len(src)}", flush=True)

    d = pd.DataFrame(rows)
    lon = d[d.Subject_ID.isin(d.Subject_ID.value_counts()[lambda s: s >= 2].index)]
    print(f"\nusable {len(d)} sessions, {d.Subject_ID.nunique()} participants\n")

    from scipy import stats
    print(f"{'band':>5s} {'prune':>6s} {'voxels':>7s} {'angle':>7s} {'ICC':>7s} "
          f"{'var_within':>11s} {'r age':>7s}")
    best = []
    for band in BANDS:
        for pr in PRUNES:
            tag = f"{band}_{pr if pr else 0}"
            col = f"rf_{tag}"
            if col not in d.columns or d[col].isna().mean() > 0.2:
                continue
            vc_ = variance_components(lon.dropna(subset=[col]), col)
            rr = stats.linregress(d.dropna(subset=[col])["Age"],
                                  d.dropna(subset=[col])[col])[2]
            print(f"{band:>4d}mm {str(pr) if pr else '-':>6s} {d[f'n_{tag}'].median():7.0f} "
                  f"{d[f'ang_{tag}'].median():6.1f}d {vc_['icc']:7.3f} "
                  f"{vc_['var_within']:11.6f} {rr:7.3f}")
            best.append((band, pr, d[f"ang_{tag}"].median(), vc_["icc"], rr))

    d.to_csv(HERE / "slab_tuning.csv", index=False)
    print(f"\nWrote {HERE/'slab_tuning.csv'}")


if __name__ == "__main__":
    main()
