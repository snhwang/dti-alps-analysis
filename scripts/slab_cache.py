"""
Cache both regions the refined index needs: the measurement sphere and the
direction slab.

Everything the rotation experiments reported was computed with directions
estimated from the 5 mm sphere itself. Those spheres sit 12 mm apart, so each
contains tissue belonging to the other tract and the two direction estimates are
pulled together. The method now estimates directions from the tract label
restricted to an 8 mm axial band at the ALPS level, while measuring in the
sphere, so the rotation results have to be recomputed on that basis.

Stores, per session and hemisphere:
  sph_{proj,assoc}   the 5 mm sphere, used for measurement
  slab_{proj,assoc}  the tract label in an 8 mm band, used for the directions

Rotating the head rotates both regions together, so a rotation applied to the
cached arrays reproduces what a differently oriented acquisition would give,
to the extent a coordinate rotation can.

Usage:
    python slab_cache.py --limit 400
"""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from data_paths import winpath

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from alps_common import parse_age

HERE = Path(__file__).resolve().parent
DIFF = HERE.parent.parent / "diffusion"
OUT = winpath("Q:/dti_output")
CACHE = HERE / "slab_cache_b1500"
SLAB_MM = 8.0
FA_MIN = 0.2
SHELL = "_b1500"


def main() -> None:
    import nibabel as nib

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=400)
    args = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)

    src = pd.read_csv(DIFF / "HCP" / "hcpa_alps_spheres_5mm.csv")
    mot = pd.read_csv(DIFF / "HCP" / "hcpa_motion.csv")
    src = src.merge(mot[["Subject_ID", "Visit", "Eddy_Mean_RMS"]],
                    on=["Subject_ID", "Visit"], how="left")
    rms = pd.to_numeric(src.Eddy_Mean_RMS, errors="coerce")
    thr = float(np.nanpercentile(rms.dropna(), 76.4))
    src = src[(src.status == "ok") & (rms <= thr)].copy()
    src["Age"] = parse_age(src["Age"])
    src = src.dropna(subset=["Age"])
    counts = src.Subject_ID.value_counts()
    src = src[src.Subject_ID.isin(counts[counts >= 2].index)]
    src = src.sort_values(["Subject_ID", "Visit"]).head(args.limit)
    print(f"{len(src)} sessions, {src.Subject_ID.nunique()} participants")

    done = skipped = 0
    for i, r in enumerate(src.itertuples(), 1):
        out = CACHE / f"{r.Subject_ID}_{r.Visit}.npz"
        if out.exists():
            done += 1
            continue
        sd = OUT / r.DTI_Session_ID / "processed"
        paths = {"lab": sd / "atlas" / "jhu_labels_registered.nii.gz",
                 "sph": sd / "atlas" / "sphere_roi" / "sphere_roi_combined.nii.gz",
                 "ev": sd / f"tensor_eigenvalues{SHELL}.nii.gz",
                 "vc": sd / f"tensor_eigenvectors{SHELL}.nii.gz"}
        if not all(p.exists() for p in paths.values()):
            skipped += 1
            continue
        try:
            limg = nib.load(str(paths["lab"])); lab = limg.get_fdata().astype(int)
            sph = nib.load(str(paths["sph"])).get_fdata().astype(int)
            ev = nib.load(str(paths["ev"])).get_fdata()
            vc = nib.load(str(paths["vc"])).get_fdata()
        except Exception:
            skipped += 1
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
        band = np.abs(zw - z0) <= SLAB_MM

        blocks = {}
        ok = True
        for hemi, side, scr, slf in (("L", xc < 0, 26, 42), ("R", xc > 0, 25, 41)):
            for tag, mp, ma in (
                ("sph", (sph == 1) & side & (fa >= FA_MIN),
                        (sph == 2) & side & (fa >= FA_MIN)),
                ("slab", (lab == scr) & band & (fa >= FA_MIN),
                         (lab == slf) & band & (fa >= FA_MIN)),
            ):
                for nm, m in (("proj", mp), ("assoc", ma)):
                    if m.sum() < (4 if tag == "sph" else 20):
                        ok = False
                        break
                    v1 = vc[m][:, :, 0]
                    n = np.linalg.norm(v1, axis=1, keepdims=True); n[n == 0] = 1
                    key = f"{tag}_{nm}_{hemi}"
                    blocks[f"{key}_v1"] = (v1 / n).astype(np.float32)
                    blocks[f"{key}_fa"] = fa[m].astype(np.float32)
                    blocks[f"{key}_evals"] = ev[m].astype(np.float32)
                    if tag == "sph":       # eigenvectors only needed where we measure
                        blocks[f"{key}_evecs"] = vc[m].astype(np.float32)
                if not ok:
                    break
            if not ok:
                break
        if not ok:
            skipped += 1
            continue
        blocks["Age"] = np.array([r.Age], dtype=np.float32)
        np.savez_compressed(out, **blocks)
        done += 1
        if i % 50 == 0:
            print(f"  {i}/{len(src)} cached={done} skipped={skipped}", flush=True)

    print(f"cached {done}, skipped {skipped} -> {CACHE}")


if __name__ == "__main__":
    main()
