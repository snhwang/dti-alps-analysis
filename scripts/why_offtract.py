"""
Is the association ROI contaminated because the data are poor, or because the
anatomy there cannot be represented by one tensor?

Three competing explanations for the ~19% of association-ROI voxels whose
principal direction does not match the tract:

  noise        low SNR or low FA makes the principal eigenvector arbitrary.
               Predicts the off-tract voxels have LOW FA, and predicts the
               problem is much worse in DLBS (1.75x1.75x3.0 mm) than in HCP-A
               (1.5 mm isotropic).

  crossing     the region sits where SLF meets corona radiata and callosal
               fibres, so the voxel contains two or more populations. A single
               tensor then returns a planar rather than linear shape, and its
               principal direction is not any fibre's direction. Predicts HIGH
               planarity (Westin CP) and near-normal FA, and predicts the two
               cohorts look the SAME because it is anatomy, not acquisition.

  misplacement the sphere is simply in the wrong spot. Predicts the off-tract
               voxels sit at the edge of the JHU label rather than its core.

Westin shape measures from the sorted eigenvalues:
  CL = (l1-l2)/l1   linear, one dominant direction
  CP = (l2-l3)/l1   planar, two directions in a plane, the crossing signature
  CS = 3*l3/(l1+l2+l3)  spherical, isotropic

The cohort comparison is the discriminating test, because acquisition differs
sharply between them and anatomy does not.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from data_paths import winpath

import atomic_io  # noqa: F401  writes become atomic on import

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from direction_estimators import weights_for, principal, align, Y, Z
from alps_common import parse_age

HERE = Path(__file__).resolve().parent
DIFF = HERE.parent.parent / "diffusion"
OUT = winpath("Q:/dti_output")
SLAB_MM, FA_MIN, THRESH = 8.0, 0.2, 45.0
SHELL = os.environ.get("ALPS_TENSOR_SUFFIX", "")


def main() -> None:
    import nibabel as nib
    from scipy import ndimage

    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=["hcpa", "dlbs"], default="hcpa")
    ap.add_argument("--limit", type=int, default=60)
    args = ap.parse_args()

    if args.cohort == "hcpa":
        src = pd.read_csv(DIFF / "HCP" / "hcpa_alps_spheres_5mm.csv")
    else:
        src = pd.read_csv(DIFF / "DLBS" / "dlbs_alps_spheres_5mm.csv")
        src["Visit"] = src["Session"]
    src = src[src.status == "ok"].copy()
    src["Age"] = parse_age(src["Age"])
    src = src.dropna(subset=["Age"]).head(args.limit)

    rows = []
    for i, r in enumerate(src.itertuples(), 1):
        sd = OUT / r.DTI_Session_ID / "processed"
        lab_p = sd / "atlas" / "jhu_labels_registered.nii.gz"
        sph_p = sd / "atlas" / "sphere_roi" / "sphere_roi_combined.nii.gz"
        if not (lab_p.exists() and sph_p.exists()):
            continue
        try:
            limg = nib.load(str(lab_p)); lab = limg.get_fdata().astype(int)
            sph = nib.load(str(sph_p)).get_fdata().astype(int)
            ev = nib.load(str(sd / f"tensor_eigenvalues{SHELL}.nii.gz")).get_fdata()
            vc = nib.load(str(sd / f"tensor_eigenvectors{SHELL}.nii.gz")).get_fdata()
        except Exception:
            continue

        srt = np.sort(ev, axis=-1)[..., ::-1]
        l1, l2, l3 = srt[..., 0], srt[..., 1], srt[..., 2]
        tr = srt.sum(-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            CL = np.where(l1 > 0, (l1 - l2) / l1, 0)
            CP = np.where(l1 > 0, (l2 - l3) / l1, 0)
            CS = np.where(tr > 0, 3 * l3 / tr, 0)
        md = ev.mean(-1)
        nu = np.sqrt(((ev - md[..., None]) ** 2).sum(-1))
        de = np.sqrt((ev ** 2).sum(-1))
        fa = np.clip(np.sqrt(1.5) * np.divide(nu, de, out=np.zeros_like(nu), where=de != 0), 0, 1)

        ii, jj, kk = np.indices(lab.shape)
        Af = limg.affine
        xw = Af[0, 0] * ii + Af[0, 1] * jj + Af[0, 2] * kk + Af[0, 3]
        zw = Af[2, 0] * ii + Af[2, 1] * jj + Af[2, 2] * kk + Af[2, 3]

        for hemi, side, slf in (("L", xw < 0, 42), ("R", xw > 0, 41)):
            ma_s = (sph == 2) & side & (fa >= FA_MIN)
            if ma_s.sum() < 6:
                continue
            z0 = float(np.median(zw[sph > 0])) if (sph > 0).any() else 0.0
            ma_l = (lab == slf) & (fa >= FA_MIN) & (np.abs(zw - z0) <= SLAB_MM)
            if ma_l.sum() < 10:
                continue
            v1l = vc[ma_l][:, :, 0]
            n = np.linalg.norm(v1l, axis=1, keepdims=True); n[n == 0] = 1
            va = align(principal(v1l / n, weights_for("cl", {"fa": fa[ma_l], "evals": ev[ma_l]})), Y)

            v1 = vc[ma_s][:, :, 0]
            n = np.linalg.norm(v1, axis=1, keepdims=True); n[n == 0] = 1
            ang = np.degrees(np.arccos(np.clip(np.abs((v1 / n) @ va), 0, 1)))
            off = ang > THRESH
            if off.sum() < 2 or (~off).sum() < 2:
                continue

            # distance from the label edge, to test the misplacement hypothesis
            dist = ndimage.distance_transform_edt(lab == slf,
                                                  sampling=np.abs(np.diag(Af)[:3]))
            rows.append({
                "cohort": args.cohort, "hemi": hemi,
                "frac_off": float(off.mean()),
                "FA_on": float(fa[ma_s][~off].mean()), "FA_off": float(fa[ma_s][off].mean()),
                "CL_on": float(CL[ma_s][~off].mean()), "CL_off": float(CL[ma_s][off].mean()),
                "CP_on": float(CP[ma_s][~off].mean()), "CP_off": float(CP[ma_s][off].mean()),
                "CS_on": float(CS[ma_s][~off].mean()), "CS_off": float(CS[ma_s][off].mean()),
                "edge_on": float(dist[ma_s][~off].mean()), "edge_off": float(dist[ma_s][off].mean()),
            })
        if i % 20 == 0:
            print(f"  {i}/{len(src)}", flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(HERE / f"why_offtract_{args.cohort}{SHELL}.csv", index=False)
    print(f"\n{args.cohort}: {len(d)} hemisphere-sessions\n")
    print(f"{'measure':<26s} {'on-tract':>9s} {'off-tract':>10s} {'ratio':>7s}")
    for nm, a, b in (("FA", "FA_on", "FA_off"),
                     ("CL (linear)", "CL_on", "CL_off"),
                     ("CP (planar, crossing)", "CP_on", "CP_off"),
                     ("CS (spherical)", "CS_on", "CS_off"),
                     ("mm from label edge", "edge_on", "edge_off")):
        print(f"  {nm:<24s} {d[a].mean():9.3f} {d[b].mean():10.3f} {d[b].mean()/max(d[a].mean(),1e-9):7.2f}")
    print(f"\n  median off-tract fraction {d.frac_off.median():.3f}")


if __name__ == "__main__":
    main()
