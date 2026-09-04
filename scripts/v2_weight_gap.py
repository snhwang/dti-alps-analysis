"""Does the mathematically correct v2 weight move the measured perivascular axis?

Methods weights voxels by directional reliability. For v1 that is Westin
CL = (l1 - l2) / l1, because eigenvector displacement under noise scales with
the inverse gap to the neighbouring eigenvalue, and v1's only relevant
neighbour is l2.

The manuscript and measured_pvs_axis.py then weight v2 by CP = (l2 - l3) / l1
"by the same reasoning". The same reasoning does not give that. v2 has two
neighbours, l1 and l3, so its stability is set by the SMALLER of the two gaps:

    v1:  gap = l1 - l2                       -> CL
    v2:  gap = min(l1 - l2, l2 - l3)         -> min(CL, CP)

CP alone is blind to a closing l1-l2 gap, which frees v2 inside the l1-l2
eigenplane exactly as it frees v1. That is the same failure the manuscript
correctly charges FA with. Simulated at sigma=0.05 on (1.0, 1.0, 0.1):

    CP = 0.900 while v2 moves 44.5 degrees under noise
    min(CL, CP) = 0.000, which is correct

This script measures whether the distinction matters in the regions the paper
actually pools over, by computing the pooled v2 axis both ways in each session
and reporting the angle between them. If that angle is small the manuscript
needs its reasoning corrected but no number changes. If it is not small the
weight itself has to change.

Rows accumulate to CSV every 25 sessions so an interrupted run keeps its work.

    python v2_weight_gap.py --cohort dlbs
    python v2_weight_gap.py --cohort hcpa --limit 200
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import atomic_io  # noqa: F401  writes become atomic on import
import nibabel as nib
from data_paths import winpath
from direction_estimators import principal

HERE = Path(__file__).resolve().parent
DIFF = HERE.parent.parent / "diffusion"
OUT = winpath("Q:/dti_output")
SLAB_MM = 8.0
FA_MIN = float(os.environ.get("ALPS_FA_MIN", "0.2"))
SHELL = os.environ.get("ALPS_TENSOR_SUFFIX", "")
FLUSH_EVERY = 25


def angle_between(a, b) -> float:
    """Degrees between two axes, which carry no sign."""
    c = min(1.0, abs(float(np.dot(a, b))))
    return float(np.degrees(np.arccos(c)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=["hcpa", "dlbs"], default="dlbs")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if args.cohort == "hcpa":
        src = pd.read_csv(DIFF / "HCP" / "hcpa_alps_spheres_5mm.csv")
    else:
        src = pd.read_csv(DIFF / "DLBS" / "dlbs_alps_spheres_5mm.csv")
    if args.limit:
        src = src.head(args.limit)

    outp = HERE / f"v2_weight_gap_{args.cohort}{SHELL}.csv"
    print(f"writing to {outp.name}, flushed every {FLUSH_EVERY} sessions\n", flush=True)

    rows: list[dict] = []
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

        srt = np.argsort(ev, axis=-1)[..., ::-1]
        l1 = np.take_along_axis(ev, srt[..., 0:1], -1)[..., 0]
        l2 = np.take_along_axis(ev, srt[..., 1:2], -1)[..., 0]
        l3 = np.take_along_axis(ev, srt[..., 2:3], -1)[..., 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            CP = np.where(l1 > 0, (l2 - l3) / l1, 0.0)
            CL = np.where(l1 > 0, (l1 - l2) / l1, 0.0)
        GAP = np.minimum(CL, CP)            # the correct v2 reliability weight

        md = ev.mean(-1)
        nu = np.sqrt(((ev - md[..., None]) ** 2).sum(-1))
        de = np.sqrt((ev ** 2).sum(-1))
        fa = np.clip(np.sqrt(1.5) * np.divide(nu, de, out=np.zeros_like(nu),
                                              where=de != 0), 0, 1)

        ii, jj, kk = np.indices(lab.shape)
        Af = limg.affine
        xw = Af[0, 0] * ii + Af[0, 1] * jj + Af[0, 2] * kk + Af[0, 3]
        zw = Af[2, 0] * ii + Af[2, 1] * jj + Af[2, 2] * kk + Af[2, 3]

        def evec2(m):
            V = vc[m]; o = srt[m]
            v = np.take_along_axis(V, o[:, None, 1:2], 2)[:, :, 0]
            return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)

        def axes(masks):
            """Pooled v2 under the published weight and under the correct one."""
            v2 = np.vstack([evec2(m) for m in masks])
            w_cp = np.concatenate([CP[m] for m in masks])
            w_gap = np.concatenate([GAP[m] for m in masks])
            ok = w_cp > 0
            if ok.sum() < 6:
                return None
            a_cp = principal(v2[ok], w_cp[ok])
            a_gap = principal(v2[ok], w_gap[ok]) if (w_gap[ok] > 0).sum() >= 6 else None
            if a_gap is None:
                return None
            # Share of the published weight sitting on voxels the correct weight
            # marks unreliable, which is where the two can disagree at all.
            binding = float(w_cp[ok][CL_c := (w_gap[ok] < w_cp[ok])].sum() / w_cp[ok].sum())
            return (a_cp / np.linalg.norm(a_cp), a_gap / np.linalg.norm(a_gap),
                    binding, float(np.median(w_gap[ok] / np.maximum(w_cp[ok], 1e-12))))

        rec: dict = {"Subject_ID": getattr(r, "Subject_ID", ""),
                     "Visit": getattr(r, "Visit", ""),
                     "session": r.DTI_Session_ID}
        got = False
        for hemi, side, scr, slf in (("L", xw < 0, 26, 42), ("R", xw > 0, 25, 41)):
            mp_s = ((sph == 1) & side) & (fa >= FA_MIN)
            ma_s = ((sph == 2) & side) & (fa >= FA_MIN)
            if mp_s.sum() < 4 or ma_s.sum() < 4:
                continue
            z0 = float(np.median(zw[sph > 0])) if (sph > 0).any() else 0.0
            band = np.abs(zw - z0) <= SLAB_MM
            mp_l = (lab == scr) & (fa >= FA_MIN) & band
            ma_l = (lab == slf) & (fa >= FA_MIN) & band
            if mp_l.sum() < 10 or ma_l.sum() < 10:
                continue
            for tag, masks in (("sphere", [mp_s, ma_s]), ("slab", [mp_l, ma_l])):
                got_axes = axes(masks)
                if got_axes is None:
                    continue
                a_cp, a_gap, binding, ratio = got_axes
                rec[f"{tag}_{hemi}_deg"] = angle_between(a_cp, a_gap)
                rec[f"{tag}_{hemi}_binding"] = binding
                rec[f"{tag}_{hemi}_wratio"] = ratio
                got = True
        if got:
            rows.append(rec)

        if rows and len(rows) % FLUSH_EVERY == 0:
            pd.DataFrame(rows).to_csv(outp, index=False)
            print(f"  {i} scanned, {len(rows)} kept", flush=True)

    if not rows:
        print("no sessions produced an axis")
        return
    df = pd.DataFrame(rows)
    df.to_csv(outp, index=False)

    print(f"\n  {len(df)} sessions\n")
    print(f"  {'pooling':>10s} {'median':>9s} {'p95':>9s} {'max':>9s} "
          f"{'weight on l1~l2':>16s}")
    for tag in ("sphere", "slab"):
        deg = pd.concat([df.get(f"{tag}_{h}_deg") for h in "LR"]).dropna()
        bind = pd.concat([df.get(f"{tag}_{h}_binding") for h in "LR"]).dropna()
        if deg.empty:
            continue
        print(f"  {tag:>10s} {deg.median():>8.2f}° {deg.quantile(0.95):>8.2f}° "
              f"{deg.max():>8.2f}° {100 * bind.median():>15.1f}%")
    print(f"\n  written to {outp.name}")


if __name__ == "__main__":
    main()
