"""
Should the association ROI be shifted to a part of the tract that looks right?

This is what a human rater does. The atlas puts the sphere where the SLF is on
average, and about a fifth of what lands inside it is left-right oriented. A
rater looking at a colour map would slide the region to somewhere greener.

Two criteria are implemented, and the difference between them is the point.

  tract   maximise agreement between the voxels' principal directions and the
          SUBJECT'S OWN slab-derived SLF direction. Rotation-invariant: rotating
          the head rotates the tract direction and the eigenvectors together, so
          the chosen location does not move.

  colour  maximise agreement with scanner y, which is what "looks green" means
          on a direction-encoded colour map. NOT rotation-invariant: tilt the
          head and the selected location moves, so this criterion imports head
          position into the region definition. Implemented to show that it
          should not be used, not as a candidate for adoption.

The projection region is left alone. It is already clean (median 0% off-tract),
so there is nothing to fix there.

Interpretation of the outcome is set out in advance to avoid reading whichever
result appears as favourable:

  * If the off-tract voxels are noise, shifting should raise reliability and
    leave the age association intact or better.
  * If they are perivascular spaces, which is what the region exists to detect,
    shifting removes signal and the age association should fall.

Usage:
    ALPS_TENSOR_SUFFIX=_b1500 python shifted_roi.py --cohort hcpa
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
from estimator_variants import directional_diffusivity
from direction_estimators import weights_for, principal, align, X, Y, Z
from alps_common import parse_age

HERE = Path(__file__).resolve().parent
DIFF = HERE.parent.parent / "diffusion"
OUT = winpath("Q:/dti_output")
SLAB_MM, FA_MIN, RADIUS = 8.0, 0.2, 5.0
SEARCH_MM, STEP_MM = 6.0, 2.0      # in-plane search half-width and grid step
MIN_VOX = 8
SHELL = os.environ.get("ALPS_TENSOR_SUFFIX", "")


def main() -> None:
    import nibabel as nib

    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=["hcpa", "dlbs"], default="hcpa")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if args.cohort == "hcpa":
        src = pd.read_csv(DIFF / "HCP" / "hcpa_alps_spheres_5mm.csv")
        mot = pd.read_csv(DIFF / "HCP" / "hcpa_motion.csv")
        src = src.merge(mot[["Subject_ID", "Visit", "Eddy_Mean_RMS"]],
                        on=["Subject_ID", "Visit"], how="left")
        rms = pd.to_numeric(src.Eddy_Mean_RMS, errors="coerce")
        thr = float(np.nanpercentile(rms.dropna(), 76.4))
    else:
        src = pd.read_csv(DIFF / "DLBS" / "dlbs_alps_spheres_5mm.csv")
        mot = pd.read_csv(DIFF / "DLBS" / "dlbs_motion.csv")
        src = src.merge(mot[["DTI_Session_ID", "Eddy_Mean_RMS"]],
                        on="DTI_Session_ID", how="left")
        rms = pd.to_numeric(src.Eddy_Mean_RMS, errors="coerce")
        thr = 0.5
        src["Visit"] = src["Session"]

    src = src[(src.status == "ok") & (rms <= thr)].copy()
    src["Age"] = parse_age(src["Age"])
    src = src.dropna(subset=["Age"])
    counts = src.Subject_ID.value_counts()
    src = src[src.Subject_ID.isin(counts[counts >= 2].index)]
    src = src.sort_values(["Subject_ID", "Visit"])
    if args.limit:
        rng = np.random.default_rng(20260809)
        keep = rng.choice(sorted(src.Subject_ID.unique()),
                          size=min(args.limit, src.Subject_ID.nunique()), replace=False)
        src = src[src.Subject_ID.isin(keep)]
    print(f"cohort {args.cohort}: {len(src)} sessions, {src.Subject_ID.nunique()} participants\n")

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
            evals = nib.load(str(sd / f"tensor_eigenvalues{SHELL}.nii.gz")).get_fdata()
            evecs = nib.load(str(sd / f"tensor_eigenvectors{SHELL}.nii.gz")).get_fdata()
        except Exception:
            continue

        md = evals.mean(axis=-1)
        nu = np.sqrt(((evals - md[..., None]) ** 2).sum(axis=-1))
        de = np.sqrt((evals ** 2).sum(axis=-1))
        fa = np.clip(np.sqrt(1.5) * np.divide(nu, de, out=np.zeros_like(nu), where=de != 0), 0, 1)

        ii, jj, kk = np.indices(lab.shape)
        A = limg.affine
        xw = A[0, 0] * ii + A[0, 1] * jj + A[0, 2] * kk + A[0, 3]
        yw = A[1, 0] * ii + A[1, 1] * jj + A[1, 2] * kk + A[1, 3]
        zw = A[2, 0] * ii + A[2, 1] * jj + A[2, 2] * kk + A[2, 3]
        xs = np.arange(lab.shape[0])[:, None, None] * np.ones_like(lab)
        mid = lab.shape[0] // 2

        def pack(mask):
            v1 = evecs[mask][:, :, 0]
            n = np.linalg.norm(v1, axis=1, keepdims=True); n[n == 0] = 1
            return {"v1": v1 / n, "fa": fa[mask], "evals": evals[mask], "evecs": evecs[mask]}

        def alps(P, Ax, un, up, ua):
            return float((directional_diffusivity(P["evals"], P["evecs"], un)
                          + directional_diffusivity(Ax["evals"], Ax["evecs"], un))
                         / (directional_diffusivity(P["evals"], P["evecs"], up)
                            + directional_diffusivity(Ax["evals"], Ax["evecs"], ua)))

        rec = {"Subject_ID": r.Subject_ID, "Visit": r.Visit, "Age": r.Age}
        acc = {}
        # Hemisphere from world x, not voxel index: these volumes have a
        # negative x scale, so index < mid is world x > 0, the RIGHT side.
        for hemi, side, scr, slf in (("L", xw < 0, 26, 42), ("R", xw > 0, 25, 41)):
            mp_s = (sph == 1) & side & (fa >= FA_MIN)
            ma_s = (sph == 2) & side & (fa >= FA_MIN)
            if mp_s.sum() < 4 or ma_s.sum() < 4:
                continue
            P, A0 = pack(mp_s), pack(ma_s)

            z0 = float(np.median(zw[sph > 0])) if (sph > 0).any() else 0.0
            band = np.abs(zw - z0) <= SLAB_MM
            mp_l = (lab == scr) & (fa >= FA_MIN) & band
            ma_l = (lab == slf) & (fa >= FA_MIN) & band
            if mp_l.sum() < 10 or ma_l.sum() < 10:
                continue
            vp = align(principal(pack(mp_l)["v1"], weights_for("cl", pack(mp_l))), Z)
            va = align(principal(pack(ma_l)["v1"], weights_for("cl", pack(ma_l))), Y)

            def axes(v_assoc):
                p = np.cross(vp, v_assoc); p /= max(np.linalg.norm(p), 1e-12)
                op = np.cross(p, vp); op /= max(np.linalg.norm(op), 1e-12)
                oa = np.cross(p, v_assoc); oa /= max(np.linalg.norm(oa), 1e-12)
                return p, op, oa

            p0, op0, oa0 = axes(va)
            acc[f"classic_{hemi}"] = alps(P, A0, X, Y, Z)
            acc[f"refined_{hemi}"] = alps(P, A0, p0, op0, oa0)
            ang0 = np.degrees(np.arccos(np.clip(np.abs(A0["v1"] @ va), 0, 1)))
            acc[f"off0_{hemi}"] = float((ang0 > 45).mean())

            # candidate centres on the same axial level, inside the SLF label
            c0 = np.array([xw[ma_s].mean(), yw[ma_s].mean(), zw[ma_s].mean()])
            cand_mask = (lab == slf) & (fa >= FA_MIN) & (np.abs(zw - c0[2]) <= 2.0)
            if cand_mask.sum() < MIN_VOX:
                continue
            offs = np.arange(-SEARCH_MM, SEARCH_MM + 0.1, STEP_MM)
            best = {}
            for crit, target in (("tract", va), ("colour", Y)):
                bs, bc, bm = -np.inf, c0, None
                for dx in offs:
                    for dy in offs:
                        c = c0 + np.array([dx, dy, 0.0])
                        m = cand_mask & ((xw - c[0]) ** 2 + (yw - c[1]) ** 2
                                         + (zw - c[2]) ** 2 <= RADIUS ** 2)
                        if m.sum() < MIN_VOX:
                            continue
                        v1 = evecs[m][:, :, 0]
                        n = np.linalg.norm(v1, axis=1, keepdims=True); n[n == 0] = 1
                        score = float(np.mean(np.abs((v1 / n) @ target)))
                        if score > bs:
                            bs, bc, bm = score, c, m
                if bm is None:
                    continue
                As = pack(bm)
                angs = np.degrees(np.arccos(np.clip(np.abs(As["v1"] @ va), 0, 1)))
                best[crit] = {
                    "shift": float(np.linalg.norm(bc - c0)),
                    "off": float((angs > 45).mean()),
                    "classic": alps(P, As, X, Y, Z),
                    "refined": alps(P, As, *axes(va)),
                }
            for crit, d in best.items():
                for k, v in d.items():
                    acc[f"{crit}_{k}_{hemi}"] = v

        if not acc:
            continue
        # average hemispheres for every quantity that has both
        keys = {k.rsplit("_", 1)[0] for k in acc}
        for k in keys:
            vals = [acc[f"{k}_{h}"] for h in ("L", "R") if f"{k}_{h}" in acc]
            if vals:
                rec[k] = float(np.mean(vals))
        rows.append(rec)
        if i % 100 == 0:
            print(f"  {i}/{len(src)}", flush=True)

    d = pd.DataFrame(rows)
    out = HERE / f"shifted_roi_{args.cohort}{SHELL}.csv"
    d.to_csv(out, index=False)
    print(f"\n{len(d)} sessions -> {out.name}")


if __name__ == "__main__":
    main()
