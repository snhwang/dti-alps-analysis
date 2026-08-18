"""
Estimate the direction from a large ROI, measure the diffusivity in the small one.

The ROI does two different jobs in the refined index. It locates where the
diffusivity is measured, which wants anatomical specificity at the
periventricular level and comparability with the published ALPS literature. And
it supplies the voxels used to estimate the tract direction, whose error falls
as roughly one over the square root of the voxel count. Those pull in opposite
directions, and nothing requires the same region to serve both.

Here the measurement region is always the conventional 5 mm sphere, so the
classic index is untouched and the quantity being reported stays comparable
with prior work. Only the source of the direction estimate varies:

  sphere   the sphere itself, about 106 voxels, the submitted method
  slab     the whole JHU tract label intersected with an axial band at the ALPS
           level, about 1300 voxels, so many voxels at the right level without
           integrating along the curving tract
  whole    the entire JHU tract label, about 1400 voxels, maximum voxels but a
           direction averaged over the tract's full curvature

Labels: 25 SCR_R, 26 SCR_L, 41 SLF_R, 42 SLF_L.

Usage:
    python decoupled_roi.py --limit 200
"""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from data_paths import winpath

import atomic_io  # noqa: F401  writes become atomic on import
from scipy import stats

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from estimator_variants import directional_diffusivity, variance_components
from direction_estimators import weights_for, principal, align, X, Y, Z
from alps_common import parse_age

HERE = Path(__file__).resolve().parent
DIFF = HERE.parent.parent / "diffusion"
OUT = winpath("Q:/dti_output")
# Chosen by sweeping band width against inter-fibre angle and
# within-participant reliability (slab_tuning.py). Tighter bands give
# more orthogonal directions by averaging less tract curvature; wider
# ones give more voxels. Reliability peaks at 8 to 10 mm. Pruning
# off-direction voxels was tested and rejected: it is self-reinforcing,
# locking in the initial estimate's bias rather than correcting it.
SLAB_MM = 8.0
FA_MIN = 0.2
SOURCES = ("sphere", "slab", "whole")
SHELL = os.environ.get("ALPS_TENSOR_SUFFIX", "")


def main() -> None:
    import nibabel as nib

    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=["hcpa", "dlbs"], default="hcpa")
    ap.add_argument("--limit", type=int, default=None,
                    help="random subsample of participants, for a quick check")
    args = ap.parse_args()

    if args.cohort == "hcpa":
        src = pd.read_csv(DIFF / "HCP" / "hcpa_alps_spheres_5mm.csv")
        mot = pd.read_csv(DIFF / "HCP" / "hcpa_motion.csv")
        src = src.merge(mot[["Subject_ID", "Visit", "Eddy_Mean_RMS"]],
                        on=["Subject_ID", "Visit"], how="left")
        rms = pd.to_numeric(src.Eddy_Mean_RMS, errors="coerce")
        thr = float(np.nanpercentile(rms.dropna(), 76.4))   # match DLBS stringency
        visit_col = "Visit"
    else:
        src = pd.read_csv(DIFF / "DLBS" / "dlbs_alps_spheres_5mm.csv")
        mot = pd.read_csv(DIFF / "DLBS" / "dlbs_motion.csv")
        src = src.merge(mot[["DTI_Session_ID", "Eddy_Mean_RMS"]],
                        on="DTI_Session_ID", how="left")
        rms = pd.to_numeric(src.Eddy_Mean_RMS, errors="coerce")
        thr = 0.5                                            # the published value
        src["Visit"] = src["Session"]
        visit_col = "Visit"

    src = src[(src.status == "ok") & (rms <= thr)].copy()
    src["Age"] = parse_age(src["Age"])
    src = src.dropna(subset=["Age"])
    counts = src.Subject_ID.value_counts()
    src = src[src.Subject_ID.isin(counts[counts >= 2].index)]
    if args.limit:
        rng = np.random.default_rng(20260728)
        keep = rng.choice(sorted(src.Subject_ID.unique()),
                          size=min(args.limit, src.Subject_ID.nunique()),
                          replace=False)
        src = src[src.Subject_ID.isin(keep)]
    src = src.sort_values(["Subject_ID", visit_col])
    print(f"cohort {args.cohort}: motion cutoff {thr:.3f} mm")
    print(f"{len(src)} sessions, {src.Subject_ID.nunique()} participants\n")

    rows = []
    for i, r in enumerate(src.itertuples(), 1):
        sd = OUT / r.DTI_Session_ID / "processed"
        lab_p = sd / "atlas" / "jhu_labels_registered.nii.gz"
        sph_p = sd / "atlas" / "sphere_roi" / "sphere_roi_combined.nii.gz"
        if not (lab_p.exists() and sph_p.exists()):
            continue
        try:
            limg = nib.load(str(lab_p))
            lab = limg.get_fdata().astype(int)
            sph = nib.load(str(sph_p)).get_fdata().astype(int)
            evals = nib.load(str(sd / f"tensor_eigenvalues{SHELL}.nii.gz")).get_fdata()
            evecs = nib.load(str(sd / f"tensor_eigenvectors{SHELL}.nii.gz")).get_fdata()
        except Exception:
            continue

        md = evals.mean(axis=-1)
        num = np.sqrt(((evals - md[..., None]) ** 2).sum(axis=-1))
        den = np.sqrt((evals**2).sum(axis=-1))
        fa = np.clip(np.sqrt(1.5) * np.divide(num, den, out=np.zeros_like(num),
                                              where=den != 0), 0, 1)

        ii, jj, kk = np.indices(lab.shape)
        zc = limg.affine[2, 0] * ii + limg.affine[2, 1] * jj + \
             limg.affine[2, 2] * kk + limg.affine[2, 3]
        # Hemisphere is taken from the world x coordinate, not the voxel index.
        # These volumes have a negative x scale, so voxel index < mid is world
        # x > 0, which is the RIGHT hemisphere. Pairing it with the _L labels
        # measured one hemisphere while estimating directions from the other.
        xc = (limg.affine[0, 0] * ii + limg.affine[0, 1] * jj
              + limg.affine[0, 2] * kk + limg.affine[0, 3])

        def pack(mask):
            v1 = evecs[mask][:, :, 0]
            n = np.linalg.norm(v1, axis=1, keepdims=True); n[n == 0] = 1
            return {"v1": v1 / n, "fa": fa[mask],
                    "evals": evals[mask], "evecs": evecs[mask]}

        rec = {"Subject_ID": r.Subject_ID, "Visit": r.Visit, "Age": r.Age}
        per_hemi = {s: [] for s in SOURCES}
        per_hemi_plus = {s: [] for s in SOURCES}
        classic_h = []
        for hemi, side, scr, slf in (("L", xc < 0, 26, 42), ("R", xc > 0, 25, 41)):
            m_proj_s = (sph == 1) & side & (fa >= FA_MIN)
            m_assoc_s = (sph == 2) & side & (fa >= FA_MIN)
            if m_proj_s.sum() < 4 or m_assoc_s.sum() < 4:
                continue
            proj_s, assoc_s = pack(m_proj_s), pack(m_assoc_s)

            classic_h.append(
                (directional_diffusivity(proj_s["evals"], proj_s["evecs"], X)
                 + directional_diffusivity(assoc_s["evals"], assoc_s["evecs"], X))
                / (directional_diffusivity(proj_s["evals"], proj_s["evecs"], Y)
                   + directional_diffusivity(assoc_s["evals"], assoc_s["evecs"], Z)))

            # Centre the band on the ALPS level itself, given by the sphere, not
            # on the label's centroid. The SCR label spans z +19 to +44, so its
            # centroid sits well above the level the index is defined at.
            z0 = float(np.median(zc[sph > 0])) if (sph > 0).any() else 0.0
            for source in SOURCES:
                if source == "sphere":
                    dp, da = proj_s, assoc_s
                else:
                    mp = (lab == scr) & (fa >= FA_MIN)
                    ma = (lab == slf) & (fa >= FA_MIN)
                    if source == "slab":
                        band = np.abs(zc - z0) <= SLAB_MM
                        mp, ma = mp & band, ma & band
                    if mp.sum() < 10 or ma.sum() < 10:
                        continue
                    dp, da = pack(mp), pack(ma)

                vp = align(principal(dp["v1"], weights_for("cl", dp)), Z)
                va = align(principal(da["v1"], weights_for("cl", da)), Y)
                p = np.cross(vp, va); p /= max(np.linalg.norm(p), 1e-12)
                op = np.cross(p, vp); op /= max(np.linalg.norm(op), 1e-12)
                oa = np.cross(p, va); oa /= max(np.linalg.norm(oa), 1e-12)
                # measured in the sphere regardless of where the axes came from
                per_hemi[source].append(
                    (directional_diffusivity(proj_s["evals"], proj_s["evecs"], p)
                     + directional_diffusivity(assoc_s["evals"], assoc_s["evecs"], p))
                    / (directional_diffusivity(proj_s["evals"], proj_s["evecs"], op)
                       + directional_diffusivity(assoc_s["evals"], assoc_s["evecs"], oa)))

                # Refined+ : the same PVS axis projected onto the transverse plane
                # of each measurement voxel, then pooled, so that it is
                # perpendicular to the local fibre direction rather than only to
                # the two region-mean directions. The axes still come from
                # `source`; only this correction uses the sphere's own voxels.
                acc, wts = [], []
                for roi in (proj_s, assoc_s):
                    v1 = roi["v1"]
                    pr = p - (v1 @ p)[:, None] * v1
                    nn = np.linalg.norm(pr, axis=1, keepdims=True)
                    good = nn[:, 0] > 1e-8
                    acc.append(pr[good] / nn[good])
                    wts.append(roi["fa"][good])
                if sum(len(a) for a in acc) >= 4:
                    pp = align(principal(np.vstack(acc), np.concatenate(wts)), p)
                    opp = np.cross(pp, vp); opp /= max(np.linalg.norm(opp), 1e-12)
                    oap = np.cross(pp, va); oap /= max(np.linalg.norm(oap), 1e-12)
                    per_hemi_plus[source].append(
                        (directional_diffusivity(proj_s["evals"], proj_s["evecs"], pp)
                         + directional_diffusivity(assoc_s["evals"], assoc_s["evecs"], pp))
                        / (directional_diffusivity(proj_s["evals"], proj_s["evecs"], opp)
                           + directional_diffusivity(assoc_s["evals"], assoc_s["evecs"], oap)))
        if not classic_h:
            continue
        rec["classic"] = float(np.mean(classic_h))
        for s in SOURCES:
            if per_hemi[s]:
                rec[f"refined_{s}"] = float(np.mean(per_hemi[s]))
            if per_hemi_plus[s]:
                rec[f"refinedplus_{s}"] = float(np.mean(per_hemi_plus[s]))
        rows.append(rec)
        if i % 25 == 0:
            print(f"  {i}/{len(src)}", flush=True)

    d = pd.DataFrame(rows).dropna(
        subset=["classic"] + [f"refined_{s}" for s in SOURCES])
    counts = d.Subject_ID.value_counts()
    lon = d[d.Subject_ID.isin(counts[counts >= 2].index)]
    print(f"\nusable {len(d)} sessions, longitudinal {len(lon)}, "
          f"{lon.Subject_ID.nunique()} participants")
    print("measurement region is the 5 mm sphere throughout\n")

    base = variance_components(lon, "classic")
    rb = stats.linregress(d["Age"], d["classic"])[2]
    print(f"{'direction from':<16s} {'ICC':>7s} {'var_within':>12s} {'penalty':>8s} "
          f"{'r age':>8s} {'disatt':>8s}")
    print(f"{'(classic)':<16s} {base['icc']:7.3f} {base['var_within']:12.6f} "
          f"{'':>8s} {rb:8.3f} {rb/np.sqrt(base['icc']):8.3f}")
    for s in SOURCES:
        for pre, lbl in (("refined", s), ("refinedplus", f"{s} (+)")):
            col = f"{pre}_{s}"
            if col not in d or d[col].isna().all():
                continue
            sub = lon.dropna(subset=[col])
            vc = variance_components(sub, col)
            dd_ = d.dropna(subset=[col])
            r = stats.linregress(dd_["Age"], dd_[col])[2]
            print(f"{lbl:<16s} {vc['icc']:7.3f} {vc['var_within']:12.6f} "
                  f"{vc['var_within']/base['var_within']:7.2f}x {r:8.3f} "
                  f"{r/np.sqrt(vc['icc']):8.3f}")

    d.to_csv(HERE / f"decoupled_roi_{args.cohort}{SHELL}.csv", index=False)
    print(f"\nWrote decoupled_roi_{args.cohort}.csv")


if __name__ == "__main__":
    main()
