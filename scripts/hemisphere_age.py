"""
Left and right hemispheres separately, with a formal test between correlations.

Reviewer 4 asked for this: averaging the hemispheres before correlating discards
the left-right asymmetry that is most sensitive to orientation, n=62 was
underpowered to separate the variants, and no test of the difference between
correlations was reported.

The existing age_lr_table.csv answers it on the n=62 manual cohort, which is the
very sample the reviewer called underpowered, and it uses sphere-estimated axes
rather than the tract-band axes the method now uses. This redoes it on the full
automated cohorts with the current estimator.

Williams' test is the right one here because the correlations are dependent and
overlapping: r(age, classic) and r(age, refined) share the age variable and are
computed in the same sessions, so an independent-samples comparison would be
wrong.
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
from scipy import stats

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from estimator_variants import directional_diffusivity
from direction_estimators import weights_for, principal, align, X, Y, Z
from alps_common import parse_age

HERE = Path(__file__).resolve().parent
DIFF = HERE.parent.parent / "diffusion"
OUT = winpath("Q:/dti_output")
SLAB_MM, FA_MIN = 8.0, 0.2
SHELL = os.environ.get("ALPS_TENSOR_SUFFIX", "")


def williams(r12, r13, r23, n):
    """Williams' t for dependent overlapping correlations r(1,2) vs r(1,3)."""
    if n < 6:
        return np.nan, np.nan
    R = 1 - r12 ** 2 - r13 ** 2 - r23 ** 2 + 2 * r12 * r13 * r23
    num = (r12 - r13) * np.sqrt((n - 1) * (1 + r23))
    den = np.sqrt(2 * (n - 1) / (n - 3) * R + ((r12 + r13) ** 2) / 4 * (1 - r23) ** 3)
    t = num / den
    return float(t), float(2 * (1 - stats.t.cdf(abs(t), n - 3)))


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
    src = src.dropna(subset=["Age"]).sort_values(["Subject_ID", "Visit"])
    if args.limit:
        src = src.head(args.limit)
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
        zc = (limg.affine[2, 0] * ii + limg.affine[2, 1] * jj
              + limg.affine[2, 2] * kk + limg.affine[2, 3])
        # Hemisphere is taken from the world x coordinate, not the voxel index.
        # These volumes have a negative x scale, so voxel index < mid is world
        # x > 0, which is the RIGHT hemisphere. Pairing it with the _L labels
        # measured one hemisphere while estimating directions from the other.
        xc = (limg.affine[0, 0] * ii + limg.affine[0, 1] * jj
              + limg.affine[0, 2] * kk + limg.affine[0, 3])

        def pack(mask):
            v1 = evecs[mask][:, :, 0]
            n = np.linalg.norm(v1, axis=1, keepdims=True); n[n == 0] = 1
            return {"v1": v1 / n, "fa": fa[mask], "evals": evals[mask], "evecs": evecs[mask]}

        rec = {"Subject_ID": r.Subject_ID, "Visit": r.Visit, "Age": r.Age}
        for hemi, side, scr, slf in (("L", xc < 0, 26, 42), ("R", xc > 0, 25, 41)):
            mp_s = (sph == 1) & side & (fa >= FA_MIN)
            ma_s = (sph == 2) & side & (fa >= FA_MIN)
            if mp_s.sum() < 4 or ma_s.sum() < 4:
                continue
            P, A = pack(mp_s), pack(ma_s)
            rec[f"classic_{hemi}"] = float(
                (directional_diffusivity(P["evals"], P["evecs"], X)
                 + directional_diffusivity(A["evals"], A["evecs"], X))
                / (directional_diffusivity(P["evals"], P["evecs"], Y)
                   + directional_diffusivity(A["evals"], A["evecs"], Z)))

            z0 = float(np.median(zc[sph > 0])) if (sph > 0).any() else 0.0
            band = np.abs(zc - z0) <= SLAB_MM
            mp_l = (lab == scr) & (fa >= FA_MIN) & band
            ma_l = (lab == slf) & (fa >= FA_MIN) & band
            if mp_l.sum() < 10 or ma_l.sum() < 10:
                continue
            Lp, La = pack(mp_l), pack(ma_l)
            vp = align(principal(Lp["v1"], weights_for("cl", Lp)), Z)
            va = align(principal(La["v1"], weights_for("cl", La)), Y)
            p = np.cross(vp, va); p /= max(np.linalg.norm(p), 1e-12)
            op = np.cross(p, vp); op /= max(np.linalg.norm(op), 1e-12)
            oa = np.cross(p, va); oa /= max(np.linalg.norm(oa), 1e-12)
            rec[f"refined_{hemi}"] = float(
                (directional_diffusivity(P["evals"], P["evecs"], p)
                 + directional_diffusivity(A["evals"], A["evecs"], p))
                / (directional_diffusivity(P["evals"], P["evecs"], op)
                   + directional_diffusivity(A["evals"], A["evecs"], oa)))
        rows.append(rec)
        if i % 100 == 0:
            print(f"  {i}/{len(src)}", flush=True)

    d = pd.DataFrame(rows)
    out = HERE / f"hemisphere_age_{args.cohort}{SHELL}.csv"
    d.to_csv(out, index=False)
    print(f"\n{len(d)} sessions -> {out.name}")

    # One session per participant, so the correlations do not treat repeat
    # visits as independent. Earliest visit is used.
    first = d.sort_values(["Subject_ID", "Visit"]).groupby("Subject_ID").first().reset_index()
    print(f"one-session-per-participant subset: {len(first)}\n")
    print(f"{'side':<8s} {'r classic':>10s} {'r refined':>10s} {'Williams t':>11s} {'p':>9s}")
    for side in ("L", "R", "Avg"):
        if side == "Avg":
            first["classic_Avg"] = first[["classic_L", "classic_R"]].mean(axis=1)
            first["refined_Avg"] = first[["refined_L", "refined_R"]].mean(axis=1)
        s = first[["Age", f"classic_{side}", f"refined_{side}"]].dropna()
        n = len(s)
        r1 = stats.pearsonr(s.Age, s[f"classic_{side}"])[0]
        r2 = stats.pearsonr(s.Age, s[f"refined_{side}"])[0]
        r23 = stats.pearsonr(s[f"classic_{side}"], s[f"refined_{side}"])[0]
        t, pv = williams(r1, r2, r23, n)
        print(f"{side:<8s} {r1:>10.3f} {r2:>10.3f} {t:>11.2f} {pv:>9.4f}   (n={n})")

    print("\nleft-right difference within each index (paired t on Fisher z)")
    for idx in ("classic", "refined"):
        s = first[[f"{idx}_L", f"{idx}_R"]].dropna()
        t, pv = stats.ttest_rel(s[f"{idx}_L"], s[f"{idx}_R"])
        print(f"  {idx:<10s} mean L {s.iloc[:,0].mean():.4f}  R {s.iloc[:,1].mean():.4f}  "
              f"t {t:+.2f}  p {pv:.3e}")


if __name__ == "__main__":
    main()
