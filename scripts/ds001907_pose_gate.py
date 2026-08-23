"""Can ds001907 answer the head-pose question at all, before any compute is spent?

The hypothesis is that Parkinson's patients lie differently in the scanner. That
can only be tested if the images still carry the pose. Two ways it could already
be gone:

  1. The series was resampled into a standard space during curation, in which
     case every image shares one geometry and pose is unrecoverable.
  2. The prescription tracked the head (auto-align), in which case the scanner
     affine absorbs the tilt and the brain sits square inside the voxel grid.

The paper measures pose from the subject-to-template FLIRT affine, which reads
the rotation of the brain inside the image, so case 2 does not block it. Case 1
does. This distinguishes them from headers alone, in seconds, and also reports
the obliquity the scanner recorded, which is a second and independent pose
channel this dataset happens to preserve.

Also joins the participants table, since a group contrast needs group, age and
sex, and an imbalance there decides the model before any imaging is touched.

    python ds001907_pose_gate.py
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from ds001907_common import group, assert_group_mapping, DEST, HERE

TAG = "3.0.2"
CLONE = Path(r"C:\tmp\ds001907git")


def obliquity(aff: np.ndarray) -> float:
    """Degrees between the slice normal and the nearest scanner axis.

    The third column of the affine is the slice direction. An axial prescription
    with no tilt lines it up with z, so the angle is zero. Anything the operator
    tilted, or any auto-align that followed the head, shows up here.
    """
    v = aff[:3, 2] / np.linalg.norm(aff[:3, 2])
    return float(np.degrees(np.arccos(np.clip(np.abs(v).max(), -1, 1))))


def main() -> None:
    argparse.ArgumentParser().parse_args()

    # Group, age and sex come from ds001907_common, which re-derives the
    # prefix-to-group mapping from Hoehn and Yahr rather than trusting the
    # README. The README has the two arms swapped.
    dem = HERE / "ds001907_demographics.csv"
    if not dem.exists():
        from fetch_ds001907_dwi import fetch
        dem.write_bytes(fetch("demographics.csv"))
    p = assert_group_mapping()
    print("=== cohort (group mapping verified, not taken from the README) ===")
    for grp, s in p.groupby("group"):
        print(f"   {grp:8s} n={len(s):3d}  age {s.age.mean():.1f} "
              f"(SD {s.age.std():.1f})  {(s.sex=='Male').sum()} M / "
              f"{(s.sex=='Female').sum()} F")
    from scipy import stats
    pa, ca = (p.loc[p.group == g, "age"].dropna() for g in ("patient", "control"))
    t, pt = stats.ttest_ind(pa, ca, equal_var=False)
    tab = pd.crosstab(p.group, p.sex)
    _, px, _, _ = stats.chi2_contingency(tab)
    print(f"   age difference   Welch t={t:.2f}, p={pt:.3f}")
    print(f"   sex difference   chi2 p={px:.3f}"
          + ("   <-- imbalanced, must be a covariate" if px < .1 else ""))

    # --- geometry -----------------------------------------------------------
    imgs = sorted(DEST.rglob("*_dwi.nii.gz"))
    rows = []
    for f in imgs:
        h = nib.load(str(f))
        sid = f.parts[len(DEST.parts)]
        rows.append({
            "subject": sid,
            "group": group(sid),
            "file": str(f.relative_to(DEST)),
            "shape": "x".join(str(x) for x in h.shape),
            "zooms": ",".join(f"{z:.2f}" for z in h.header.get_zooms()[:3]),
            "obliquity_deg": obliquity(h.affine),
            "affine_hash": hash(np.round(h.affine, 3).tobytes()),
        })
    g = pd.DataFrame(rows)
    g.to_csv(HERE / "ds001907_geometry.csv", index=False)
    print(f"\n{len(g)} images read\n")

    print("=== is every image on one shared grid? ===")
    print(f"   distinct shapes            {g['shape'].nunique()}  {sorted(set(g['shape']))}")
    print(f"   distinct voxel sizes       {g.zooms.nunique()}  {sorted(set(g.zooms))}")
    print(f"   distinct affines           {g.affine_hash.nunique()} of {len(g)}")
    if g.affine_hash.nunique() <= 2:
        print("\n   Every image shares one affine. The series was resampled into a")
        print("   common space, so head pose is gone and this dataset cannot test")
        print("   the hypothesis.")
        return
    print("   Affines differ per scan, so the images are in native acquisition")
    print("   space and pose survives.")

    print("\n=== scanner obliquity, the prescription's own record of tilt ===")
    for grp, d in g.groupby("group"):
        v = d.obliquity_deg
        print(f"   {grp:8s} n={len(v):3d}  median {v.median():5.2f} deg  "
              f"IQR {v.quantile(.25):.2f}-{v.quantile(.75):.2f}  max {v.max():.2f}")
    from scipy import stats
    a = g.loc[g.group == "patient", "obliquity_deg"]
    b = g.loc[g.group == "control", "obliquity_deg"]
    u, pu = stats.mannwhitneyu(a, b)
    print(f"\n   Mann-Whitney patient vs control: U={u:.0f}, p={pu:.3f}")
    if g.obliquity_deg.max() < 0.5:
        print("   Prescriptions were essentially straight axial, so the scanner")
        print("   affine carries no tilt. Any pose signal has to come from the")
        print("   brain's rotation inside the image, which needs registration.")
    print(f"\n   wrote {HERE / 'ds001907_geometry.csv'}")


if __name__ == "__main__":
    main()
