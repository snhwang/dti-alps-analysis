"""What does the corrected v2 pooling weight change in the reported numbers?

Methods weights voxels by directional reliability, which for v2 means the gap to
the NEAREST eigenvalue, min(CL, CP). The analysis pools v2 with CP alone. Only
v2_sphere and v2_slab consume that weight: classic and cross use no v2 axis,
anat_x comes from the affine, and pv_perp uses eigenvalues with no axis at all.

This puts the two weights side by side on identical samples, using the same
ICC and age-correlation code the production script prints, so the difference
is read off the reported quantities rather than off an axis angle.

Samples are matched by construction: both files come from the same selection
rule, so this compares row for row on the participants present in both.

    python v2_weight_compare.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from estimator_variants import variance_components  # noqa: E402
from alps_common import parse_age  # noqa: E402

VARIANTS = ["classic", "cross", "v2_sphere", "v2_slab", "pv_perp", "anat_x"]
# The weight touches only these two. The rest are a control: if any of them
# moves, something other than the weight changed and the comparison is invalid.
AFFECTED = {"v2_sphere", "v2_slab"}


def summarise(d: pd.DataFrame) -> dict:
    d = d.copy()
    d["Age"] = parse_age(d["Age"])
    d = d.dropna(subset=["Age"])
    base = variance_components(d.dropna(subset=["classic"]), "classic")
    out = {}
    for k in VARIANTS:
        if k not in d:
            continue
        s = d.dropna(subset=[k])
        if len(s) < 20:
            continue
        vc = variance_components(s, k)
        first = s.sort_values(["Subject_ID", "Visit"]).groupby("Subject_ID").first()
        r = float(stats.pearsonr(first[k], first["Age"])[0])
        out[k] = dict(icc=vc["icc"], var=vc["var_within"] / base["var_within"], r=r)
    return out


def main() -> None:
    for cohort, cp_f, gap_f in (
            ("DLBS", "measured_pvs_axis_dlbs.csv",
             "measured_pvs_axis_dlbs_v2gap.csv"),
            ("HCP-A", "measured_pvs_axis_hcpa_b1500_all.csv",
             "measured_pvs_axis_hcpa_b1500_v2gap.csv")):
        pc, pg = HERE / cp_f, HERE / gap_f
        if not (pc.exists() and pg.exists()):
            print(f"{cohort}: missing input\n")
            continue
        a, b = pd.read_csv(pc), pd.read_csv(pg)
        for x in (a, b):
            x["Subject_ID"] = x.Subject_ID.astype(str)
            x["Visit"] = x.Visit.astype(str)
        key = ["Subject_ID", "Visit"]
        both = set(map(tuple, a[key].values)) & set(map(tuple, b[key].values))
        a = a[[tuple(v) in both for v in a[key].values]]
        b = b[[tuple(v) in both for v in b[key].values]]
        print(f"{cohort}: {len(a)} sessions, {a.Subject_ID.nunique()} participants "
              f"present in both")

        A, B = summarise(a), summarise(b)
        print(f"  {'variant':<11s} {'ICC cp':>7s} {'ICC gap':>8s} "
              f"{'r cp':>8s} {'r gap':>8s} {'d r':>8s}")
        for k in VARIANTS:
            if k not in A or k not in B:
                continue
            mark = "  <-- uses the weight" if k in AFFECTED else ""
            print(f"  {k:<11s} {A[k]['icc']:>7.3f} {B[k]['icc']:>8.3f} "
                  f"{A[k]['r']:>8.3f} {B[k]['r']:>8.3f} "
                  f"{B[k]['r'] - A[k]['r']:>+8.3f}{mark}")

        moved = [k for k in VARIANTS if k in A and k in B
                 and k not in AFFECTED
                 and (abs(A[k]["r"] - B[k]["r"]) > 5e-4
                      or abs(A[k]["icc"] - B[k]["icc"]) > 5e-4)]
        print(f"  control: unaffected variants that moved: {moved or 'none'}")
        if "sph_to_slab" in b:
            print(f"  sphere axis to band axis: median "
                  f"{b.sph_to_slab.median():.2f} deg")
        print()


if __name__ == "__main__":
    main()
