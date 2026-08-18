"""One table of every headline quantity, with the sample that produced it.

The same labelled quantity appears with different values in different sections
of the manuscript, because different analyses run on different subsets and the
text does not say so at the point of use. Every value is individually correct
against its own data, which is why the per-number checks in verify_manuscript.py
all pass while the paper still looks inconsistent to a reader.

This regenerates all of them side by side so each in-text number can be tagged
with its sample. The samples are:

  variants      every session where all variants could be computed
  variants-1    the same, one session per participant
  pose          sessions that also carry a recovered head rotation
  pose-1        the same, one session per participant   <- abstract and 3.4
  decoupled     the decoupling analysis, direction estimated from the tract band
  longitudinal  participants imaged at least twice, used for every ICC
  placement     the placement-quality set, used for composition and off-tract

Run from revision/. Writes reconciliation_table.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import atomic_io  # noqa: F401  writes become atomic on import

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from estimator_variants import variance_components  # noqa: E402

VARIANTS = ("classic", "cross", "v2_slab", "pv_perp")


def key(d):
    d = d.copy()
    d["Subject_ID"] = d.Subject_ID.astype(str)
    d["Visit"] = d.Visit.astype(str)
    return d


def one_per(d):
    return d.sort_values(["Subject_ID", "Visit"]).groupby("Subject_ID").first().reset_index()


def longitudinal(d, col):
    d = d.dropna(subset=[col])
    return d[d.Subject_ID.isin(d.Subject_ID.value_counts()[lambda s: s >= 2].index)]


def age_r(d, col):
    s = d.dropna(subset=["Age", col])
    return (float(np.corrcoef(s.Age, s[col])[0, 1]), len(s), s.Subject_ID.nunique()) if len(s) > 3 else (np.nan, 0, 0)


def build(cohort, vfile, pfile, dfile):
    v = key(pd.read_csv(HERE / vfile))
    pose = key(pd.read_csv(HERE / pfile)).merge(v, on=["Subject_ID", "Visit"])
    dec = key(pd.read_csv(HERE / dfile))
    samples = {"variants": v, "variants-1": one_per(v),
               "pose": pose, "pose-1": one_per(pose), "decoupled": dec}
    rows = []
    for sname, s in samples.items():
        for col in VARIANTS + ("refined_slab",):
            if col not in s.columns:
                continue
            r, n, npart = age_r(s, col)
            if not np.isnan(r):
                rows.append(dict(cohort=cohort, sample=sname, quantity="age r",
                                 variant=col, value=round(r, 4), n=n, participants=npart))
            lon = longitudinal(s, col)
            if len(lon) > 10:
                icc = variance_components(lon, col)["icc"]
                rows.append(dict(cohort=cohort, sample=f"{sname}/longitudinal",
                                 quantity="ICC", variant=col, value=round(icc, 4),
                                 n=len(lon), participants=lon.Subject_ID.nunique()))
    return rows


def main() -> None:
    rows = build("HCP-A", "measured_pvs_axis_hcpa_b1500_all.csv",
                 "head_rotation_hcpa.csv", "decoupled_roi_hcpa_b1500.csv")
    rows += build("DLBS", "measured_pvs_axis_dlbs.csv",
                  "head_rotation_dlbs.csv", "decoupled_roi_dlbs.csv")
    d = pd.DataFrame(rows)
    d.to_csv(HERE / "reconciliation_table.csv", index=False)

    for cohort in ("HCP-A", "DLBS"):
        for q in ("age r", "ICC"):
            sub = d[(d.cohort == cohort) & (d.quantity == q)]
            if sub.empty:
                continue
            print(f"\n=== {cohort}  {q} ===")
            piv = sub.pivot_table(index=["sample", "n", "participants"],
                                  columns="variant", values="value")
            print(piv.to_string(float_format=lambda x: f"{x:7.4f}"))

    print("\n\n=== values that appear more than once, and where they differ ===")
    for q in ("age r", "ICC"):
        for cohort in ("HCP-A", "DLBS"):
            for var in VARIANTS:
                sub = d[(d.cohort == cohort) & (d.quantity == q) & (d.variant == var)]
                vals = sorted(sub.value.unique())
                if len(vals) > 1:
                    spread = max(vals) - min(vals)
                    print(f"  {cohort:6s} {q:6s} {var:9s} spans {min(vals):+.4f} to "
                          f"{max(vals):+.4f}  (spread {spread:.4f}) across {len(vals)} samples")


if __name__ == "__main__":
    main()
