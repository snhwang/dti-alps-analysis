"""Which normalization of the transverse anisotropy is the better metric?

NOT part of the current manuscript. Exploratory, alongside
generalized_alps_model.py, and outside the regeneration chain.

The index reduces to lambda2/lambda3, and the paper treats that ratio as the
quantity of interest. But the ratio form was inherited from the structure of
ALPS, which is a ratio, and never chosen for its statistical properties. It has
an obvious defect. Its denominator is the smallest eigenvalue, which is the
noisiest, the most affected by sorting bias, and the one that can approach zero.
Body CC came back with a median ratio of 13 and a maximum of 924 before the
regional form was reformed as a ratio of means.

So compare the candidates on the terms that matter for a measurement:

  ratio         l2 / l3                     the current quantity, unbounded
  planarity     (l2 - l3) / l1              Westin CP, bounded in [0, 1]
  norm_diff     (l2 - l3) / (l2 + l3)       transverse asymmetry, bounded
  diff          l2 - l3                     raw, carries units
  log_ratio     ln(l2 / l3)                 symmetric about zero, unbounded

All five are functions of the eigenvalues alone, so all five are exactly
rotation-invariant. They differ in conditioning, in reliability, and possibly in
how strongly they track age.

Judged on:

  tail          how heavy the upper tail is, as the ratio of the 99th
                percentile to the median. The ratio's failure mode.
  ICC           between-visit reliability, same estimator as the paper
  CoV           within-participant variability relative to the mean
  r age         cross-sectional age association, first visit per participant
  disattenuated r/sqrt(ICC), which is what the comparison should turn on,
                since a metric cannot be preferred merely for being noisier

Needs the regional eigenvalue means, which measured_pvs_axis.py writes.

    python normalization_comparison.py

Writes normalization_comparison.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import atomic_io  # noqa: F401  writes become atomic on import

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from estimator_variants import variance_components  # noqa: E402
COHORTS = (("HCP-A", "measured_pvs_axis_hcpa_b1500_all.csv"),
           ("DLBS", "measured_pvs_axis_dlbs.csv"))
NEED = [f"{e}_{r}" for e in ("l1", "l2", "l3") for r in ("proj", "assoc")]


def normalizations(d):
    """Region-pooled, formed as ratios of means to match the paper's convention."""
    l1 = (d.l1_proj + d.l1_assoc) / 2
    l2 = (d.l2_proj + d.l2_assoc) / 2
    l3 = (d.l3_proj + d.l3_assoc) / 2
    return pd.DataFrame({
        "ratio": l2 / l3,
        "planarity": (l2 - l3) / l1,
        "norm_diff": (l2 - l3) / (l2 + l3),
        "diff": (l2 - l3) * 1e3,
        "log_ratio": np.log(l2 / l3),
    })


def icc_and_cov(long, col):
    """Between-visit ICC and within-participant CoV.

    Uses the paper's own estimator so the numbers here sit on the same footing
    as its Table of variants, rather than on a second definition of ICC.
    """
    rep = long.Subject_ID.value_counts()
    s = long[long.Subject_ID.isin(rep[rep >= 2].index)]
    if s.Subject_ID.nunique() < 10:
        return float("nan"), float("nan")
    vc = variance_components(s, col)
    return float(vc["icc"]), float(vc["wcv_pct"]) / 100.0


def main() -> None:
    argparse.ArgumentParser().parse_args()
    rows = []

    for cohort, fname in COHORTS:
        path = HERE / fname
        if not path.exists():
            print(f"{cohort}: {fname} not found")
            continue
        d = pd.read_csv(path)
        missing = [c for c in NEED if c not in d.columns]
        if missing:
            print(f"{cohort}: regional eigenvalue means absent "
                  f"({missing[0]} ...), rerun measured_pvs_axis.py")
            continue
        d["Subject_ID"] = d.Subject_ID.astype(str)
        v = normalizations(d)
        d = pd.concat([d[["Subject_ID", "Visit", "Age"]], v], axis=1)
        first = d.sort_values(["Subject_ID", "Visit"]).groupby(
            "Subject_ID", as_index=False).first()

        print(f"\n{cohort}  n = {len(d)} sessions, "
              f"{d.Subject_ID.nunique()} participants\n")
        print(f"   {'metric':11s} {'p99/median':>11s} {'ICC':>7s} {'CoV':>8s} "
              f"{'r age':>8s} {'disatt':>8s}")
        for col in v.columns:
            s = d[[col, "Subject_ID"]].replace([np.inf, -np.inf], np.nan).dropna()
            tail = float(s[col].quantile(0.99) / s[col].median())
            icc, cov = icc_and_cov(d.dropna(subset=[col]), col)
            f = first[[col, "Age"]].replace([np.inf, -np.inf], np.nan).dropna()
            r = float(np.corrcoef(f[col], f.Age)[0, 1])
            dis = r / np.sqrt(icc) if icc and icc > 0 else float("nan")
            rows.append(dict(cohort=cohort, metric=col, tail_p99_over_median=round(tail, 3),
                             icc=round(icc, 4), cov=round(cov, 4),
                             r_age=round(r, 4), r_disattenuated=round(dis, 4),
                             n_sessions=len(s),
                             n_participants=int(d.Subject_ID.nunique())))
            print(f"   {col:11s} {tail:11.2f} {icc:7.3f} {cov:8.4f} "
                  f"{r:+8.3f} {dis:+8.3f}")

    if not rows:
        raise SystemExit("no cohort had the eigenvalue columns")

    out = pd.DataFrame(rows)
    out.to_csv(HERE / "normalization_comparison.csv", index=False)

    print("\n\n   Reading it\n")
    for cohort, g in out.groupby("cohort", sort=False):
        g = g.set_index("metric")
        best_icc = g.icc.idxmax()
        best_dis = g.r_disattenuated.abs().idxmax()
        worst_tail = g.tail_p99_over_median.idxmax()
        print(f"   {cohort:8s} heaviest tail {worst_tail}, best ICC {best_icc}, "
              f"strongest disattenuated {best_dis}")
    print("\n   A normalization is worth switching to only if it wins on the")
    print("   disattenuated association, since a bounded form that is merely")
    print("   less variable buys nothing a larger sample would not.")


if __name__ == "__main__":
    main()
