"""Do the beyond-ratio phenotype associations survive a rank test and outlier
removal?

phenotype_longitudinal.py reports Pearson partial correlations. Two of its
results are large enough to matter for what the paper claims, and Pearson is
the wrong tool for deciding whether either is real:

  glucose   The measured-axis index gives r = 0.324 raw and 0.376 with the
            ratio partialled, against 0.055 for classic, 0.071 for the cross
            product and 0.079 for the ratio itself. Only the two measured-axis
            forms see it, the sphere-pooled form sees it twice as strongly as
            the band-pooled one although the two agree to 0.005 on age, and it
            gets stronger when the ratio is removed. Glucose also has a long
            right tail and varies within a participant with fasting state, so a
            handful of visits can carry a Pearson coefficient.

  moca_sum  The paper's own beyond-ratio result, r = 0.128 for the cross
            product. Carried here as a positive control: whatever the battery
            does to glucose, it has to leave a real association standing.

Four tests, on the same within-participant design as the parent script. Each
variable is centered on the participant's own mean and centered age is
partialled out, so time-invariant confounds cancel exactly.

  pearson     the parent script's test, recomputed here so the rest is
              comparable rather than quoted
  spearman    every variable replaced by its rank over the analysis sample
              before centering, which is the same design on ranks
  trim3       Pearson after dropping sessions whose centered index or centered
              phenotype exceeds three within-participant standard deviations
  winsor      Pearson after winsorising both centered variables at the 1st and
              99th percentiles

and one influence measure, the largest single-session change in r under
leave-one-out. An association that a rank test and a 1% winsorisation both
survive is not carried by a few visits. One that halves is.

    python beyond_ratio_robustness.py
    python beyond_ratio_robustness.py --phenotypes glucose moca_sum hdl

Writes beyond_ratio_robustness.csv one row at a time, so a long leave-one-out
pass can be interrupted without losing what it has already established.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

from phenotype_longitudinal import (ANGLES, VARIANTS, index_table,
                                    long_phenotypes, partial_corr, within)

HERE = Path(__file__).resolve().parent
OUT = HERE / "beyond_ratio_robustness.csv"
FIELDS = ["cohort", "phenotype", "variant", "arm", "test", "n", "r",
          "p", "max_loo_delta"]


def ranked(s: pd.Series) -> pd.Series:
    return s.rank(method="average")


def trimmed(x, y, k=3.0):
    ok = np.ones(len(x), bool)
    for v in (x, y):
        sd = np.nanstd(v)
        if sd > 0:
            ok &= np.abs(v - np.nanmean(v)) <= k * sd
    return ok


def winsorise(v, lo=1, hi=99):
    a, b = np.nanpercentile(v, [lo, hi])
    return np.clip(v, a, b)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=["hcpa", "dlbs"], default="hcpa")
    ap.add_argument("--phenotypes", nargs="+",
                    default=["glucose", "moca_sum"])
    ap.add_argument("--loo", action="store_true", default=True)
    args = ap.parse_args()

    d = index_table(args.cohort)
    ph, _ = long_phenotypes(args.cohort)
    ph["Subject_ID"] = ph.Subject_ID.astype(str)
    ph["Visit"] = ph.Visit.astype(str)
    m = d.merge(ph, on=["Subject_ID", "Visit"], how="inner")
    n_ses = m.groupby("Subject_ID").size()
    m = m[m.Subject_ID.isin(n_ses[n_ses >= 2].index)].copy()
    variants = [v for v in VARIANTS + ANGLES if v in m.columns]
    print(f"{args.cohort}: {len(m)} sessions, "
          f"{m.Subject_ID.nunique()} participants with repeats\n")

    new = not OUT.exists()
    fh = OUT.open("a", newline="", encoding="utf-8")
    w = csv.DictWriter(fh, FIELDS)
    if new:
        w.writeheader()

    def emit(**row):
        w.writerow(row)
        fh.flush()
        print(f"  {row['phenotype']:<10s} {row['variant']:<10s} "
              f"{row['arm']:<10s} {row['test']:<9s} n={row['n']:>5} "
              f"r={row['r']:+.3f}  p={row['p']:.2e}"
              + (f"  loo={row['max_loo_delta']:+.3f}"
                 if row["max_loo_delta"] == row["max_loo_delta"] else ""))

    for phen in args.phenotypes:
        if phen not in m.columns:
            print(f"  {phen}: not in this cohort, skipped")
            continue
        for variant in variants:
            for arm, extra in (("age", []), ("age+ratio", ["pv_perp"])):
                if variant in ("pv_perp", "ratio") and extra:
                    continue
                cols = [variant, phen, "Age"] + extra
                sub = m.dropna(subset=[c for c in cols if c in m.columns])
                if len(sub) < 40:
                    continue
                for test in ("pearson", "spearman", "trim3", "winsor"):
                    f = sub.copy()
                    if test == "spearman":
                        for c in cols:
                            f[c] = ranked(f[c])
                    c = within(f, cols)
                    x = c[variant].to_numpy(float)
                    y = c[phen].to_numpy(float)
                    Z = c[["Age"] + extra].to_numpy(float)
                    if test == "trim3":
                        keep = trimmed(x, y)
                        x, y, Z = x[keep], y[keep], Z[keep]
                    elif test == "winsor":
                        x, y = winsorise(x), winsorise(y)
                    r, p, n = partial_corr(x, y, Z)
                    delta = float("nan")
                    # Leave-one-out only where the result is worth defending.
                    if (test == "pearson" and args.loo and r == r
                            and abs(r) > 0.05):
                        rs = []
                        for i in range(len(x)):
                            k = np.ones(len(x), bool)
                            k[i] = False
                            ri, _, _ = partial_corr(x[k], y[k], Z[k])
                            rs.append(ri)
                        rs = np.asarray(rs, float)
                        delta = float(np.nanmax(np.abs(rs - r)))
                    emit(cohort=args.cohort, phenotype=phen, variant=variant,
                         arm=arm, test=test, n=n, r=r, p=p,
                         max_loo_delta=delta)
    fh.close()

    # How far apart are the two measured-axis forms the paper calls equivalent?
    if {"v2_sphere", "v2_slab"} <= set(m.columns):
        c = within(m.dropna(subset=["v2_sphere", "v2_slab"]),
                   ["v2_sphere", "v2_slab"])
        print(f"\n  within-participant r(v2_sphere, v2_slab) = "
              f"{np.corrcoef(c.v2_sphere, c.v2_slab)[0, 1]:+.3f} "
              f"over {len(c)} sessions")
    print(f"\n  wrote {OUT.name}")


if __name__ == "__main__":
    main()
