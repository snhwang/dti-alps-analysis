"""Does the FA floor change any conclusion?

Reviewer 1 asked for FA-threshold sensitivity. The floor is configurable in
measured_pvs_axis.py through ALPS_FA_MIN and has been since the revision began,
but it had never actually been swept, so the ask was the one that went nowhere.

The floor is not a neutral parameter here, and that is why it is worth a sweep
rather than an assurance. It cuts both ways with age. Keeping it removes more
voxels in older brains, because fractional anisotropy falls, and dropping it
admits more ventricular partial volume in older brains, because the ventricles
enlarge. Either direction could in principle manufacture an age association, so
what has to be shown is that the paper's conclusions do not move with it.

The conclusions this checks, one per row of the output:

  attainment   how close each variant sits to lambda2/lambda3
  icc          between-visit reliability of each variant, both cohorts
  age          the age association of each variant, both cohorts
  beyond       the age association after partialling the ratio out, which is
               the claim that nothing survives it

Run the three cohorts-by-threshold first. Each takes a few hours and flushes
every 50 sessions, so a killed run resumes by rerunning the same command:

    ALPS_FA_MIN=0.15 python measured_pvs_axis.py --cohort hcpa --all-sessions
    ALPS_FA_MIN=0.25 python measured_pvs_axis.py --cohort hcpa --all-sessions
    ALPS_FA_MIN=0.15 python measured_pvs_axis.py --cohort dlbs
    ALPS_FA_MIN=0.25 python measured_pvs_axis.py --cohort dlbs

then

    python fa_threshold_sweep.py

which reads whichever of the six files exist, reports the rest as missing, and
writes fa_threshold_sweep.csv one row at a time.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
OUT = HERE / "fa_threshold_sweep.csv"
FIELDS = ["cohort", "fa_min", "variant", "measure", "value", "n"]
THRESHOLDS = [0.15, 0.20, 0.25]
VARIANTS = ["classic", "cross", "v2_slab", "anat_x", "pv_perp"]


def path_for(cohort: str, fa: float) -> Path:
    base = ("measured_pvs_axis_hcpa_b1500_all" if cohort == "hcpa"
            else "measured_pvs_axis_dlbs")
    sfx = "" if abs(fa - 0.2) < 1e-9 else f"_fa{fa:g}"
    return HERE / f"{base}{sfx}.csv"


def icc11(d: pd.DataFrame, col: str) -> tuple[float, int]:
    """ICC(1,1) over participants with two or more usable sessions."""
    s = d[["Subject_ID", col]].dropna()
    s = s[s.Subject_ID.isin(s.Subject_ID.value_counts()[lambda x: x >= 2].index)]
    if len(s) < 20:
        return float("nan"), len(s)
    g = s.groupby("Subject_ID")[col]
    k = g.size().mean()
    msb = g.mean().var(ddof=1) * k
    msw = g.var(ddof=1).mean()
    if not np.isfinite(msb) or not np.isfinite(msw) or msb + (k - 1) * msw == 0:
        return float("nan"), len(s)
    return float((msb - msw) / (msb + (k - 1) * msw)), int(s.Subject_ID.nunique())


def partial_r(x, y, z) -> float:
    """Correlation of x and y with z regressed out of both."""
    ok = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    x, y, z = x[ok], y[ok], z[ok]
    if len(x) < 20:
        return float("nan")
    Z = np.column_stack([np.ones(len(z)), z])

    def rz(v):
        b, *_ = np.linalg.lstsq(Z, v, rcond=None)
        return v - Z @ b
    return float(np.corrcoef(rz(x), rz(y))[0, 1])


def main() -> None:
    new = not OUT.exists()
    fh = OUT.open("a", newline="", encoding="utf-8")
    w = csv.DictWriter(fh, FIELDS)
    if new:
        w.writeheader()

    def emit(**row):
        w.writerow(row)
        fh.flush()
        print(f"  {row['cohort']:<6s} FA>={row['fa_min']:<5} "
              f"{row['variant']:<9s} {row['measure']:<11s} "
              f"{row['value']:+.3f}  n={row['n']}")

    missing = []
    for cohort in ("hcpa", "dlbs"):
        for fa in THRESHOLDS:
            f = path_for(cohort, fa)
            if not f.exists():
                missing.append(f"  ALPS_FA_MIN={fa:g} ... -> {f.name}")
                continue
            d = pd.read_csv(f)
            d["Subject_ID"] = d.Subject_ID.astype(str)
            print(f"\n{f.name}: {len(d)} sessions")
            for v in VARIANTS:
                if v not in d.columns:
                    continue
                if {"Age", v} <= set(d.columns):
                    s = d[["Age", v]].dropna()
                    emit(cohort=cohort, fa_min=fa, variant=v, measure="age",
                         value=float(stats.pearsonr(s.Age, s[v])[0]),
                         n=len(s))
                icc, n = icc11(d, v)
                if icc == icc:
                    emit(cohort=cohort, fa_min=fa, variant=v, measure="icc",
                         value=icc, n=n)
                if "pv_perp" in d.columns and v != "pv_perp":
                    s = d[["Age", v, "pv_perp"]].dropna()
                    emit(cohort=cohort, fa_min=fa, variant=v,
                         measure="attainment",
                         value=float((s[v] / s.pv_perp).median()), n=len(s))
                    emit(cohort=cohort, fa_min=fa, variant=v,
                         measure="beyond",
                         value=partial_r(s[v].to_numpy(float),
                                         s.Age.to_numpy(float),
                                         s.pv_perp.to_numpy(float)),
                         n=len(s))
    fh.close()
    if missing:
        print("\n  not yet computed, run these first:")
        for m in missing:
            print(m)
    print(f"\n  wrote {OUT.name}")


if __name__ == "__main__":
    main()
