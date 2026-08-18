"""
R4.2: template reorientation (FSL vecreg) against the closed-form correction, DLBS.

Reviewer 4 asked for the Tatekawa comparison to be implemented rather than cited.
This scores vecreg on the same endpoints as every other variant here: reliability
across visits, sensitivity to genuine repositioning, and the age association.

One caveat carries through to the interpretation. vecreg resamples into template
space, and resampling smooths, which lowers within-participant variance for
reasons unrelated to orientation. Any reliability advantage it shows is therefore
an upper bound on the reorientation benefit.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import atomic_io  # noqa: F401  writes become atomic on import

BASE = Path(__file__).resolve().parent
DIFF = BASE.parent.parent / "diffusion"


def icc11(values, groups):
    """ICC(1,1) from an unbalanced one-way random-effects ANOVA, closed form."""
    df = pd.DataFrame({"v": values, "g": groups}).dropna()
    k = df.groupby("g").size()
    df = df[df.g.isin(k[k >= 2].index)]
    if df.empty:
        return np.nan, 0, 0
    n, a = len(df), df.g.nunique()
    gm, means = df.v.mean(), df.groupby("g").v.mean()
    sizes = df.groupby("g").size()
    msb = (sizes * (means - gm) ** 2).sum() / (a - 1)
    msw = df.groupby("g").v.apply(lambda s: ((s - s.mean()) ** 2).sum()).sum() / (n - a)
    k0 = (n - (sizes ** 2).sum() / n) / (a - 1)
    icc = (msb - msw) / (msb + (k0 - 1) * msw)
    return float(icc), a, n


def wcv(values, groups):
    """Within-participant coefficient of variation, %."""
    df = pd.DataFrame({"v": values, "g": groups}).dropna()
    k = df.groupby("g").size()
    df = df[df.g.isin(k[k >= 2].index)]
    within = df.groupby("g").v.apply(lambda s: ((s - s.mean()) ** 2).sum())
    dof = len(df) - df.g.nunique()
    return float(np.sqrt(within.sum() / dof) / df.v.mean() * 100)


vec = pd.read_csv(DIFF / "DLBS" / "dlbs_vecreg_alps.csv")
vec = vec[vec.status == "ok"][["Subject_ID", "Session", "vecreg_classic_Avg"]]
vec = vec.rename(columns={"Session": "Visit", "vecreg_classic_Avg": "vecreg"})

dec = pd.read_csv(BASE / "decoupled_roi_dlbs.csv")

# Visit labels differ in type between the two pipelines; normalise before merging.
for d in (vec, dec):
    d["Visit"] = d["Visit"].astype(str).str.extract(r"(\d+)").astype(float)
d = dec.merge(vec, on=["Subject_ID", "Visit"], how="inner")
print(f"merged {len(d)} sessions, {d.Subject_ID.nunique()} participants\n")

VARIANTS = ["classic", "refined_slab", "vecreg"]
LABEL = {"classic": "Classic", "refined_slab": "Refined (slab)", "vecreg": "vecreg"}

rows = []
for v in VARIANTS:
    icc, a, n = icc11(d[v], d.Subject_ID)
    r = d[["Age", v]].dropna().corr().iloc[0, 1]
    rows.append({"variant": LABEL[v], "ICC": round(icc, 3), "wCV_pct": round(wcv(d[v], d.Subject_ID), 2),
                 "n_subj": a, "n_sess": n, "age_r": round(r, 3)})
out = pd.DataFrame(rows)
print(out.to_string(index=False))

# Reliability relative to classic, the quantity the paper reports as a penalty.
base = out.loc[out.variant == "Classic", "wCV_pct"].iloc[0]
out["var_ratio_vs_classic"] = (out.wCV_pct / base) ** 2
print("\nwithin-participant variance relative to classic:")
for _, r in out.iterrows():
    print(f"  {r.variant:<16s} {r.var_ratio_vs_classic:.2f}x")

out.to_csv(BASE / "vecreg_summary.csv", index=False)
print(f"\nwrote {BASE / 'vecreg_summary.csv'}")
