"""
Every available phenotype, not a hand-picked nine.

The earlier test used nine outcomes chosen by hand, which risks both missing
something and looking like the nine were chosen after the fact. This sweeps
every numeric variable in the AABC subject table that has enough coverage to
support a correlation, and corrects across the whole set.

Objective sleep is the addition that matters most. The previous run used PSQI,
which is self-reported. The table also carries actigraphy: total sleep time,
sleep efficiency and wake after sleep onset. The DTI-ALPS literature rests
heavily on sleep, so an objective measure is the fairer test of it.

Age and sex are adjusted throughout. The unadjusted comparison was shown to be
uninterpretable for choosing between variants, because a variant with a steeper
age slope inherits an advantage on every age-related outcome without carrying
any information about the outcome itself.

Columns that are identifiers, dates, visit counters or age itself are excluded
by name. Everything else with at least MIN_N matched participants is tested.
Benjamini-Hochberg is applied across the whole sweep, per variant.

Usage:
    python phenotype_sweep.py
"""

from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path

import os

import numpy as np
import pandas as pd

import atomic_io  # noqa: F401  writes become atomic on import
from scipy import stats

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from hemisphere_age import williams

HERE = Path(__file__).resolve().parent
DIFF = HERE.parent.parent / "diffusion"
AABC = DIFF / "HCP" / "AABC2_subjects_2026_02_05_14_29_11.csv"
VARIANTS = ["classic", "cross", "v2_sphere", "v2_slab", "pv_perp", "anat_x"]
MIN_N = 60
UNADJUSTED = os.environ.get("ALPS_SWEEP_UNADJUSTED", "") == "1"
DROP = re.compile(r"age|days_from|yearquarter|^id|guid|pedid|_nda$|visit|wave|"
                  r"pctcompl|count$|^mr_|qc|scanner|site|^bulk|idps|msmall", re.I)


def bh(p):
    p = np.asarray(p, float)
    ok = ~np.isnan(p)
    q = np.full_like(p, np.nan)
    n = int(ok.sum())
    if n == 0:
        return q
    idx = np.argsort(p[ok])
    ranked = p[ok][idx] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n); out[idx] = np.clip(ranked, 0, 1)
    q[ok] = out
    return q


def main() -> None:
    d = pd.read_csv(HERE / "measured_pvs_axis_hcpa_b1500_all.csv")
    d = d.sort_values(["Subject_ID", "Visit"]).groupby("Subject_ID").first().reset_index()
    d["id_event"] = d.Subject_ID.astype(str) + "_" + d.Visit.astype(str)

    a = pd.read_csv(AABC, low_memory=False)
    a["Subject_ID"] = a.id_event.astype(str).str.split("_").str[0]
    num = [c for c in a.columns
           if pd.api.types.is_numeric_dtype(a[c]) and not DROP.search(c)
           and a[c].notna().sum() >= 100 and a[c].nunique() > 4]
    # Merge on participant, not participant-visit. Phenotype sub-studies run at
    # whichever visit they ran at, so joining on visit silently discarded any
    # measurement taken at a different visit from the diffusion session.
    ph = a.groupby("Subject_ID")[num].first().reset_index()
    sx = a.groupby("Subject_ID")["sex"].first().reset_index()
    m = d.merge(ph, on="Subject_ID", how="inner").merge(sx, on="Subject_ID", how="left")
    m["sex_n"] = (m.sex.astype(str).str.upper().str[0] == "M").astype(float)
    print(f"{len(m)} participants merged; {len(num)} candidate phenotypes\n")

    rows = []
    for c in num:
        s = m[[c, "Age", "sex_n"] + VARIANTS].copy()
        s[c] = pd.to_numeric(s[c], errors="coerce")
        s = s.replace([np.inf, -np.inf], np.nan).dropna()
        if len(s) < MIN_N or s[c].nunique() < 5:
            continue
        # The unadjusted arm exists to test one explanation for why this sweep
        # finds so little where the literature finds a great deal: DTI-ALPS falls
        # steeply with age, so any age-related outcome correlates with it whether
        # or not the index carries information about that outcome.
        C = (np.column_stack([np.ones(len(s)), s.Age.to_numpy(float),
                              s.sex_n.to_numpy(float)])
             if not UNADJUSTED else np.ones((len(s), 1)))
        def rz(v):
            b, *_ = np.linalg.lstsq(C, np.asarray(v, float), rcond=None)
            return np.asarray(v, float) - C @ b
        yy = rz(s[c])
        if yy.std() < 1e-12:
            continue
        rec = {"phenotype": c, "n": len(s)}
        res = {}
        for k in VARIANTS:
            xx = rz(s[k])
            if xx.std() < 1e-12:
                res[k] = np.nan; continue
            r = float(np.corrcoef(xx, yy)[0, 1])
            res[k] = r
            dof = len(s) - (2 if UNADJUSTED else 4)
            t = r * np.sqrt(dof / max(1 - r * r, 1e-12))
            rec[k] = r
            rec[f"p_{k}"] = float(2 * (1 - stats.t.cdf(abs(t), dof)))
        if any(np.isnan(v) for v in res.values()):
            continue
        c1, c2 = rz(s["classic"]), rz(s["v2_slab"])
        t, p = williams(res["classic"], res["v2_slab"],
                        float(np.corrcoef(c1, c2)[0, 1]), len(s))
        rec["williams_p"] = p
        rows.append(rec)

    out = pd.DataFrame(rows)
    for k in VARIANTS:
        out[f"q_{k}"] = bh(out[f"p_{k}"].values)
    name = "phenotype_sweep_unadjusted.csv" if UNADJUSTED else "phenotype_sweep.csv"
    out.to_csv(HERE / name, index=False)
    print(f"wrote {name}")
    print(f"tested {len(out)} phenotypes with n >= {MIN_N}\n")

    any_sig = False
    for k in VARIANTS:
        sig = out[out[f"q_{k}"] < 0.05].sort_values(f"p_{k}")
        print(f"{k}: {len(sig)} phenotypes surviving FDR")
        for r in sig.head(12).itertuples():
            print(f"    {getattr(r, 'phenotype'):<42s} n={r.n:4d} "
                  f"r={getattr(r, k):+.3f}  q={getattr(r, f'q_{k}'):.4f}")
        any_sig |= len(sig) > 0

    if not any_sig:
        print("\nNothing survives FDR for any variant.")
        best = out.reindex(out[[f"p_{k}" for k in VARIANTS]].min(axis=1).sort_values().index)
        print("Strongest uncorrected associations, for reference only:")
        for r in best.head(10).itertuples():
            cells = "  ".join(f"{k}={getattr(r, k):+.3f}" for k in VARIANTS)
            print(f"    {getattr(r, 'phenotype'):<42s} n={r.n:4d}  {cells}")

    sleep = out[out.phenotype.str.contains("cobra|psqi", case=False, na=False)]
    if len(sleep):
        print("\nsleep measures specifically:")
        for r in sleep.itertuples():
            cells = "  ".join(f"{k}={getattr(r, k):+.3f}" for k in VARIANTS)
            print(f"    {getattr(r, 'phenotype'):<42s} n={r.n:4d}  {cells}")


if __name__ == "__main__":
    main()
