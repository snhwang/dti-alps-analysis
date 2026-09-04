"""Find numbers in the manuscript left over from the warped-mask analysis.

The numerical verifier guards a few hundred specific quantities. The manuscript
contains several thousand numbers. When the region placement changed, every
index-derived value moved, and any that the verifier does not happen to guard
could stay behind silently. Several did, including one that left the Results
contradicting the Abstract about the paper's only surviving positive result.

This finds them by working backwards. Both placements are on disk, so for every
quantity it can compute, this evaluates it twice, then searches the manuscript
for the warped-mask value. A hit is a number that was never updated, unless it
happens to coincide with something else, which the printed context lets a human
judge.

It cannot find everything. A stale value that rounds to the same three decimals
as its replacement is invisible here, and so is anything derived by a script
that was never re-run at all. Those need the timestamp audit instead.

    python find_stale_numbers.py
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
TEX = HERE.parent / "mri_revision.tex"
VARIANTS = ["classic", "cross", "v2_sphere", "v2_slab", "pv_perp", "anat_x"]


def icc(d: pd.DataFrame, col: str) -> float:
    d = d.dropna(subset=[col])
    n = d.groupby("Subject_ID")[col].count()
    d = d[d.Subject_ID.isin(n[n >= 2].index)]
    if len(d) < 20:
        return np.nan
    g = d.groupby("Subject_ID")[col]
    within = float(((d[col] - g.transform("mean")) ** 2).sum()
                   / max(len(d) - g.ngroups, 1))
    between = float(g.mean().var(ddof=1))
    tot = between + within
    return between / tot if tot > 0 else np.nan


def quantities(d: pd.DataFrame) -> dict:
    """Every scalar this table can produce, keyed by a readable name."""
    out = {}
    one = (d.sort_values(["Subject_ID", "Visit"])
            .groupby("Subject_ID").first().reset_index())
    for v in VARIANTS:
        if v not in d.columns or d[v].notna().sum() < 30:
            continue
        out[f"ICC {v}"] = icc(d, v)
        s = d[[v, "Age"]].dropna()
        out[f"age r all sessions {v}"] = float(stats.pearsonr(s[v], s.Age)[0])
        s = one[[v, "Age"]].dropna()
        out[f"age r one per participant {v}"] = float(stats.pearsonr(s[v], s.Age)[0])
        if v != "pv_perp" and "pv_perp" in d.columns:
            s = d[[v, "pv_perp"]].dropna()
            out[f"r with the ratio {v}"] = float(stats.pearsonr(s[v], s.pv_perp)[0])
            out[f"median {v} over the ratio"] = float((d[v] / d.pv_perp).median())
            out[f"bound violations pct {v}"] = float(
                (s[v] > s.pv_perp + 1e-9).mean() * 100)
        out[f"mean {v}"] = float(d[v].mean())
    return out


def main() -> None:
    argparse.ArgumentParser().parse_args()
    tex = TEX.read_text(encoding="utf-8")
    flat = " ".join(tex.split())
    hits = 0

    for cohort, stem in (("HCP-A", "measured_pvs_axis_hcpa_b1500_all"),
                         ("DLBS", "measured_pvs_axis_dlbs")):
        # "new" is the canonical placement, whatever it currently is, and
        # "old" is the superseded one. The pair was inverted when the warped
        # mask became canonical again, so it is named by role rather than by
        # placement and the detector keeps working in either direction.
        new = pd.read_csv(HERE / f"{stem}.csv")
        old = pd.read_csv(HERE / f"{stem}_sphere5.csv")
        for f in (new, old):
            f["Subject_ID"] = f.Subject_ID.astype(str)
            f["Visit"] = f.Visit.astype(str)
        qn, qo = quantities(new), quantities(old)

        print(f"\n{'=' * 70}\n{cohort}\n{'=' * 70}")
        for k in sorted(qo):
            o, n = qo[k], qn.get(k, np.nan)
            if np.isnan(o) or np.isnan(n):
                continue
            so, sn = f"{o:.3f}", f"{n:.3f}"
            if so == sn:
                continue                      # indistinguishable at 3 dp
            for pat in (f"${so}$", f"${so.lstrip('0')}$", so):
                if pat in flat:
                    ctx = [flat[max(0, m.start() - 95):m.end() + 65]
                           for m in re.finditer(re.escape(pat), flat)]
                    ctx = [c for c in ctx if "bibitem" not in c and "doi.org" not in c]
                    if not ctx:
                        continue
                    print(f"\n  STALE? {k}")
                    print(f"     superseded {so}  ->  canonical {sn}   ({len(ctx)} place(s))")
                    for c in ctx[:2]:
                        print(f"     ...{c}...")
                    hits += 1
                    break

    print(f"\n{'=' * 70}")
    print(f"{hits} candidate stale values. Each needs a look: the warped value may")
    print("also be a legitimate number from some other quantity entirely.\n")


if __name__ == "__main__":
    main()
