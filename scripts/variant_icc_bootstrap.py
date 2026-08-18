"""Are the reliability differences between variants real, or sampling noise?

Table 4 reports an ICC per variant per cohort, and the differences between them
get read as a ranking. On 156 DLBS participants the marginal confidence
intervals are wide enough to overlap almost completely, which invites the
conclusion that none of the differences mean anything. That conclusion would be
wrong, and the reason is worth stating: every variant is computed from the same
voxels of the same sessions, so the comparisons are paired. Overlapping marginal
intervals are not a test of a paired difference.

This bootstraps the difference directly, resampling participants and recomputing
both ICCs within each resample.

What it settles, on DLBS:

    anat_x - refined    +0.061  [+0.012, +0.108]  p = 0.015   real
    anat_x - classic    -0.078  [-0.126, -0.023]  p = 0.005   real
    anat_x - vecreg     +0.023  [-0.020, +0.066]  p = 0.30    not established
    classic - vecreg    +0.101  [+0.033, +0.165]  p = 0.006   real

So the cross-product penalty is real: taking the perivascular axis from a
registration, which estimates nothing from the tensor, is more reliable than
compounding two estimated tract directions. The resampling penalty vecreg pays
is also real. But anat_x is not more reliable than vecreg, and any claim that it
is comes from comparing point estimates rather than testing them.

    python variant_icc_bootstrap.py

Reads anatomical_x_variant_dlbs.csv and the vecreg values, both of which must
exist first. Writes variant_icc_bootstrap.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import atomic_io  # noqa: F401  writes become atomic on import

HERE = Path(__file__).resolve().parent
DIFF = HERE.parent.parent / "diffusion"


def icc11(vals, g, ngroups):
    """ICC(1,1) from an unbalanced one-way random-effects ANOVA, on raw arrays.

    Takes group codes rather than a frame so the bootstrap can call it tens of
    thousands of times without pandas overhead.
    """
    n = np.bincount(g, minlength=ngroups).astype(float)
    keep = n > 0
    s = np.bincount(g, weights=vals, minlength=ngroups)
    mi = np.divide(s, n, out=np.zeros_like(s), where=n > 0)
    N, a, grand = vals.size, int(keep.sum()), vals.mean()
    if a < 3:
        return np.nan
    msb = float((n[keep] * (mi[keep] - grand) ** 2).sum() / (a - 1))
    msw = float((((vals - mi[g]) ** 2).sum()) / max(N - a, 1))
    n0 = (N - (n[keep] ** 2).sum() / N) / (a - 1)
    den = msb + (n0 - 1) * msw
    return (msb - msw) / den if den else np.nan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    a = pd.read_csv(HERE / "anatomical_x_variant_dlbs.csv")
    m = (a.groupby(["Subject_ID", "Visit"])
          .agg(anat_x=("anat_x", "mean"), refined=("refined", "mean"),
               classic=("classic", "mean"), scanner_x=("scanner_x", "mean"))
          .reset_index())
    v = pd.read_csv(DIFF / "DLBS" / "dlbs_vecreg_alps.csv")
    v = (v[v.status == "ok"][["Subject_ID", "Session", "vecreg_classic_Avg"]]
         .rename(columns={"Session": "Visit", "vecreg_classic_Avg": "vecreg"}))
    for f in (m, v):
        f["Subject_ID"] = f.Subject_ID.astype(str)
        f["Visit"] = f.Visit.astype(str)
    j = m.merge(v, on=["Subject_ID", "Visit"])

    rep = j[j.Subject_ID.isin(j.Subject_ID.value_counts()[lambda s: s >= 2].index)].copy()
    codes, uniq = pd.factorize(rep.Subject_ID)
    G = len(uniq)
    idx = [np.flatnonzero(codes == i) for i in range(G)]
    cols = ["classic", "anat_x", "refined", "scanner_x", "vecreg"]
    V = {c: rep[c].to_numpy() for c in cols}
    pt = {c: icc11(V[c], codes, G) for c in cols}

    print(f"DLBS, {len(j)} sessions paired across all variants, "
          f"{G} participants with repeat visits\n")
    for c in cols:
        print(f"  {c:10s} ICC {pt[c]:.3f}")

    pairs = [("anat_x", "refined"), ("anat_x", "classic"), ("anat_x", "vecreg"),
             ("anat_x", "scanner_x"), ("refined", "vecreg"), ("classic", "vecreg")]
    rng = np.random.default_rng(args.seed)
    D = {p: np.empty(args.draws) for p in pairs}
    for b in range(args.draws):
        pick = rng.integers(0, G, G)
        sel, gg = [], []
        for k, p in enumerate(pick):
            ii = idx[p]
            sel.append(ii)
            gg.append(np.full(ii.size, k))
        sel, gg = np.concatenate(sel), np.concatenate(gg)
        ic = {c: icc11(V[c][sel], gg, G) for c in cols}
        for x, y in pairs:
            D[(x, y)][b] = ic[x] - ic[y]

    rows = []
    print(f"\npaired bootstrap of the ICC difference, {args.draws} resamples of participants")
    for x, y in pairs:
        dd = D[(x, y)]
        lo, hi = np.nanpercentile(dd, [2.5, 97.5])
        p = 2 * min((dd <= 0).mean(), (dd >= 0).mean())
        verdict = "real" if p < 0.05 else "not established"
        rows.append(dict(a=x, b=y, diff=pt[x] - pt[y], lo=lo, hi=hi, p=p))
        print(f"  {x:9s} - {y:10s} {pt[x] - pt[y]:+.3f}   "
              f"95% CI [{lo:+.3f}, {hi:+.3f}]   p={p:.4f}   {verdict}")
    pd.DataFrame(rows).to_csv(HERE / "variant_icc_bootstrap.csv", index=False)
    print("\n  Marginal intervals on these ICCs overlap heavily. They are not the")
    print("  test: the variants share sessions, so the paired difference is.")


if __name__ == "__main__":
    main()
