"""
Does the off-tract tissue inside the association ROI actually matter?

roi_placement_quality.py established that about a fifth of the association
sphere is left-right oriented in essentially every session of both cohorts, and
that this is not head tilt (the orientation-free measure tracks the scanner-axis
one). That is a real property of the region, and it is what the DEC quality
control images show. The question this answers is whether it changes anything.

Four things are checked:
  1. how much the index moves when the off-tract voxels are removed
  2. whether reliability improves, since heterogeneous tissue could add noise
  3. whether the age association changes
  4. whether the off-tract fraction ITSELF varies with age, which would make it
     a compositional confound of the same kind as region volume

(4) is the one that would matter most. If the tissue composition of the region
changes with age, then an age association measured in that region is partly a
change in what is being averaged rather than a change in diffusion.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from estimator_variants import variance_components

HERE = Path(__file__).resolve().parent
COHORTS = [("HCP-A", "roi_placement_quality_hcpa_b1500.csv"),
           ("DLBS", "roi_placement_quality_dlbs.csv")]


def clustered_r(d, xcol, ycol, n_boot=2000, seed=20260809):
    """Correlation with a participant-clustered bootstrap interval.

    Resampling is done on precomputed row-index arrays rather than by
    concatenating DataFrame slices, which is what makes 2000 draws feasible
    at n=628 participants.
    """
    sub = d[[xcol, ycol, "Subject_ID"]].dropna().reset_index(drop=True)
    x = sub[xcol].to_numpy(float)
    y = sub[ycol].to_numpy(float)
    r = stats.pearsonr(x, y)[0]
    groups = [np.asarray(v) for v in sub.groupby("Subject_ID").indices.values()]
    rng = np.random.default_rng(seed)
    n_g = len(groups)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, n_g, size=n_g)
        idx = np.concatenate([groups[i] for i in pick])
        xs, ys = x[idx], y[idx]
        xs = xs - xs.mean(); ys = ys - ys.mean()
        den = np.sqrt((xs**2).sum() * (ys**2).sum())
        boots[b] = (xs*ys).sum()/den if den > 0 else np.nan
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return r, lo, hi


for name, fn in COHORTS:
    p = HERE / fn
    if not p.exists():
        print(f"{name}: {fn} missing, skipped\n")
        continue
    d = pd.read_csv(p)
    lon = d[d.Subject_ID.isin(d.Subject_ID.value_counts()[lambda s: s >= 2].index)]
    print("=" * 68)
    print(f"{name}: {len(d)} sessions, {d.Subject_ID.nunique()} participants")
    print("=" * 68)

    print("\n1. how far the index moves when off-tract voxels are dropped")
    for a, b in (("classic", "classic_screened"), ("refined_slab", "refined_slab_screened")):
        sub = d[[a, b]].dropna()
        ch = 100 * (sub[b] - sub[a]) / sub[a]
        print(f"   {a:<14s} {sub[a].mean():.4f} -> {sub[b].mean():.4f}   "
              f"mean {ch.mean():+.2f}%  sd {ch.std():.2f}  |max| {ch.abs().max():.1f}%")

    print("\n2. reliability, before and after screening")
    for c in ("classic", "classic_screened", "refined_slab", "refined_slab_screened"):
        s = lon.dropna(subset=[c])
        vc = variance_components(s, c)
        print(f"   {c:<24s} ICC {vc['icc']:.3f}   var_within {vc['var_within']:.6f}")

    print("\n3. association with age")
    for c in ("classic", "classic_screened", "refined_slab", "refined_slab_screened"):
        s = d[["Age", c]].dropna()
        r, pv = stats.pearsonr(s["Age"], s[c])
        print(f"   {c:<24s} r {r:+.3f}  p {pv:.2e}")

    print("\n4. does the off-tract fraction itself vary with age?")
    for c in ("slf_off_tract", "slf_red", "scr_off_tract"):
        r, lo, hi = clustered_r(d, "Age", c)
        star = "  <-- varies with age" if lo > 0 or hi < 0 else ""
        print(f"   {c:<16s} vs age  r {r:+.3f}  95% CI [{lo:+.3f},{hi:+.3f}]{star}")

    print("\n   off-tract fraction vs the index value")
    for c in ("classic", "refined_slab"):
        r, lo, hi = clustered_r(d, "slf_off_tract", c)
        print(f"   slf_off_tract vs {c:<14s} r {r:+.3f}  95% CI [{lo:+.3f},{hi:+.3f}]")
    print()
