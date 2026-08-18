"""
What the hemisphere-pairing fix changed, on identical sessions.

The bug: these volumes have a negative x scale in the affine, so voxel index
< mid is world x > 0, the RIGHT hemisphere. That was paired with the _L tract
labels, which sit at world x < 0. The refined index therefore measured
diffusivity in one hemisphere while estimating its measurement axes from the
other. The classic index never touches the labels and is unaffected, which makes
it a useful internal control here: if classic moves at all, something other than
the pairing changed.

Compares the pre-fix and post-fix outputs on the sessions present in both, so
the difference is the pairing and not a change of cohort.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from estimator_variants import variance_components

HERE = Path(__file__).resolve().parent
PRE = HERE / "_backup_pre_variants"

PAIRS = [("HCP-A", PRE / "decoupled_roi_hcpa_b1500_PREHEMIFIX.csv",
          HERE / "decoupled_roi_hcpa_b1500.csv"),
         ("DLBS", PRE / "decoupled_roi_dlbs_PREHEMIFIX.csv",
          HERE / "decoupled_roi_dlbs.csv")]
COLS = ["classic", "refined_sphere", "refined_slab", "refined_whole"]


def summarise(d, col):
    lon = d[d.Subject_ID.isin(d.Subject_ID.value_counts()[lambda s: s >= 2].index)]
    lon = lon.dropna(subset=[col])
    vc = variance_components(lon, col)
    sub = d[["Age", col]].dropna()
    r = stats.pearsonr(sub.Age, sub[col])[0]
    return vc["icc"], vc["var_within"], r


for name, pre_p, post_p in PAIRS:
    if not (pre_p.exists() and post_p.exists()):
        print(f"{name}: missing input, skipped\n")
        continue
    pre, post = pd.read_csv(pre_p), pd.read_csv(post_p)
    key = ["Subject_ID", "Visit"]
    for d in (pre, post):
        d["Visit"] = d["Visit"].astype(str)
    common = pre.merge(post[key], on=key, how="inner")[key]
    pre = pre.merge(common, on=key); post = post.merge(common, on=key)
    print("=" * 74)
    print(f"{name}: {len(common)} sessions common to both runs")
    print("=" * 74)
    print(f"{'column':<16s} {'ICC pre':>8s} {'ICC post':>9s} "
          f"{'r age pre':>10s} {'r age post':>11s} {'median |d|%':>12s}")
    for c in COLS:
        if c not in pre or c not in post:
            continue
        i0, v0, r0 = summarise(pre, c)
        i1, v1, r1 = summarise(post, c)
        both = pre[[*key, c]].merge(post[[*key, c]], on=key, suffixes=("_pre", "_post")).dropna()
        dpc = 100 * (both[f"{c}_post"] - both[f"{c}_pre"]).abs() / both[f"{c}_pre"]
        print(f"{c:<16s} {i0:>8.3f} {i1:>9.3f} {r0:>10.3f} {r1:>11.3f} {dpc.median():>11.2f}%")
    print()
