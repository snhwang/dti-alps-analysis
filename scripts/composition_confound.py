"""
Is part of the ALPS age effect a change in what is inside the region?

Two facts from placement_consequences.py set this up.

  * The fraction of the association region that is left-right oriented predicts
    the index value, r = +0.32 to +0.36 in both cohorts and for both variants.
    That is mechanically unsurprising: those voxels have high diffusivity along
    x, and x-diffusivity is the numerator of the ratio.
  * That same fraction varies with age (HCP-A: slf_red r = -0.137, and
    scr_off_tract r = +0.165, both with clustered intervals excluding zero).

Put together, they describe a path from age to the index that does not run
through perivascular diffusion at all. If the region's tissue composition
changes with age, an age association measured in that region is partly a change
in what is being averaged.

This is the same species of confound as the region-volume result already in the
manuscript, and it is tested the same way: refit the standardised age
coefficient with the composition measures entered as covariates, and report how
much of the coefficient they absorb, for each variant.

The caveat that applies to the volume analysis applies here too and is reported
alongside. If age genuinely reduces the amount of coherently oriented tissue,
composition lies on the causal path rather than beside it, and adjusting for it
removes real signal. The comparison BETWEEN variants is the interpretable part:
both are exposed to the identical composition change in the identical voxels, so
a variant that loses more of its coefficient is more sensitive to it.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

HERE = Path(__file__).resolve().parent
COHORTS = [("HCP-A", "roi_placement_quality_hcpa_b1500.csv"),
           ("DLBS", "roi_placement_quality_dlbs.csv")]
INDICES = ["classic", "refined_slab"]
COMP = ["slf_off_tract", "scr_off_tract"]


def zscore(a):
    a = np.asarray(a, float)
    return (a - a.mean()) / a.std(ddof=1)


def ols_beta(y, Xcols):
    """Standardised coefficient on the first column, with clustered SE ignored
    here because only the point estimate is compared."""
    X = np.column_stack([np.ones(len(y))] + [zscore(c) for c in Xcols])
    beta, *_ = np.linalg.lstsq(X, zscore(y), rcond=None)
    return float(beta[1])


# The response letter quotes the absorbed fractions, so they need to survive
# outside this script's stdout. Printing a number that a document then repeats
# is how the two drift apart with nothing to notice.
rows = []

for name, fn in COHORTS:
    p = HERE / fn
    if not p.exists():
        print(f"{name}: missing {fn}\n")
        continue
    d = pd.read_csv(p).dropna(subset=["Age"] + INDICES + COMP)
    print("=" * 72)
    print(f"{name}: {len(d)} sessions, {d.Subject_ID.nunique()} participants")
    print("=" * 72)
    print(f"{'index':<16s} {'beta_age':>9s} {'+composition':>13s} {'absorbed':>10s}")
    for c in INDICES:
        raw = ols_beta(d[c], [d["Age"]])
        adj = ols_beta(d[c], [d["Age"]] + [d[k] for k in COMP])
        pct = 100 * (1 - adj / raw)
        print(f"{c:<16s} {raw:>9.3f} {adj:>13.3f} {pct:>9.1f}%")
        rows.append(dict(cohort=name, index=c, n_sessions=len(d),
                         n_participants=int(d.Subject_ID.nunique()),
                         beta_age=raw, beta_age_adj=adj, pct_absorbed=pct))
    pd.DataFrame(rows).to_csv(HERE / "composition_confound.csv", index=False)

    print("\n  composition measures against age")
    for k in COMP + ["slf_red"]:
        r = np.corrcoef(d["Age"], d[k])[0, 1]
        print(f"    {k:<16s} r {r:+.3f}")

    print("\n  composition measures against the index")
    for c in INDICES:
        for k in COMP:
            r = np.corrcoef(d[c], d[k])[0, 1]
            print(f"    {k:<16s} vs {c:<14s} r {r:+.3f}")
    print()
