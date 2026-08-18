"""
Is the 45% drop what adjustment does mechanically, or evidence of confounding?

A reasonable objection to the head-pose result is that adjusting for any
covariate shrinks a coefficient, so a 45% reduction would be unremarkable. That
is not how partial regression works: a covariate uncorrelated with both the
predictor and the outcome leaves the standardised coefficient essentially
unchanged, and can even raise it by removing residual variance. But the
objection deserves a test rather than an argument.

Two controls:

  permutation  Shuffle the head-pose values across participants, breaking their
               link to both age and the index while preserving their marginal
               distribution exactly, then re-run the same adjustment. If the
               drop were mechanical it would survive shuffling.

  HCP-A        The same adjustment in a cohort where preprocessing has already
               removed head position. Real pose values, real distribution, but
               no relation to age. This is the built-in negative control.

Reported for the classic index in DLBS, which is where the 45% is claimed.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
N_PERM = 2000


def load(head, alps):
    h, a = pd.read_csv(HERE / head), pd.read_csv(HERE / alps)
    for d in (h, a):
        d["Subject_ID"] = d.Subject_ID.astype(str)
        d["Visit"] = d.Visit.astype(str)
    m = (h.merge(a, on=["Subject_ID", "Visit"])
           .sort_values(["Subject_ID", "Visit"]).groupby("Subject_ID")
           .first().reset_index())
    return m.dropna(subset=["Age", "pitch", "total", "classic"])


def z(v):
    v = np.asarray(v, float)
    return (v - v.mean()) / v.std(ddof=1)


def absorbed(d, col, pitch, total):
    """Percent of the standardised age coefficient removed by adjusting for pose."""
    y, age = z(d[col]), z(d.Age)
    b0 = np.linalg.lstsq(np.column_stack([np.ones(len(d)), age]), y, rcond=None)[0][1]
    X1 = np.column_stack([np.ones(len(d)), age, z(pitch), z(total)])
    b1 = np.linalg.lstsq(X1, y, rcond=None)[0][1]
    return 100 * (1 - abs(b1) / abs(b0))


for tag, hf, af in (("DLBS (oblique)", "head_rotation_dlbs.csv",
                     "measured_pvs_axis_dlbs.csv"),
                    ("HCP-A (aligned)", "head_rotation_hcpa.csv",
                     "measured_pvs_axis_hcpa_b1500_all.csv")):
    d = load(hf, af)
    obs = absorbed(d, "classic", d.pitch.abs(), d.total)
    rng = np.random.default_rng(20260811)
    null = np.array([
        absorbed(d, "classic",
                 d.pitch.abs().to_numpy()[rng.permutation(len(d))],
                 d.total.to_numpy()[rng.permutation(len(d))])
        for _ in range(N_PERM)])
    p = float((null >= obs).mean())
    print(f"{tag}: n={len(d)}")
    print(f"   observed absorbed          {obs:7.2f}%")
    print(f"   shuffled pose, mean        {null.mean():7.2f}%   "
          f"95th pct {np.percentile(null, 95):6.2f}%   max {null.max():6.2f}%")
    print(f"   permutation p              {p:.4f}")
    print(f"   -> adjustment is {'NOT mechanical' if p < 0.05 else 'indistinguishable from mechanical'}")
    print()

print("A mechanical effect would give a shuffled distribution centred near the")
print("observed value. Centred near zero instead means the reduction requires")
print("head pose to actually covary with age and with the index.")
