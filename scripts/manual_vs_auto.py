"""
Do atlas-derived ROIs reproduce the hand-drawn ones?

The submitted manuscript used ROIs drawn by hand on 81 DLBS sessions. Everything
reported in the revision uses atlas-derived spherical ROIs placed automatically.
Reviewer 1 asked for a comparison against an automated pipeline, and the
substitution needs justifying in any case, so this compares the two on the same
sessions rather than on separate cohorts.

Agreement is reported as the correlation between methods, the mean difference,
and the limits of agreement, for the classic and refined indices separately.
Agreement between ROI definitions is expected to be imperfect, since the two
sample different voxels; the question is whether the automated placement
reproduces the hand-drawn measurement closely enough that conclusions drawn from
one carry to the other.

Usage:
    python manual_vs_auto.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import atomic_io  # noqa: F401  writes become atomic on import
from scipy import stats

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
DIFF = HERE.parent.parent / "diffusion"

PAIRS = [("Traditional_Avg", "classic"), ("Refined_Avg", "refined"),
         ("RefinedPlus_Avg", "refined+"), ("ALPS_PAS_Avg", "ALPS-PAS")]


def main() -> None:
    man = pd.read_csv(DIFF / "HCP" / "lifespan_alps_results.csv")
    auto = pd.read_csv(DIFF / "DLBS" / "dlbs_alps_spheres_5mm.csv")
    auto = auto[auto.status == "ok"]

    keep = ["DTI_Session_ID"] + [c for c, _ in PAIRS]
    m = man[keep + ["Age"]].merge(auto[keep], on="DTI_Session_ID",
                                  suffixes=("_man", "_auto"))
    for c, _ in PAIRS:
        m[c + "_man"] = pd.to_numeric(m[c + "_man"], errors="coerce")
        m[c + "_auto"] = pd.to_numeric(m[c + "_auto"], errors="coerce")
    m = m.dropna(subset=[c + s for c, _ in PAIRS for s in ("_man", "_auto")])
    print(f"sessions with both hand-drawn and atlas-derived ROIs: {len(m)}\n")

    print(f"{'index':<10s} {'manual':>8s} {'auto':>8s} {'bias':>8s} {'bias %':>8s} "
          f"{'r':>7s} {'ICC(A,1)':>9s} {'LoA':>18s}")
    rows = []
    for col, name in PAIRS:
        a, b = m[col + "_man"].to_numpy(), m[col + "_auto"].to_numpy()
        d = b - a
        bias, sd = d.mean(), d.std(ddof=1)
        r = float(np.corrcoef(a, b)[0, 1])
        # two-way absolute-agreement ICC for a single measurement
        n = len(a)
        both = np.column_stack([a, b])
        gm = both.mean()
        ms_r = 2 * ((both.mean(axis=1) - gm) ** 2).sum() / (n - 1)
        ms_c = n * ((both.mean(axis=0) - gm) ** 2).sum() / 1
        ms_e = ((both - both.mean(axis=1, keepdims=True)
                 - both.mean(axis=0) + gm) ** 2).sum() / (n - 1)
        icc = (ms_r - ms_e) / (ms_r + ms_e + 2 * (ms_c - ms_e) / n)
        print(f"{name:<10s} {a.mean():8.3f} {b.mean():8.3f} {bias:+8.3f} "
              f"{100*bias/a.mean():+8.2f} {r:7.3f} {icc:9.3f} "
              f"{f'{bias-1.96*sd:+.3f} to {bias+1.96*sd:+.3f}':>18s}")
        rows.append({"index": name, "mean_manual": a.mean(), "mean_auto": b.mean(),
                     "bias": bias, "bias_pct": 100 * bias / a.mean(), "r": r,
                     "icc": icc, "loa_lo": bias - 1.96 * sd, "loa_hi": bias + 1.96 * sd,
                     "n": n})

    print("\nassociation with age, same sessions, each ROI definition:")
    print(f"{'index':<10s} {'r manual':>10s} {'r auto':>10s} {'difference':>11s}")
    for col, name in PAIRS:
        ra = stats.linregress(m["Age"], m[col + "_man"])[2]
        rb = stats.linregress(m["Age"], m[col + "_auto"])[2]
        print(f"{name:<10s} {ra:10.3f} {rb:10.3f} {rb-ra:+11.3f}")

    pd.DataFrame(rows).to_csv(HERE / "manual_vs_auto.csv", index=False)
    print(f"\nWrote {HERE/'manual_vs_auto.csv'}")


if __name__ == "__main__":
    main()
