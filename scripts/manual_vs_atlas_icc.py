"""Hand-drawn against atlas-placed reliability, on one estimator.

The manuscript quotes DTI-ALPS between-visit reliability from two pipelines.
reconciliation_table.py reports the DLBS classic index at 0.5945 using
variance_components, which is the figure Table 4, the placement section and the
limitations all carry. reliability_analysis.py reports 0.559 for the same
quantity using its own estimator with bootstrap intervals. Both are correct
against their own definitions, and a sentence comparing hand-drawn with atlas
placement cannot take one number from each.

This computes both sides with variance_components, the estimator the rest of the
manuscript uses, so the comparison is like for like.

    python manual_vs_atlas_icc.py

Writes manual_vs_atlas_icc.csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import atomic_io  # noqa: F401  writes become atomic on import
from estimator_variants import variance_components

HERE = Path(__file__).resolve().parent
DIFF = HERE.parent.parent / "diffusion"


def repeats(d, col):
    d = d.dropna(subset=[col])
    keep = d.Subject_ID.value_counts()[lambda s: s >= 2].index
    return d[d.Subject_ID.isin(keep)]


def main() -> None:
    argparse.ArgumentParser().parse_args()

    man = pd.read_csv(DIFF / "HCP" / "lifespan_alps_results.csv")
    man = man.rename(columns={"Session": "Visit", "Traditional_Avg": "classic"})
    man["Subject_ID"] = man.Subject_ID.astype(str)
    man["Visit"] = man.Visit.astype(str)

    atlas = pd.read_csv(HERE / "measured_pvs_axis_dlbs.csv")
    atlas["Subject_ID"] = atlas.Subject_ID.astype(str)
    atlas["Visit"] = atlas.Visit.astype(str)

    rows = []
    for label, d in (("hand-drawn", man), ("atlas", atlas)):
        lon = repeats(d, "classic")
        icc = float(variance_components(lon, "classic")["icc"])
        rows.append(dict(placement=label, icc=round(icc, 4), sessions=len(lon),
                         participants=int(lon.Subject_ID.nunique())))

    # and the same on only the sessions the hand-drawn set covers, which is the
    # narrower comparison a reader might expect from one sentence
    keys = set(zip(man.Subject_ID, man.Visit))
    sub = atlas[[(a, b) in keys for a, b in zip(atlas.Subject_ID, atlas.Visit)]]
    lon = repeats(sub, "classic")
    if len(lon) > 3:
        rows.append(dict(placement="atlas, hand-drawn sessions only",
                         icc=round(float(variance_components(lon, "classic")["icc"]), 4),
                         sessions=len(lon), participants=int(lon.Subject_ID.nunique())))

    out = pd.DataFrame(rows)
    out.to_csv(HERE / "manual_vs_atlas_icc.csv", index=False)

    print("Between-visit ICC of the classic index, variance_components estimator\n")
    for r in out.itertuples():
        print(f"   {r.placement:34s} {r.icc:+.4f}   "
              f"{r.sessions:4d} sessions, {r.participants:3d} participants")
    print("\n   The atlas figure is the one the rest of the manuscript quotes.")


if __name__ == "__main__":
    main()
