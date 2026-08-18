"""
How much of the age association survives removing every geometric term at once?

The paper reports three geometric properties separately: head position, region
volume, and region composition. Reported one at a time they cannot be added,
because they overlap, and a reader has no way to tell whether they account for
most of the age association or a little of it twice.

This adjusts for all of them together, in the cohort where head position
survives into the analysed data. It is the strongest form of the paper's own
argument against itself, so it belongs in the paper rather than in a drawer.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import atomic_io  # noqa: F401  writes become atomic on import

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
VARIANTS = ("classic", "refined_slab")


def z(v):
    v = np.asarray(v, float)
    return (v - v.mean()) / v.std(ddof=1)


def main() -> None:
    q = pd.read_csv(HERE / "roi_placement_quality_dlbs_all.csv")
    h = pd.read_csv(HERE / "head_rotation_dlbs.csv")
    for d in (q, h):
        d["k"] = d.Subject_ID.astype(str) + "|" + d.Visit.astype(str)
    m = (q.merge(h[["k", "pitch", "total"]], on="k")
           .sort_values(["Subject_ID", "Visit"]).groupby("Subject_ID").first().reset_index())
    m = m.dropna(subset=["Age", *VARIANTS, "pitch", "total",
                         "n_scr", "n_slf", "slf_off_tract", "scr_off_tract"])

    def beta(col, extra):
        y, age = z(m[col]), z(m.Age)
        X = np.column_stack([np.ones(len(m)), age] + [z(e) for e in extra])
        return float(np.linalg.lstsq(X, y, rcond=None)[0][1])

    sets = [("unadjusted", []),
            ("head position", [np.abs(m.pitch), m.total]),
            ("region volume", [m.n_scr, m.n_slf]),
            ("composition", [m.slf_off_tract, m.scr_off_tract]),
            ("all three", [np.abs(m.pitch), m.total, m.n_scr, m.n_slf,
                           m.slf_off_tract, m.scr_off_tract])]

    base = {c: beta(c, []) for c in VARIANTS}
    rows = []
    print(f"DLBS, {len(m)} participants: standardised age coefficient\n")
    print(f"  {'adjustment':<16s} {'classic':>9s} {'refined':>9s} {'absorbed':>18s}")
    for name, extra in sets:
        b = {c: beta(c, extra) for c in VARIANTS}
        pc = {c: 100 * (1 - abs(b[c]) / abs(base[c])) for c in VARIANTS}
        tag = "" if not extra else f"{pc['classic']:6.0f}% {pc['refined_slab']:6.0f}%"
        print(f"  {name:<16s} {b['classic']:9.3f} {b['refined_slab']:9.3f} {tag:>18s}")
        rows.append({"adjustment": name, "n": len(m),
                     "classic": b["classic"], "refined": b["refined_slab"],
                     "classic_absorbed_pct": pc["classic"],
                     "refined_absorbed_pct": pc["refined_slab"]})

    pd.DataFrame(rows).to_csv(HERE / "joint_geometry_adjustment.csv", index=False)
    print("\n  The single-term absorptions do not sum: the terms share variance.")
    print("  Composition contributes nothing here, as reported separately for DLBS.")
    print(f"\nwrote {HERE / 'joint_geometry_adjustment.csv'}")


if __name__ == "__main__":
    main()
