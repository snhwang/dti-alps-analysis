"""Is hand placement age-dependent, and does that justify automating it?

A rater placing an ALPS region works from a directionally encoded colour map
and puts the region where the colour is vivid, blue for the superior-inferior
projection fibres and green for the anterior-posterior association fibres. Hue
encodes the fibre direction and brightness encodes fractional anisotropy, so
the rater is selecting on the two quantities the index is built from.

That matters because FA falls with age. In HCP-A the association region loses
FA at r=-0.207 against age. A rater in an older brain therefore hunts for
whatever vivid colour survives and lands somewhere anatomically different from
where they would land in a younger brain. The selection rule itself becomes a
function of age, which is the same defect that rules out choosing the highest-FA
voxels automatically.

Atlas placement cannot do this. A warped template region goes where the
registration puts it, and the registration never sees FA. If the hand-placed
regions drift with age relative to the atlas ones, that is a reason to prefer
the automated placement rather than merely a convenience.

Two tests, on the sessions that carry both placements:

    index    does the hand-minus-atlas difference in the index track age
    geometry does the distance between the hand and atlas region centres
             track age

    python manual_placement_age.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
DIFF = HERE.parent.parent / "diffusion"


def report(x, y, label, xname="age"):
    ok = ~(pd.isna(x) | pd.isna(y))
    x, y = np.asarray(x, float)[ok], np.asarray(y, float)[ok]
    if len(x) < 10:
        print(f"   {label:<34s} too few ({len(x)})")
        return None
    r, p = stats.pearsonr(x, y)
    print(f"   {label:<34s} n={len(x):3d}  r={r:+.3f}  p={p:.4f}")
    return {"comparison": label, "n": len(x), "r": r, "p": p}


def main() -> None:
    argparse.ArgumentParser().parse_args()
    rows = []

    m = pd.read_csv(HERE / "manual_pvs_axis.csv")
    print(f"index test: {len(m)} sessions with both placements\n")
    print("=== hand-minus-atlas index difference against age ===")
    for col, auto in (("classic", "auto_classic"), ("cross", "auto_cross"),
                      ("v2_slab", "auto_v2_slab")):
        if col in m.columns and auto in m.columns:
            rows.append(report(m.Age, m[col] - m[auto], f"{col}: manual - atlas"))
    print()
    print("   For reference, each placement's own age association:")
    for col in ("classic", "auto_classic"):
        rows.append(report(m.Age, m[col], f"{col} vs age"))

    # manual_roi_placement_check.csv is NOT usable for this. It is keyed by
    # HCP-A session, and the HCP-A regions were placed automatically, so it
    # compares automated placement against automated placement and is null by
    # construction. The genuinely hand-drawn regions are the DLBS ones, and
    # manual_roi_offtract.csv records how far each of them sits from its
    # intended tract.
    off = HERE / "manual_roi_offtract.csv"
    if off.exists():
        o = pd.read_csv(off)
        sp = pd.read_csv(DIFF / "DLBS" / "dlbs_alps_spheres_5mm.csv")
        sp["DTI_Session_ID"] = sp.DTI_Session_ID.astype(str)
        o["sid"] = o.sid.astype(str)
        j2 = o.merge(sp[["DTI_Session_ID", "Age"]], left_on="sid",
                     right_on="DTI_Session_ID", how="inner")
        j2["Age"] = pd.to_numeric(j2.Age, errors="coerce")
        print()
        print(f"=== hand-drawn off-tract distance against age "
              f"({len(j2)} DLBS sessions) ===")
        for c in ("proj_off", "assoc_off", "worst"):
            if c in j2.columns:
                rows.append(report(j2.Age, j2[c], f"{c} (mm from tract)"))

    out = pd.DataFrame([r for r in rows if r])
    out.to_csv(HERE / "manual_placement_age.csv", index=False)
    print(f"\n   wrote manual_placement_age.csv")


if __name__ == "__main__":
    main()
