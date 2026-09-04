"""Which residual rotation moves which departure angle?

tbl:departure-stability reports one number per column: the association between
each departure angle and the total rotation surviving HCP-A's anatomical
alignment. Projection shares 3.3 per cent of its variance with it and the other
two share none, which invites the question of why a single rotation should move
one tract's departure and not the other's.

Total rotation is a magnitude and hides the answer. A rotation about x moves a
vector near z and a vector near y by similar amounts, but leaves their common
perpendicular on x, so the three columns should not respond alike and the
per-axis breakdown is what shows it.

    python residual_posture_breakdown.py

Writes residual_posture_breakdown.csv.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import atomic_io  # noqa: F401  writes become atomic on import

HERE = Path(__file__).resolve().parent
ANGLES = [("theta_scr", "projection from z"),
          ("theta_slf", "association from y"),
          ("theta_pvs", "cross product from x")]
ROTS = ["pitch", "roll", "yaw", "total"]


def main() -> None:
    q = pd.read_csv(HERE / "roi_placement_quality_hcpa_b1500.csv")
    h = pd.read_csv(HERE / "head_rotation_hcpa.csv")
    for x in (q, h):
        x["Subject_ID"] = x.Subject_ID.astype(str)
        x["Visit"] = x.Visit.astype(str)
    m = q.merge(h, on=["Subject_ID", "Visit"], how="inner")
    print(f"HCP-A, {len(m)} sessions with both a departure and a residual rotation")
    print("residual rotation, degrees: "
          + ", ".join(f"{r} median {m[r].abs().median():.2f}" for r in ROTS) + "\n")

    rows = []
    print(f"  {'departure':<24s}" + "".join(f"{r:>16s}" for r in ROTS))
    for col, label in ANGLES:
        if col not in m:
            continue
        cells = []
        for r in ROTS:
            s = m.dropna(subset=[col, r])
            rr, pp = stats.pearsonr(s[col], s[r].abs() if r != "total" else s[r])
            cells.append(f"{rr:+.3f}{'*' if pp < 0.05 else ' '} ({100*rr*rr:4.1f}%)")
            rows.append(dict(departure=label, rotation=r, r=rr, p=pp,
                             var_pct=100 * rr * rr, n=len(s)))
        print(f"  {label:<24s}" + "".join(f"{c:>16s}" for c in cells))

    pd.DataFrame(rows).to_csv(HERE / "residual_posture_breakdown.csv", index=False)
    print("\n  * p < 0.05. Percentages are shared variance.")
    print("  A rotation about x leaves the common perpendicular on x, so the")
    print("  cross-product column is expected to be the flattest against pitch.")


if __name__ == "__main__":
    main()
