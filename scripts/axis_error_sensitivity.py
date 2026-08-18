"""
Two ways the measurement axes can be wrong, and why only one of them matters.

The paper says head orientation is a large confound and also that the estimated
axes need not be reproduced precisely. Those sound contradictory and are not.
They describe different errors.

  Frame tilted off the tract.  What head rotation does to fixed scanner axes.
      The axes stop being perpendicular to the fibre, so the fibre's own
      lambda1 leaks into both numerator and denominator. lambda1 is two to
      three times lambda2 and lambda3, so a small angle admits a large
      diffusivity.

  Axis wrong within the perpendicular plane.  What imprecise estimation does.
      The denominators are built as p x v_proj and p x v_assoc from the
      measured tract directions, so perpendicularity to the fibre survives
      wherever p points. The error only trades lambda2 against lambda3, which
      are similar in size.

Both are imposed here at the same angles on the same tensors, so the comparison
is like for like. Uses the synthetic geometry shipped with the implementation,
so it needs no data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import atomic_io  # noqa: F401  writes become atomic on import

REPO = Path(r"C:\Users\Scott\Documents\Work\dti-alps-refined")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))
from test_invariance import synthetic, rotation                      # noqa: E402
from dti_alps_refined.alps import (sorted_eigen, directional_diffusivity,  # noqa: E402
                                   _region, fractional_anisotropy,
                                   L_SCR, L_SLF, X, Y, Z)

ANGLES = (2.0, 5.0, 10.0, 15.0, 20.0)
HERE = Path(__file__).resolve().parent


def main() -> None:
    evals, evecs, affine, rois = synthetic()
    lam, vec = sorted_eigen(evals, evecs)
    fa = fractional_anisotropy(evals)
    P = _region(lam, vec, (rois == L_SCR) & (fa >= 0.2))
    A = _region(lam, vec, (rois == L_SLF) & (fa >= 0.2))

    def ratio(p, vp, va):
        op = np.cross(p, vp); op /= np.linalg.norm(op)
        oa = np.cross(p, va); oa /= np.linalg.norm(oa)
        return ((directional_diffusivity(P["lam"], P["vec"], p)
                 + directional_diffusivity(A["lam"], A["vec"], p))
                / (directional_diffusivity(P["lam"], P["vec"], op)
                   + directional_diffusivity(A["lam"], A["vec"], oa)))

    base = ratio(X, Z, Y)
    rows = []
    for deg in ANGLES:
        # the perivascular axis is wrong, but the denominators are rebuilt from
        # the true tract directions, so the triad stays perpendicular to the fibre
        within = ratio(rotation("z", deg) @ X, Z, Y)
        # the whole frame tilts relative to the tract, as head rotation does to
        # fixed scanner axes
        R = rotation("x", deg)
        off = ((directional_diffusivity(P["lam"], P["vec"], R @ X)
                + directional_diffusivity(A["lam"], A["vec"], R @ X))
               / (directional_diffusivity(P["lam"], P["vec"], R @ Y)
                  + directional_diffusivity(A["lam"], A["vec"], R @ Z)))
        rows.append({"deg": deg,
                     "within_plane_pct": 100 * (within - base) / base,
                     "frame_off_tract_pct": 100 * (off - base) / base})

    d = pd.DataFrame(rows)
    d.to_csv(HERE / "axis_error_sensitivity.csv", index=False)
    print(f"{'angle':>6s} {'within perpendicular plane':>28s} {'frame off the tract':>22s}")
    for r in d.itertuples():
        print(f"{r.deg:4.0f} deg {r.within_plane_pct:26.2f}% {r.frame_off_tract_pct:21.2f}%")
    ratio_20 = abs(d.frame_off_tract_pct.iloc[-1]) / abs(d.within_plane_pct.iloc[-1])
    print(f"\nat {ANGLES[-1]:.0f} degrees the two differ by a factor of {ratio_20:.0f}")

    # Matched angles overstate how close the two errors are. What decides the
    # question is the angle each one actually takes: the estimated axes wobble
    # by a degree or a few between visits, while the head rotation the
    # correction exists to remove is an order larger and of the harmful kind.
    print("\nat the angles observed in these cohorts")
    obs = {}
    for label, deg, kind in (("axis estimate, HCP-A", 1.32, "within"),
                             ("axis estimate, DLBS", 4.14, "within"),
                             ("head rotation, DLBS", 10.66, "frame")):
        if kind == "within":
            v = 100 * (ratio(rotation("z", deg) @ X, Z, Y) - base) / base
        else:
            R = rotation("x", deg)
            f = ((directional_diffusivity(P["lam"], P["vec"], R @ X)
                  + directional_diffusivity(A["lam"], A["vec"], R @ X))
                 / (directional_diffusivity(P["lam"], P["vec"], R @ Y)
                    + directional_diffusivity(A["lam"], A["vec"], R @ Z)))
            v = 100 * (f - base) / base
        obs[label] = v
        print(f"  {label:<22s} {deg:5.2f} deg  {v:+7.3f}%")
    factor = abs(obs["head rotation, DLBS"]) / abs(obs["axis estimate, DLBS"])
    print(f"  as they actually occur the two differ by a factor of {factor:.0f}")
    pd.DataFrame([{"case": k, "pct": v} for k, v in obs.items()]).to_csv(
        HERE / "axis_error_observed.csv", index=False)
    print(f"\nwrote {HERE / 'axis_error_sensitivity.csv'}")


if __name__ == "__main__":
    main()
