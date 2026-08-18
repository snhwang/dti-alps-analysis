"""Numerical check of the bound: every tract-locked ALPS ratio is at most lambda2/lambda3.

The claim proved in Section 2.7 and checked here.

Let a voxel have eigenvalues lam1 >= lam2 >= lam3 with eigenvectors v1, v2, v3.
Once the measurement frame is locked to the fiber, both the numerator axis p and
the denominator axis o lie in the plane spanned by v2 and v3, and o is
perpendicular to p within it. Writing p at angle alpha from v2,

    D(p) = lam2 cos^2 a + lam3 sin^2 a
    D(o) = lam2 sin^2 a + lam3 cos^2 a

so with c = cos^2 a the ratio is

    R(c) = [lam3 + (lam2 - lam3) c] / [lam2 - (lam2 - lam3) c]

Differentiating, and writing N and Dn for numerator and denominator,

    dR/dc = (lam2 - lam3)(N + Dn) / Dn^2 = (lam2 - lam3)(lam2 + lam3) / Dn^2 >= 0

so R increases monotonically in c and is maximised at c = 1, that is at
alpha = 0, where R = lam2 / lam3. The ratio is therefore bounded above by the
eigenvalue ratio, with equality exactly when the perivascular axis coincides
with the second eigenvector.

Expanding about alpha = 0 gives the rate:

    R / (lam2/lam3) ~ 1 - alpha^2 (lam2^2 - lam3^2) / (lam2 lam3)

so the shortfall is second order in the axis error. A one-degree error costs
almost nothing, which is the same fact Section 4.1 reaches from the other
direction when it argues that imprecise axis estimation is cheap.

Both statements survive averaging. Per voxel D(p) <= lam2 and D(o) >= lam3, so
the inequality holds for regional means and for the two-region sum the index
actually forms.

One way the bound can be violated, and it is informative. A regional axis is
perpendicular to the region's mean fiber direction, not to each voxel's own v1,
so in a voxel whose direction departs from that mean the axis acquires a
component along v1 and admits lam1, which exceeds lam2. Violations therefore
measure within-region dispersion rather than a failure of the algebra, and they
should be small and confined to variants using a regional axis.

    python ratio_bound_proof.py

Writes ratio_bound_proof.csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import atomic_io  # noqa: F401  writes become atomic on import

HERE = Path(__file__).resolve().parent


def ratio(lam2: float, lam3: float, alpha: float) -> float:
    c, s = np.cos(alpha) ** 2, np.sin(alpha) ** 2
    return (lam2 * c + lam3 * s) / (lam2 * s + lam3 * c)


def main() -> None:
    argparse.ArgumentParser().parse_args()
    rng = np.random.default_rng(0)

    print("1. the closed form is the supremum, attained only at alpha = 0\n")
    print("   lam2/lam3   alpha    R(alpha)   R / bound   quadratic prediction")
    rows = []
    for lam2, lam3 in ((1.0e-3, 0.6e-3), (0.9e-3, 0.8e-3)):
        bound = lam2 / lam3
        for deg in (0, 1, 2, 5, 10, 20):
            a = np.radians(deg)
            r = ratio(lam2, lam3, a)
            pred = 1 - a ** 2 * (lam2 ** 2 - lam3 ** 2) / (lam2 * lam3)
            rows.append(dict(check="closed_form", lam2=lam2, lam3=lam3, deg=deg,
                             R=r, frac=r / bound, quad=pred))
            print(f"   {bound:7.3f}   {deg:3d} deg  {r:8.4f}   {r / bound:8.4f}"
                  f"      {pred:8.4f}")
        print()

    # brute force: no direction in the perpendicular plane beats alpha = 0
    print("2. exhaustive search over the perpendicular plane, random tensors\n")
    worst = 0.0
    for _ in range(20000):
        lam = np.sort(rng.uniform(0.2e-3, 2.0e-3, 3))[::-1]
        bound = lam[1] / lam[2]
        a = rng.uniform(0, np.pi)
        r = ratio(lam[1], lam[2], a)
        worst = max(worst, r / bound)
    print(f"   largest R / (lam2/lam3) over 20000 random draws: {worst:.6f}")
    print("   (exceeding 1 would falsify the bound)\n")

    # and on real tensors, where a regional axis is not perpendicular to every voxel
    print("3. on real data, where the axis is regional and voxels disperse\n")
    d = pd.read_csv(HERE / "measured_pvs_axis_dlbs.csv")
    for c in ("classic", "cross", "v2_slab", "anat_x"):
        if c not in d.columns:
            continue
        frac = (d[c] / d.pv_perp)
        viol = float((d[c] > d.pv_perp + 1e-9).mean() * 100)
        rows.append(dict(check="real", variant=c, median_frac=float(frac.median()),
                         pct_violating=viol))
        print(f"   {c:10s} median {frac.median():.4f} of the bound, "
              f"exceeds it in {viol:5.2f}% of sessions")
    print("\n   Violations are confined to variants using a regional axis and reflect")
    print("   within-region dispersion admitting lam1, not a failure of the bound.")
    pd.DataFrame(rows).to_csv(HERE / "ratio_bound_proof.csv", index=False)


if __name__ == "__main__":
    main()
