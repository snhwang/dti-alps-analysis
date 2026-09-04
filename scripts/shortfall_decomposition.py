"""Is each variant's shortfall from the bound its own axis error?

Section 2 shows that a tract-locked variant attains at most lambda2/lambda3 and
falls below it at second order in the angle alpha between its perivascular axis
and the second eigenvector. That predicts a specific number for each variant,
which this checks.

Inverting the closed form on the observed shortfall gives an effective angle,
the alpha a variant would need if axis error were the whole story. Comparing
that against the angles actually measured tests the account.

The reference is the regional measured axis. Its axis IS the pooled second
eigenvector, so its own alpha is zero by construction and whatever shortfall it
shows is within-region dispersion: the pooled axis is not each voxel's own v2.
Every other tract-locked variant should then carry that dispersion plus its own
measured misalignment, and independent angular errors should add in quadrature.

    predicted effective angle = sqrt(dispersion^2 + misalignment^2)

Classic ALPS is excluded from the quadrature test and reported only for scale.
It is not tract-locked, so its axes do not lie in the plane perpendicular to the
fiber and the single-angle parameterization does not describe it.

    python shortfall_decomposition.py

Writes shortfall_decomposition.csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import atomic_io  # noqa: F401  writes become atomic on import

HERE = Path(__file__).resolve().parent
# variant -> the measured angle between its axis and the second eigenvector
# The misalignment angles must be measured against whichever axis is the
# reference, since that is the direction the quadrature model treats as
# systematically correct.
MISALIGNED_BY_REF = {
    "v2_slab": {"anat_x": "v2_to_x", "cross": "v2_to_cross"},
    "v2_sphere": {"anat_x": "v2sph_to_x", "cross": "v2sph_to_cross"},
}


def attained(alpha_deg, rho):
    """Fraction of the bound reached at angle alpha, closed form."""
    a = np.radians(alpha_deg)
    c, s = np.cos(a) ** 2, np.sin(a) ** 2
    return ((rho * c + s) / (rho * s + c)) / rho


def invert(frac, rho, grid=np.linspace(0, 45, 45001)):
    """The angle whose shortfall matches the one observed."""
    return float(grid[np.argmin(np.abs(attained(grid, rho) - frac))])


def main() -> None:
    ap = argparse.ArgumentParser()
    # v2_slab pools its axis over the tract band while the diffusivities come
    # from the sphere, so its own alpha is not zero by construction. v2_sphere
    # pools over exactly the measured voxels, which is the condition the
    # reference role needs. The switch exists so the two can be compared.
    ap.add_argument("--reference", choices=["v2_slab", "v2_sphere"],
                    default="v2_slab")
    ap.add_argument("--source", default="measured_pvs_axis_dlbs.csv")
    ap.add_argument("--out", default="shortfall_decomposition.csv")
    args = ap.parse_args()
    REFERENCE = args.reference
    MISALIGNED = MISALIGNED_BY_REF[REFERENCE]
    d = pd.read_csv(HERE / args.source)
    rho = float(d.pv_perp.median())
    print(f"DLBS, {len(d)} sessions, median lambda2/lambda3 = {rho:.3f}\n")

    obs = {c: float((d[c] / d.pv_perp).median())
           for c in ("classic", "cross", "anat_x", REFERENCE) if c in d}
    eff = {c: invert(v, rho) for c, v in obs.items()}

    disp = eff[REFERENCE]
    print(f"Reference: {REFERENCE} attains {obs[REFERENCE]*100:.1f}% of the bound,")
    print(f"an effective {disp:.2f} deg. Its own axis is the pooled second")
    print("eigenvector, so this is within-region dispersion.\n")

    rows = [dict(variant=REFERENCE, attained=obs[REFERENCE], effective_deg=disp,
                 misalign_deg=0.0, predicted_deg=disp, role="dispersion reference")]

    print(f"   {'variant':10s} {'attained':>9s} {'effective':>10s} "
          f"{'misalign':>9s} {'predicted':>10s} {'error':>7s}")
    print(f"   {REFERENCE:10s} {obs[REFERENCE]*100:8.1f}% {disp:9.2f}d "
          f"{0.0:8.2f}d {disp:9.2f}d {0.0:6.2f}d")

    for v, col in MISALIGNED.items():
        if v not in obs or col not in d:
            continue
        mis = float(d[col].median())
        pred = float(np.hypot(disp, mis))
        rows.append(dict(variant=v, attained=obs[v], effective_deg=eff[v],
                         misalign_deg=mis, predicted_deg=pred,
                         role="dispersion + own misalignment"))
        print(f"   {v:10s} {obs[v]*100:8.1f}% {eff[v]:9.2f}d {mis:8.2f}d "
              f"{pred:9.2f}d {eff[v]-pred:+6.2f}d")

    if "classic" in obs:
        rows.append(dict(variant="classic", attained=obs["classic"],
                         effective_deg=eff["classic"], misalign_deg=np.nan,
                         predicted_deg=np.nan, role="not tract-locked, scale only"))
        print(f"   {'classic':10s} {obs['classic']*100:8.1f}% {eff['classic']:9.2f}d"
              f"{'':10s}{'not tract-locked':>11s}")

    pd.DataFrame(rows).to_csv(HERE / args.out, index=False)

    print("\n   What perfect tract directions would buy. If a variant's only remaining")
    print("   error were its own misalignment, with dispersion removed, it would reach")
    for v, col in MISALIGNED.items():
        if v in obs and col in d:
            a = float(d[col].median())
            print(f"     {v:10s} {attained(a, rho)*100:5.1f}% of the bound "
                  f"(alpha = {a:.2f} deg), against {obs[v]*100:.1f}% now")
    print("\n   Not the bound itself. Reaching that means pointing the numerator at")
    print("   the second eigenvector, which is what makes the index lambda2/lambda3.")


if __name__ == "__main__":
    main()
