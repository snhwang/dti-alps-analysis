"""What the near-perpendicular arrangement of the two tracts costs.

Classic ALPS is

    (Dxx_proj + Dxx_assoc) / (Dyy_proj + Dzz_assoc)

with x left-right, y anterior-posterior, z superior-inferior. Projection fibers
run along z and association fibers along y, so the two tracts are close to
perpendicular. That is what lets a single axis serve both regions, and it has two
consequences the manuscript reports.

Part one, which rotation hurts. Pitch is rotation about x. Rotation about x
leaves Dxx exactly unchanged, so both numerator terms are invariant, and it mixes
y with z, which are precisely the two axes carrying the two fibers. lambda1
therefore enters both denominator terms at once, in the same direction, with
nothing in the numerator to offset it. Roll and yaw each disturb one term only,
and that term is partly offset by its partner in the same sum. This part is
simulation and uses no participant data.

Part two, what the residual non-perpendicularity costs. The common perpendicular
to two non-parallel directions exists and is unique, so non-perpendicularity
never prevents a shared axis. What it does is use up the freedom. One
perpendicularity constraint leaves a plane of admissible axes, within which one
could be chosen to align with v2 and attain the bound exactly. A second
constraint pins the axis to a line, leaving nothing with which to point it at v2
in either region. The determined axis is then not the aligned one, and measured,
it is farther from v2 than scanner x is. This part reads the measured-axis tables
and skips with a message if they are absent.

    python tract_orthogonality.py

Writes tract_orthogonality_rotation.csv and tract_orthogonality_alignment.csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import atomic_io  # noqa: F401  writes become atomic on import

HERE = Path(__file__).resolve().parent
L1 = 1.2e-3              # mm^2/s, along the fiber
LT = 0.5e-3              # mm^2/s, mean of the transverse pair, held fixed
DEGREES = (5, 10, 15, 20, 30)
RATIOS = (1.5, 1.72)     # representative, and the measured regional value
AXES = (("pitch", "x"), ("roll", "y"), ("yaw", "z"))
COHORTS = (("HCP-A", "measured_pvs_axis_hcpa_b1500_all.csv"),
           ("DLBS", "measured_pvs_axis_dlbs.csv"))


def transverse(rho):
    """The transverse pair with the requested ratio, at fixed transverse mean."""
    l3 = 2 * LT / (1 + rho)
    return rho * l3, l3


def tensor(fiber, rho):
    """Diagonal tensor in the scanner frame with the fiber along the named axis.

    lambda2 lies on x in both regions, which is the premise classic ALPS makes.
    """
    l2, l3 = transverse(rho)
    return np.diag([l2, l3, L1]) if fiber == "z" else np.diag([l2, L1, l3])


def rot(axis, deg):
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def alps(axis, deg, rho):
    """Classic index after the head rotates by deg about the named scanner axis."""
    R = rot(axis, deg)
    P = R @ tensor("z", rho) @ R.T          # projection region, fiber along z
    A = R @ tensor("y", rho) @ R.T          # association region, fiber along y
    return (P[0, 0] + A[0, 0]) / (P[1, 1] + A[2, 2])


def rotation_part():
    rows = []
    for rho in RATIOS:
        base = alps("x", 0, rho)
        print(f"\n   rho = {rho:.2f}, index at zero rotation = {base:.4f}\n")
        print(f"   {'degrees':>8s} {'pitch':>10s} {'roll':>10s} {'yaw':>10s}")
        for d in DEGREES:
            pct = {}
            for name, ax in AXES:
                pct[name] = 100 * (alps(ax, d, rho) / base - 1)
                rows.append(dict(rho=rho, rotation=name, degrees=d,
                                 pct_change=round(pct[name], 4)))
            print(f"   {d:8d} {pct['pitch']:+10.2f} {pct['roll']:+10.2f} "
                  f"{pct['yaw']:+10.2f}")
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "tract_orthogonality_rotation.csv", index=False)

    p15 = out[(out.rho == 1.5) & (out.degrees == 15)].set_index("rotation")
    ratio = abs(p15.loc["pitch", "pct_change"] / p15.loc["roll", "pct_change"])
    print(f"\n   At 15 degrees and rho = 1.5, pitch costs {ratio:.0f} times what "
          f"roll or yaw does.")
    print("   Roll and yaw are equal because exchanging y and z exchanges the two")
    print("   regions and leaves the formula unchanged, in this idealized pair.")
    return out


def alignment_part():
    rows = []
    print("\n\n   How far v2 sits from each candidate axis\n")
    print(f"   {'cohort':8s} {'measure':14s} {'median':>8s} {'IQR':>18s} {'n':>7s}")
    for cohort, fname in COHORTS:
        path = HERE / fname
        if not path.exists():
            print(f"   {cohort}: {fname} not found, skipping")
            continue
        m = pd.read_csv(path)
        for col in ("v2_to_x", "v2_to_cross"):
            if col not in m.columns:
                continue
            s = m[col].dropna()
            lo, hi = s.quantile(.25), s.quantile(.75)
            rows.append(dict(cohort=cohort, measure=col,
                             median_deg=round(float(s.median()), 3),
                             iqr_lo=round(float(lo), 3), iqr_hi=round(float(hi), 3),
                             n=int(len(s))))
            print(f"   {cohort:8s} {col:14s} {s.median():8.2f} "
                  f"{f'[{lo:.2f}, {hi:.2f}]':>18s} {len(s):7d}")

    if not rows:
        print("   no measured-axis tables present, alignment part skipped")
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out.to_csv(HERE / "tract_orthogonality_alignment.csv", index=False)

    print("\n   The common perpendicular is the axis the tract geometry determines.")
    print("   In both cohorts it sits farther from v2 than scanner x does:")
    for cohort, g in out.groupby("cohort", sort=False):
        g = g.set_index("measure")
        if {"v2_to_x", "v2_to_cross"} <= set(g.index):
            print(f"     {cohort:8s} x {g.loc['v2_to_x', 'median_deg']:5.2f} deg, "
                  f"cross {g.loc['v2_to_cross', 'median_deg']:5.2f} deg")
    print("   Constraint satisfaction and alignment are different objectives, and")
    print("   only the second governs the ratio.")
    return out


def main() -> None:
    argparse.ArgumentParser().parse_args()
    rotation_part()
    alignment_part()


if __name__ == "__main__":
    main()
