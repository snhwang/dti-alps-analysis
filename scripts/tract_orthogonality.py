"""What the near-perpendicular arrangement of the two tracts costs.

Classic ALPS is

    (Dxx_proj + Dxx_assoc) / (Dyy_proj + Dzz_assoc)

with x left-right, y anterior-posterior, z superior-inferior. Projection fibers
run along z and association fibers along y, so with lambda2 on x in both regions
the tensors are

    D_proj  = diag(l2, l3, l1)          D_assoc = diag(l2, l1, l3)

and the index reads l2 / l3 = rho at zero rotation.

Part one is closed form. Rotate the head by theta and read off the transformed
diagonal elements of R D R^T.

  Pitch, about x. Rotation about x leaves Dxx unchanged, so the numerator stays
  2 l2, and it maps both denominator terms to l3 cos^2 + l1 sin^2. Writing
  kappa = l1 / l3,

      R_pitch(theta) / rho = 1 / (1 + (kappa - 1) sin^2 theta)

  This depends on kappa alone, not on rho. Since l1 > l3 in any anisotropic
  voxel, kappa > 1, so the factor is strictly below one for every non-zero
  theta. Pitch always lowers the index. The sign is not an empirical finding
  and admits no tissue exception.

  Roll, about y, and yaw, about z, give algebraically identical numerators and
  denominators,

      R(theta) = (2 l2 + (l1 + l3 - 2 l2) sin^2) / (2 l3 + (l2 - l3) sin^2)

  which proves the equality rather than observing it. Exchanging y and z
  exchanges the two regions and leaves the formula invariant, and that same
  exchange carries roll into yaw.

  To first order in sin^2 theta the fractional changes are

      pitch     -(kappa - 1)
      roll/yaw  (kappa + 1 - rho - rho^2) / (2 rho)

  The pitch coefficient cannot vanish. The roll and yaw coefficient is the
  difference between a numerator gain and a denominator gain, it vanishes at
  kappa = rho^2 + rho - 1, and it changes sign there. In white matter that
  crossing sits inside the plausible range, so the transverse rotations are
  both small and of undetermined sign while pitch is neither.

Both forms are checked against direct tensor rotation below, and agree to
machine precision. This part uses no participant data.

Part two is measured. The common perpendicular to two non-parallel directions
exists and is unique, so non-perpendicularity never prevents a shared axis. What
it does is use up the freedom. One perpendicularity constraint leaves a plane of
admissible axes, within which one could be chosen to align with v2 and attain
the bound exactly. A second constraint pins the axis to a line, leaving nothing
with which to point it at v2 in either region. The determined axis is then not
the aligned one, and measured, it is farther from v2 than scanner x is. This
part reads the measured-axis tables and skips with a message if they are absent.

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
L1 = 1.2e-3              # mm^2/s along the fiber, held fixed
LT = 0.5e-3              # mm^2/s, mean of the transverse pair, held fixed
DEGREES = (5, 10, 15, 20, 30)
RATIOS = (1.2, 1.5, 1.72, 2.0)
RHO_TYPICAL = 1.72       # the measured regional value, for costing the angles
COHORTS = (("HCP-A", "measured_pvs_axis_hcpa_b1500_all.csv"),
           ("DLBS", "measured_pvs_axis_dlbs.csv"))


def eigenvalues(rho):
    """The triple with the requested transverse ratio, at fixed transverse mean."""
    l3 = 2 * LT / (1 + rho)
    return L1, rho * l3, l3


def R_of_S(S, rho=RHO_TYPICAL):
    """Two regions at misalignments alpha_p, alpha_a enter only through

        S = sin^2(alpha_p) + sin^2(alpha_a)

    because the index is a ratio of sums. Minimising S at fixed separation
    delta between the two regional v2 directions gives alpha_p = alpha_a =
    delta/2 and S = 1 - cos(delta), so a two-region disagreement of delta costs
    exactly what a single region misaligned by delta/2 would. That equality is
    exact, not a small-angle result.
    """
    return (2 * rho - (rho - 1) * S) / (2 + (rho - 1) * S)


def rot(axis, deg):
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def by_rotation(axis, deg, rho):
    """The index after rotating both tensors directly. The thing to be matched."""
    l1, l2, l3 = eigenvalues(rho)
    R = rot(axis, deg)
    P = R @ np.diag([l2, l3, l1]) @ R.T
    A = R @ np.diag([l2, l1, l3]) @ R.T
    return (P[0, 0] + A[0, 0]) / (P[1, 1] + A[2, 2])


def pitch_closed(deg, rho):
    l1, _, l3 = eigenvalues(rho)
    return rho / (1 + (l1 / l3 - 1) * np.sin(np.radians(deg)) ** 2)


def transverse_closed(deg, rho):
    """Roll and yaw, which share one expression."""
    l1, l2, l3 = eigenvalues(rho)
    u = np.sin(np.radians(deg)) ** 2
    return (2 * l2 + (l1 + l3 - 2 * l2) * u) / (2 * l3 + (l2 - l3) * u)


def rotation_part():
    rows, worst = [], 0.0
    for rho in RATIOS:
        l1, _, l3 = eigenvalues(rho)
        kappa = l1 / l3
        for d in DEGREES:
            for name, axis, closed in (("pitch", "x", pitch_closed),
                                       ("roll", "y", transverse_closed),
                                       ("yaw", "z", transverse_closed)):
                exact, direct = closed(d, rho), by_rotation(axis, d, rho)
                worst = max(worst, abs(exact - direct))
                rows.append(dict(rho=rho, kappa=round(kappa, 6), rotation=name,
                                 degrees=d, index=round(exact, 8),
                                 pct_change=round(100 * (exact / rho - 1), 4),
                                 closed_minus_direct=exact - direct))

    out = pd.DataFrame(rows)
    out.to_csv(HERE / "tract_orthogonality_rotation.csv", index=False)

    print("   Closed forms against direct tensor rotation")
    print(f"   worst absolute disagreement over all conditions: {worst:.2e}\n")

    print("   First-order coefficients, fractional change per sin^2(theta)\n")
    print(f"   {'rho':>6} {'kappa':>7} {'pitch':>9} {'roll/yaw':>10} "
          f"{'sign flip at kappa':>19} {'ratio':>7}")
    for rho in RATIOS:
        l1, _, l3 = eigenvalues(rho)
        k = l1 / l3
        pitch = -(k - 1)
        trans = (k + 1 - rho - rho ** 2) / (2 * rho)
        print(f"   {rho:6.2f} {k:7.3f} {pitch:9.3f} {trans:10.4f} "
              f"{rho**2 + rho - 1:19.3f} {abs(pitch / trans):7.1f}")

    print("\n   The pitch coefficient is -(kappa - 1) and kappa exceeds one in any")
    print("   anisotropic voxel, so it cannot vanish and cannot change sign. The")
    print("   transverse coefficient is a difference of competing terms, crosses")
    print("   zero at kappa = rho^2 + rho - 1, and is small on either side of it.")

    r15 = out[(out.rho == 1.5) & (out.degrees == 15)].set_index("rotation")
    print(f"\n   At 15 degrees and rho = 1.5: pitch "
          f"{r15.loc['pitch', 'pct_change']:+.2f}%, roll and yaw "
          f"{r15.loc['roll', 'pct_change']:+.2f}%.")
    return out


def alignment_part():
    rows, header = [], False
    print("\n\n   How far v2 sits from each candidate axis\n")
    for cohort, fname in COHORTS:
        path = HERE / fname
        if not path.exists():
            print(f"   {cohort}: {fname} not found, skipping")
            continue
        if not header:
            print(f"   {'cohort':8s} {'measure':14s} {'median':>8s} "
                  f"{'IQR':>18s} {'n':>7s}")
            header = True
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


def floor_part():
    """How well could any single axis possibly do, and does the determined one?

    The two regional v2 directions are separated by some angle delta. For any
    single axis, the spherical triangle inequality gives

        alpha_proj + alpha_assoc >= delta

    so the worse of the two errors is at least delta/2, attained only by the
    bisector. That is a floor no single-axis variant can beat, set entirely by
    how much the two regions disagree. Measuring it turns the over-determination
    argument into a quantity.
    """
    rows = []
    print("\n\n   The floor on any single axis\n")
    print(f"   {'cohort':8s} {'v2 disagree':>12s} {'floor':>7s} {'cross':>7s} "
          f"{'floor cost':>11s} {'cross cost':>11s} {'n':>6s}")
    for cohort, fname in COHORTS:
        path = HERE / fname
        if not path.exists():
            continue
        m = pd.read_csv(path)
        need = ("v2_proj_to_assoc", "v2_proj_to_cross", "v2_assoc_to_cross")
        if not all(c in m.columns for c in need):
            print(f"   {cohort}: regional v2 columns absent, rerun "
                  f"measured_pvs_axis.py")
            continue
        s = m[list(need)].dropna()
        delta = s.v2_proj_to_assoc
        floor = delta / 2.0
        # the determined axis is judged by its worse region, as the floor is
        attained = s[["v2_proj_to_cross", "v2_assoc_to_cross"]].max(axis=1)
        # and the cost of each, through R(S)
        S_floor = 1 - np.cos(np.radians(delta))
        S_cross = (np.sin(np.radians(s.v2_proj_to_cross)) ** 2
                   + np.sin(np.radians(s.v2_assoc_to_cross)) ** 2)
        cost = lambda S: 100 * (1 - R_of_S(S) / RHO_TYPICAL)
        rows.append(dict(cohort=cohort,
                         v2_disagreement_deg=round(float(delta.median()), 3),
                         floor_deg=round(float(floor.median()), 3),
                         cross_attains_deg=round(float(attained.median()), 3),
                         excess_deg=round(float((attained - floor).median()), 3),
                         floor_cost_pct=round(float(cost(S_floor).median()), 3),
                         cross_cost_pct=round(float(cost(S_cross).median()), 3),
                         n=int(len(s))))
        print(f"   {cohort:8s} {delta.median():12.2f} {floor.median():7.2f} "
              f"{attained.median():7.2f} {cost(S_floor).median():10.2f}% "
              f"{cost(S_cross).median():10.2f}% {len(s):6d}")

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "tract_orthogonality_floor.csv", index=False)
    print("\n   The two regions disagree, so no single axis is aligned in both. The")
    print("   best any could do is half that disagreement, and the axis the tract")
    print("   geometry determines does not reach it. The over-determination is not")
    print("   an argument about degrees of freedom, it is this many degrees.")
    return out


def main() -> None:
    argparse.ArgumentParser().parse_args()
    rotation_part()
    alignment_part()
    floor_part()


if __name__ == "__main__":
    main()
