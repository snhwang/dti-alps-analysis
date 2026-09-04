"""Under what condition does pitch lower the index?

Section on the two-region geometry derives the pitch factor after placing
lambda1 exactly on the axis each region is assumed to follow. Under that
placement pitch cannot raise the index, and the manuscript said so without
qualification: "no tissue exception", "in every voxel and for any participant".

Real fibers are not exactly on those axes. This finds the angle at which the
statement stops holding, by tilting the fiber away from its assumed axis inside
the plane pitch rotates and asking when the sign flips.

The mechanism is that pitch mixes y and z. In the projection region the
denominator is D_yy, and pitch raises it only while D_zz exceeds D_yy, which
holds while the fiber is nearer z than y. Past the crossover the same rotation
lowers the denominator and raises the index.

    python pitch_sign_condition.py
"""
from __future__ import annotations

import numpy as np


def rot_x(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def index(l1, l2, l3, phi, theta):
    """Classic index after pitching by theta, with each region's fiber tilted
    phi from its assumed axis inside the y-z plane."""
    # projection: fiber nominally on z, tilted by phi toward y
    Rp = rot_x(phi)
    Dp = Rp @ np.diag([l2, l3, l1]) @ Rp.T
    # association: fiber nominally on y, tilted by phi toward z
    Ra = rot_x(-phi)
    Da = Ra @ np.diag([l2, l1, l3]) @ Ra.T
    R = rot_x(theta)
    Dp, Da = R @ Dp @ R.T, R @ Da @ R.T
    num = Dp[0, 0] + Da[0, 0]
    den = Dp[1, 1] + Da[2, 2]
    return num / den


def main() -> None:
    l1, l2, l3 = 1.6, 0.75, 0.5          # representative white matter
    theta = np.radians(10.0)
    print(f"  lambda = ({l1}, {l2}, {l3}), pitch {np.degrees(theta):.0f} deg\n")
    print(f"  {'fiber tilt':>11s} {'R(0)':>9s} {'R(pitch)':>10s} {'change':>9s}")
    flip = None
    for deg in (0, 10, 20, 30, 40, 44, 45, 46, 50, 60):
        phi = np.radians(deg)
        r0 = index(l1, l2, l3, phi, 0.0)
        rp = index(l1, l2, l3, phi, theta)
        d = rp - r0
        if flip is None and d > 0:
            flip = deg
        print(f"  {deg:>9d}° {r0:>9.4f} {rp:>10.4f} {d:>+9.4f}")
    print(f"\n  sign flips between {flip - 1} and {flip} degrees of fiber tilt")
    print("  The observed departures are 8 to 16 degrees (Section 3.1), so the")
    print("  condition holds in these data, but it is a condition and not a")
    print("  universal property of the index.")


if __name__ == "__main__":
    main()
