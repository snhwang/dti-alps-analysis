"""Check the two-angle form of the bound that Appendix B now states.

The appendix claims three things a reader could not verify from the text: that
the two-angle decomposition reproduces the quadratic form exactly, that a
numerator tilt carries the index above the eigenvalue ratio, and that it does so
by specific amounts at specific angles. All three are arithmetic, so they are
checked here rather than quoted from the standalone derivation they came from.

The third is what makes the appendix worth having. Section 3.5 reports regional
axes exceeding a bound proved two sections earlier, which reads as a
contradiction until the tilt is named as the mechanism. If the quoted
magnitudes did not match the observed exceedance, the explanation would be
decorative.

    python general_bound_check.py
"""
from __future__ import annotations

import numpy as np

RNG = np.random.default_rng(20260826)


def axis(alpha: float, beta: float, v2, v3, f):
    return (np.cos(beta) * (np.cos(alpha) * v2 + np.sin(alpha) * v3)
            + np.sin(beta) * f)


def main() -> None:
    # 1. the decomposition reproduces the quadratic form
    worst = 0.0
    for _ in range(20000):
        l1, l2, l3 = np.sort(RNG.uniform(0.2, 2.0, 3))[::-1]
        D = np.diag([l1, l2, l3])
        f, v2, v3 = np.eye(3)
        a, b = RNG.uniform(0, 2 * np.pi), RNG.uniform(-np.pi / 2, np.pi / 2)
        u = axis(a, b, v2, v3, f)
        direct = u @ D @ u
        model = (np.cos(b) ** 2 * (l2 * np.cos(a) ** 2 + l3 * np.sin(a) ** 2)
                 + np.sin(b) ** 2 * l1)
        worst = max(worst, abs(direct - model))
    print(f"  two-angle form vs quadratic form, worst of 20000: {worst:.2e}")
    assert worst < 1e-15, worst

    # 2 and 3. A numerator tilt carries R above rho, by how much. At alpha=0
    # the inflation is exactly cos^2(beta) + sin^2(beta)*(l1/l2), so it is set
    # by the axial-to-radial eigenvalue ratio and a bare number per angle is
    # incomplete. The first version of this check drew eigenvalues uniformly,
    # which gives l1/l2 near 1.2, and disagreed with the appendix by a factor
    # of three. The appendix figures assume 1.9, at the lower end of the two-
    # to three-fold range these regions show, and it now says so.
    print(f"\n  {'beta':>5s} {'l1/l2=1.9':>11s} {'l1/l2=3.0':>11s}"
          f" {'denominator, 1.9':>18s}")
    quoted = {5: (1.007, 1.015), 10: (1.027, 1.060), 20: (1.105, 1.234)}
    rho = 1.5
    for deg in (0, 2, 5, 10, 20):
        b = np.radians(deg)
        lo = np.cos(b) ** 2 + np.sin(b) ** 2 * 1.9
        hi = np.cos(b) ** 2 + np.sin(b) ** 2 * 3.0
        dd = rho / (rho * np.cos(b) ** 2 + np.sin(b) ** 2 * 1.9 * rho)
        print(f"  {deg:>4d}d {lo:>11.3f} {hi:>11.3f} {dd:>18.3f}")
        if deg in quoted:
            a, c = quoted[deg]
            assert abs(lo - a) < 6e-4, (deg, lo, a)
            assert abs(hi - c) < 6e-4, (deg, hi, c)

    print("\n  Both columns match what the appendix quotes.")
    print("  A numerator tilt raises the index above the ratio; a denominator")
    print("  tilt lowers it. The bound is therefore a condition on beta.")


if __name__ == "__main__":
    main()
