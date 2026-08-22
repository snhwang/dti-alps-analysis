"""A general model for ALPS-type metrics, and the reductions it contains.

NOT part of the current manuscript. This supports a separate write-up on the
design space of ALPS-like indices. Nothing here is cited by mri_revision.tex and
it is deliberately absent from the regeneration chain.

The idea. Every published ALPS variant is the same object with different axes
substituted, so write the object once and let the axes be parameters.

    R  =  sum_i w_i D(n_i)  /  sum_i w_i D(d_i)

over N regions, where D(u) = u^T D_i u is the apparent diffusivity along u. The
classic index is N = 2 with n fixed to scanner x in both regions and d fixed to
scanner y and z.

Parameterizing the axes. Within region i, let f_i be the fiber direction and
(v2_i, v3_i) the radial plane. Any unit axis is

    u(alpha, beta) = cos(beta) [cos(alpha) v2 + sin(alpha) v3] + sin(beta) f

so alpha is the in-plane angle from v2 and beta the tilt out of the radial plane
toward the fiber. Then

    D(u) = cos^2(beta) [l2 cos^2(alpha) + l3 sin^2(alpha)] + sin^2(beta) l1

which is the whole model. Two angles per axis, three eigenvalues per region.

Three consequences fall straight out, and the script checks each.

  1. beta is the only route to lambda1. A numerator tilt raises the index, a
     denominator tilt lowers it, and R <= rho holds exactly when every numerator
     beta is zero. The bound is a statement about beta, not about alpha.

  2. Building the denominator as d = n x f forces its beta to zero whatever the
     numerator does, and puts it at alpha + 90 degrees in plane. So a tract-locked
     denominator cannot leak, and only the numerator can.

  3. With a common radial anisotropy and every beta zero, the N regions enter
     through one number,

         R(S, N) = (N rho - (rho - 1) S) / (N + (rho - 1) S),
         S = sum_i sin^2(alpha_i)

     How the misalignment distributes across regions is irrelevant. At N = 2 this
     is the two-region form in Appendix A of the manuscript.

What the model is for. Each named variant is a point in the parameter space, and
the space contains combinations nobody has tried, notably per-region numerator
axes, which dissolve the over-determination entirely: give each region its own
axis and every alpha_i can be zero at once.

    python generalized_alps_model.py

Writes generalized_alps_model.csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import atomic_io  # noqa: F401  writes become atomic on import

HERE = Path(__file__).resolve().parent
X, Y, Z = np.eye(3)


# --------------------------------------------------------------------------
# the model


class Region:
    """One measurement region: a diagonalized tensor and its frame."""

    def __init__(self, l1, l2, l3, f=Z, v2=X):
        self.l1, self.l2, self.l3 = l1, l2, l3
        f = f / np.linalg.norm(f)
        v2 = v2 - (v2 @ f) * f
        v2 = v2 / np.linalg.norm(v2)
        self.f, self.v2, self.v3 = f, v2, np.cross(f, v2)

    @property
    def rho(self):
        return self.l2 / self.l3

    @property
    def tensor(self):
        B = np.column_stack([self.v2, self.v3, self.f])
        return B @ np.diag([self.l2, self.l3, self.l1]) @ B.T

    def axis(self, alpha_deg, beta_deg=0.0):
        """The unit axis at in-plane angle alpha and out-of-plane tilt beta."""
        a, b = np.radians(alpha_deg), np.radians(beta_deg)
        return (np.cos(b) * (np.cos(a) * self.v2 + np.sin(a) * self.v3)
                + np.sin(b) * self.f)

    def D(self, u):
        """Apparent diffusivity along u, by direct quadratic form."""
        u = u / np.linalg.norm(u)
        return float(u @ self.tensor @ u)

    def D_param(self, alpha_deg, beta_deg=0.0):
        """The same thing from the two angles, which is the model's claim."""
        a, b = np.radians(alpha_deg), np.radians(beta_deg)
        inplane = self.l2 * np.cos(a) ** 2 + self.l3 * np.sin(a) ** 2
        return float(np.cos(b) ** 2 * inplane + np.sin(b) ** 2 * self.l1)

    def angles_of(self, u):
        """Recover (alpha, beta) for an arbitrary axis. The inverse map."""
        u = u / np.linalg.norm(u)
        beta = np.degrees(np.arcsin(np.clip(abs(u @ self.f), 0, 1)))
        p = u - (u @ self.f) * self.f
        if np.linalg.norm(p) < 1e-12:
            return 0.0, beta
        p /= np.linalg.norm(p)
        alpha = np.degrees(np.arctan2(abs(p @ self.v3), abs(p @ self.v2)))
        return alpha, beta


def index(regions, num_axes, den_axes, weights=None):
    """The general ALPS-type metric: a weighted ratio of sums."""
    w = np.ones(len(regions)) if weights is None else np.asarray(weights, float)
    num = sum(wi * r.D(u) for wi, r, u in zip(w, regions, num_axes))
    den = sum(wi * r.D(u) for wi, r, u in zip(w, regions, den_axes))
    return num / den


def tract_locked_denominator(region, n):
    """d = n x f. Always perpendicular to the fiber, so it cannot leak l1."""
    d = np.cross(n, region.f)
    return d / max(np.linalg.norm(d), 1e-12)


def R_of_S(S, rho, N=2):
    """The reduction when every beta is zero and rho is common."""
    return (N * rho - (rho - 1) * S) / (N + (rho - 1) * S)


def rho_from_R(R, S):
    """Invert R(S) for rho. The correction, if one is wanted.

    Solving R = (2 rho - (rho-1) S) / (2 + (rho-1) S) gives

        rho = [S(R+1) - 2R] / [S(R+1) - 2]

    which returns R when S = 0 and is well posed while S(R+1) < 2. Past that the
    denominator changes sign and the inversion is meaningless, which is the
    algebraic form of the fact that the index carries no information about rho
    once the axes reach 45 degrees.
    """
    den = S * (R + 1) - 2
    return (S * (R + 1) - 2 * R) / den if abs(den) > 1e-12 else float("nan")


def inversion_is_circular():
    """What the correction needs, and why that is the whole difficulty.

    S is a function of the angles from the shared axis to v2 in each region. It
    cannot be obtained from the tract directions, because the tracts fix the
    axis without fixing its relation to v2. So a correction requires v2, and
    anything that has measured v2 can form lambda2/lambda3 directly and more
    precisely than by inverting an attenuated index.
    """
    return True


# --------------------------------------------------------------------------
# the classic geometry, as a special case


def classic_pair(rho=1.72, l1=1.2e-3, lt=0.5e-3):
    """Projection fiber on z, association on y, lambda2 on x in both."""
    l3 = 2 * lt / (1 + rho)
    l2 = rho * l3
    return [Region(l1, l2, l3, f=Z, v2=X), Region(l1, l2, l3, f=Y, v2=X)]


# --------------------------------------------------------------------------


def main() -> None:
    argparse.ArgumentParser().parse_args()
    rows = []
    rng = np.random.default_rng(0)

    # 1 -----------------------------------------------------------------
    print("1. the two-angle parameterization reproduces the quadratic form\n")
    r = Region(1.2e-3, 0.63e-3, 0.37e-3, f=Z, v2=X)
    worst = 0.0
    for _ in range(20000):
        al, be = rng.uniform(0, 90), rng.uniform(-90, 90)
        worst = max(worst, abs(r.D(r.axis(al, be)) - r.D_param(al, be)))
    print(f"   worst |direct - parameterized| over 20000 axes: {worst:.2e}")
    rows.append(dict(check="two-angle form vs quadratic form", value=worst))

    # and the inverse map
    worst_inv = 0.0
    for _ in range(5000):
        al, be = rng.uniform(0, 90), rng.uniform(0, 89)
        a2, b2 = r.angles_of(r.axis(al, be))
        worst_inv = max(worst_inv, abs(a2 - al), abs(b2 - be))
    print(f"   worst angle recovery error:                     {worst_inv:.2e}\n")
    rows.append(dict(check="inverse map recovers the angles", value=worst_inv))

    # 2 -----------------------------------------------------------------
    print("2. the denominator built as n x f never leaks\n")
    worst_beta = 0.0
    for _ in range(5000):
        al, be = rng.uniform(0, 90), rng.uniform(-89, 89)
        n = r.axis(al, be)
        d = tract_locked_denominator(r, n)
        _, bd = r.angles_of(d)
        worst_beta = max(worst_beta, abs(bd))
    print(f"   worst denominator beta, for any numerator tilt: {worst_beta:.2e}")
    rows.append(dict(check="tract-locked denominator has zero beta",
                     value=worst_beta))
    _, bd = r.angles_of(tract_locked_denominator(r, r.axis(30, 40)))
    ad, _ = r.angles_of(tract_locked_denominator(r, r.axis(30, 40)))
    print(f"   its in-plane angle at alpha = 30:               {ad:.4f} deg\n")

    # 3 -----------------------------------------------------------------
    print("3. beta is the only route past the bound\n")
    print(f"   {'beta':>6} {'R/rho':>9}  numerator tilt, alpha = 0")
    regs = classic_pair()
    rho = regs[0].rho
    for be in (0, 2, 5, 10, 20):
        na = [g.axis(0.0, be) for g in regs]
        da = [tract_locked_denominator(g, n) for g, n in zip(regs, na)]
        R = index(regs, na, da)
        flag = "" if R <= rho + 1e-12 else "   exceeds rho"
        print(f"   {be:6d} {R / rho:9.5f}{flag}")
        rows.append(dict(check=f"numerator tilt beta={be}", value=R / rho))

    print()
    print(f"   {'beta':>6} {'R/rho':>9}  denominator tilt instead")
    for be in (0, 2, 5, 10, 20):
        na = [g.axis(0.0, 0.0) for g in regs]
        da = [g.axis(90.0, be) for g in regs]
        R = index(regs, na, da)
        print(f"   {be:6d} {R / rho:9.5f}")
        rows.append(dict(check=f"denominator tilt beta={be}", value=R / rho))
    print("\n   A numerator tilt raises the index above rho, a denominator tilt")
    print("   lowers it. The bound is a statement about beta, not about alpha.\n")

    # 4 -----------------------------------------------------------------
    print("4. with every beta zero, N regions enter only through S\n")
    worst_S = 0.0
    for N in (2, 3, 4, 6):
        for _ in range(2000):
            al = rng.uniform(0, 90, N)
            gs = [Region(1.2e-3, 0.63e-3, 0.37e-3,
                         f=Z, v2=X) for _ in range(N)]
            na = [g.axis(a) for g, a in zip(gs, al)]
            da = [tract_locked_denominator(g, n) for g, n in zip(gs, na)]
            S = float((np.sin(np.radians(al)) ** 2).sum())
            worst_S = max(worst_S,
                          abs(index(gs, na, da) - R_of_S(S, gs[0].rho, N)))
    print(f"   worst |general - R(S, N)| over N = 2,3,4,6:     {worst_S:.2e}")
    rows.append(dict(check="R(S,N) reduction", value=worst_S))
    print("   How the misalignment distributes across regions does not matter.\n")

    # 5 -----------------------------------------------------------------
    print("5. per-region axes dissolve the over-determination\n")
    # two regions whose v2 directions disagree by delta
    for delta in (0, 10, 14.62, 17.14, 25):
        a = Region(1.2e-3, 0.63e-3, 0.37e-3, f=Z, v2=X)
        tilt = np.radians(delta)
        v2b = np.cos(tilt) * X + np.sin(tilt) * Y
        b = Region(1.2e-3, 0.63e-3, 0.37e-3, f=Z, v2=v2b)
        # shared axis, best case: the bisector
        bis = a.v2 + b.v2
        bis /= np.linalg.norm(bis)
        na_s = [bis, bis]
        da_s = [tract_locked_denominator(g, bis) for g in (a, b)]
        shared = index([a, b], na_s, da_s)
        # per-region axes, each on its own v2
        na_p = [a.v2, b.v2]
        da_p = [tract_locked_denominator(g, n) for g, n in zip((a, b), na_p)]
        per = index([a, b], na_p, da_p)
        rows.append(dict(check=f"shared axis at delta={delta}",
                         value=shared / a.rho))
        rows.append(dict(check=f"per-region axes at delta={delta}",
                         value=per / a.rho))
        print(f"   delta {delta:6.2f}:  shared {shared / a.rho:.5f}   "
              f"per-region {per / a.rho:.5f}")
    print("\n   Per-region axes reach rho at every separation. The whole cost of")
    print("   the two-region design is the decision to share one axis, which the")
    print("   perivascular premise motivates and nothing else requires.\n")

    # 6 -----------------------------------------------------------------
    print("6. is the tract-locked construction already immune to")
    print("   non-orthogonality, and can the residual be corrected?\n")

    worst_ortho = 0.0
    for inter in (90, 80, 70, 60, 45):
        # two fibers separated by inter degrees, neither along a scanner axis
        th = np.radians(inter)
        fa = Z
        fb = np.cos(th) * Z + np.sin(th) * Y
        a = Region(1.2e-3, 0.63e-3, 0.37e-3, f=fa, v2=X)
        b = Region(1.2e-3, 0.63e-3, 0.37e-3, f=fb, v2=X)
        n = np.cross(a.f, b.f)
        n /= np.linalg.norm(n)
        da, db = (tract_locked_denominator(g, n) for g in (a, b))
        _, bna = a.angles_of(n)
        _, bnb = b.angles_of(n)
        _, bda = a.angles_of(da)
        _, bdb = b.angles_of(db)
        worst_ortho = max(worst_ortho, abs(bna), abs(bnb), abs(bda), abs(bdb))
        R = index([a, b], [n, n], [da, db])
        print(f"   tracts {inter:3d} deg apart:  worst beta "
              f"{max(abs(bna), abs(bnb), abs(bda), abs(bdb)):.2e}   "
              f"R/rho = {R / a.rho:.6f}")
    rows.append(dict(check="tract-locked beta under non-orthogonality",
                     value=worst_ortho))
    print("\n   Every beta is zero at every separation, so the cross-product")
    print("   construction needs no correction for non-orthogonality. The bound")
    print("   holds for any non-parallel pair.\n")

    # and the inversion, for the residual in-plane error
    print("   Inverting R(S) for rho, given S\n")
    print(f"   {'delta':>7} {'S':>8} {'R/rho':>9} {'recovered rho':>15} {'err':>10}")
    rho_true = 0.63 / 0.37
    for deg in (0, 10, 14.62, 20, 30, 45):
        S = 1 - np.cos(np.radians(deg))
        R = R_of_S(S, rho_true)
        got = rho_from_R(R, S)
        print(f"   {deg:7.2f} {S:8.5f} {R / rho_true:9.5f} {got:15.6f} "
              f"{abs(got - rho_true):10.2e}")
        rows.append(dict(check=f"rho recovered at delta={deg}", value=got))
    print("\n   The inversion is exact. It is also circular: S needs v2, and")
    print("   anything that has v2 can form lambda2/lambda3 directly, with less")
    print("   noise than inverting an attenuated index.\n")

    out = pd.DataFrame(rows)
    out.to_csv(HERE / "generalized_alps_model.csv", index=False)
    print(f"   wrote generalized_alps_model.csv, {len(out)} rows")


if __name__ == "__main__":
    main()
