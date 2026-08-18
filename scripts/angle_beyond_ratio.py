"""Does the axis angle carry age signal beyond the eigenvalue ratio, and beyond
the anatomical departure already reported?

Given the tensor, a fiber-locked ALPS index is a function of lambda2, lambda3
and one angle alpha, the angle between the measurement axis and the second
eigenvector (Section 2). Conditional on lambda2/lambda3, the only remaining
channel is alpha. Section 3 shows that no index variant retains an association
once the ratio is partialled out. This asks the complementary question about the
angle itself, which the index discards.

Three nested adjustments, because a positive result has two innocent
explanations that have to be excluded:

  ratio            alpha against age, adjusting for lambda2/lambda3
  ratio + theta    additionally adjusting for theta_SCR and theta_SLF, the
                   departure of the measured tract directions from the scanner
                   axes, which the paper already reports at 8 to 10 degrees.
                   If alpha survives this, it is not merely re-expressing that.
  ratio + theta + pose   additionally adjusting for absolute pitch and total
                   head rotation, so a residual cannot be posture.

Reading it. alpha is measured against the scanner x axis, which knows nothing
about the participant, so an association can only mean the tissue side moved:
the direction of fastest perpendicular diffusion rotates relative to anatomy.
That is a geometric statement, not a perivascular one, since Schilling et al.
report that the second eigenvector does not generally follow vasculature.

Benjamini-Hochberg across every test reported here, since the question was asked
after seeing one of them.

    python angle_beyond_ratio.py

Writes angle_beyond_ratio.csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import atomic_io  # noqa: F401  writes become atomic on import
from scipy import stats

HERE = Path(__file__).resolve().parent
DIFF = HERE.parent.parent / "diffusion"
ANGLES = ("v2_to_x", "v2_to_cross", "cross_to_x")


def partial(y, x, covs):
    C = np.column_stack([np.ones(len(y))] + [np.asarray(c, float) for c in covs])

    def rz(v):
        b, *_ = np.linalg.lstsq(C, np.asarray(v, float), rcond=None)
        return np.asarray(v, float) - C @ b

    a, b = rz(x), rz(y)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan"), float("nan")
    r = float(np.corrcoef(a, b)[0, 1])
    dof = len(y) - C.shape[1] - 1
    return r, float(2 * stats.t.sf(abs(r * np.sqrt(dof / max(1 - r * r, 1e-12))), dof))


def fdr(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    q = np.empty_like(p)
    prev = 1.0
    n = len(p)
    for rank, i in enumerate(order[::-1], 1):
        prev = min(prev, p[i] * n / (n - rank + 1))
        q[i] = prev
    return q


def load(cohort):
    if cohort == "hcpa":
        d = pd.read_csv(HERE / "measured_pvs_axis_hcpa_b1500_all.csv")
        dev = pd.read_csv(DIFF / "HCP" / "alps_axis_deviations.csv")
        pose = pd.read_csv(HERE / "head_rotation_hcpa.csv")
    else:
        d = pd.read_csv(HERE / "measured_pvs_axis_dlbs.csv")
        dev = pd.read_csv(DIFF / "DLBS" / "dlbs_alps_spheres_axis_deviations.csv")
        pose = pd.read_csv(HERE / "head_rotation_dlbs.csv")
    dev = dev.rename(columns={"Session": "Visit"})
    for f in (d, dev, pose):
        f["Subject_ID"] = f.Subject_ID.astype(str)
        f["Visit"] = f.Visit.astype(str)
    dev["theta_SCR"] = dev[["theta_SCR_L", "theta_SCR_R"]].mean(axis=1)
    dev["theta_SLF"] = dev[["theta_SLF_L", "theta_SLF_R"]].mean(axis=1)
    m = d.merge(dev[["Subject_ID", "Visit", "theta_SCR", "theta_SLF"]],
                on=["Subject_ID", "Visit"], how="left")
    m = m.merge(pose[["Subject_ID", "Visit", "pitch", "total"]],
                on=["Subject_ID", "Visit"], how="left")
    m["abs_pitch"] = m.pitch.abs()
    return m.sort_values(["Subject_ID", "Visit"]).groupby("Subject_ID").first().reset_index()


def main() -> None:
    argparse.ArgumentParser().parse_args()
    rows = []
    for cohort in ("hcpa", "dlbs"):
        m = load(cohort)
        for ang in ANGLES:
            if ang not in m.columns:
                continue
            for label, covs in (("ratio", ["pv_perp"]),
                                ("ratio+theta", ["pv_perp", "theta_SCR", "theta_SLF"]),
                                ("ratio+theta+pose",
                                 ["pv_perp", "theta_SCR", "theta_SLF",
                                  "abs_pitch", "total"])):
                s = m[[ang, "Age"] + covs].replace([np.inf, -np.inf], np.nan).dropna()
                if len(s) < 40:
                    continue
                r, p = partial(s.Age, s[ang], [s[c] for c in covs])
                rows.append(dict(cohort=cohort, angle=ang, adjustment=label,
                                 r=round(r, 4), p=p, n=len(s)))

    out = pd.DataFrame(rows)
    out["q"] = fdr(out.p.to_numpy())
    out.to_csv(HERE / "angle_beyond_ratio.csv", index=False)

    for coh, g in out.groupby("cohort", sort=False):
        print(f"\n{coh}, n up to {int(g.n.max())}\n")
        print(f"   {'angle':12s} {'adjustment':18s} {'r':>8s} {'p':>10s} {'q':>8s}")
        for r in g.itertuples():
            star = "  *" if r.q < 0.05 else ""
            print(f"   {r.angle:12s} {r.adjustment:18s} {r.r:+8.4f} "
                  f"{r.p:10.3g} {r.q:8.3f}{star}")

    surv = out[(out.angle == "v2_to_x") & (out.adjustment == "ratio+theta+pose")]
    print("\n   v2_to_x under the fullest adjustment:")
    for r in surv.itertuples():
        print(f"     {r.cohort:6s} r = {r.r:+.4f}, q = {r.q:.3f}, n = {r.n}")
    print("\n   Surviving means the angle is not the anatomical departure already")
    print("   reported, and not posture. It would remain a geometric quantity, not")
    print("   a perivascular one.")


if __name__ == "__main__":
    main()
