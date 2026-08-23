"""The regression identity worked through on DLBS, as the paper's example.

Section sec:diagnostic gives readers a way to estimate how much of a published
ALPS association is postural without recomputing an index:

    beta_{y|g,p} = beta_{y|g} - beta_{y|p,g} * beta_{p|g}

with g the variable of interest, p a pose measure and y the index. The example
that accompanied it was drawn from the patient cohort. DLBS supplies the same
worked example on age, which is the variable the identity is most likely to be
applied to, and it is the cohort where pose genuinely covaries with the
contrast.

All terms are standardized so they read as correlations, and the identity is
checked numerically rather than assumed, since an identity that does not close
means the terms were not computed consistently.

    python diagnostic_worked_example.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def z(v):
    v = np.asarray(v, float)
    return (v - v.mean()) / v.std(ddof=1)


def beta(y, X):
    """Standardized coefficient on the first column of X, with an intercept."""
    A = np.column_stack([np.ones(len(y))] + [z(c) for c in X])
    b, *_ = np.linalg.lstsq(A, z(y), rcond=None)
    return float(b[1])


def main() -> None:
    argparse.ArgumentParser().parse_args()

    d = pd.read_csv(HERE / "measured_pvs_axis_dlbs.csv")
    h = pd.read_csv(HERE / "head_rotation_dlbs.csv")
    for f in (d, h):
        f["Subject_ID"] = f.Subject_ID.astype(str)
        f["Visit"] = f.Visit.astype(str)
    m = d.merge(h[["Subject_ID", "Visit", "pitch", "total"]],
                on=["Subject_ID", "Visit"], how="inner")
    m["abs_pitch"] = m.pitch.abs()
    # One session per participant, matching every other age association reported.
    m = m.sort_values(["Subject_ID", "Visit"]).groupby("Subject_ID").first().reset_index()
    m = m.dropna(subset=["classic", "Age", "abs_pitch"])
    print(f"DLBS: {len(m)} participants\n")

    g = m.Age.to_numpy(float)
    p = m.abs_pitch.to_numpy(float)
    y = m.classic.to_numpy(float)

    b_yg = beta(y, [g])                 # unadjusted age association
    b_pg = beta(p, [g])                 # does pose covary with age
    b_ygp = beta(y, [g, p])             # age association after pose
    b_ypg = beta(y, [p, g])             # pose-to-index, given age

    print("=== the identity's terms, standardized ===")
    print(f"   beta_y|g    unadjusted age association      {b_yg:+.3f}")
    print(f"   beta_p|g    pose against age                {b_pg:+.3f}")
    print(f"   beta_y|p,g  pose against index, given age   {b_ypg:+.3f}")
    print(f"   beta_y|g,p  age association after pose      {b_ygp:+.3f}")

    predicted = b_yg - b_ypg * b_pg
    print(f"\n=== does the identity close? ===")
    print(f"   beta_y|g - beta_y|p,g * beta_p|g = {b_yg:+.3f} - "
          f"({b_ypg:+.3f})({b_pg:+.3f}) = {predicted:+.3f}")
    print(f"   directly computed beta_y|g,p     = {b_ygp:+.3f}")
    ok = abs(predicted - b_ygp) < 0.005
    print(f"   {'closes' if ok else 'DOES NOT CLOSE, terms are inconsistent'}"
          f"   (difference {abs(predicted - b_ygp):.4f})")

    frac = 100 * (1 - b_ygp / b_yg)
    print(f"\n   fraction of the age association carried by pose: {frac:.0f}%")

    pd.DataFrame([{"cohort": "dlbs", "n": len(m), "beta_y_g": b_yg,
                   "beta_p_g": b_pg, "beta_y_pg": b_ypg, "beta_y_gp": b_ygp,
                   "product": b_ypg * b_pg, "pct_pose": frac}]).to_csv(
        HERE / "diagnostic_worked_example.csv", index=False)
    print(f"\n   wrote diagnostic_worked_example.csv")


if __name__ == "__main__":
    main()
