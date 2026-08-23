"""How well did each session register, and did that vary within a participant?

anat_x takes its perivascular axis from the subject-to-template registration, so
it inherits registration behavior in a way the rotation-invariant ratio does
not. That matters for one specific result: anat_x retains an association with
MoCA and crystallized IQ after the ratio and practice effects are partialled
out. If a participant's registration degrades on the same visit their cognition
dips, the association is manufactured.

Head pose is already tested and does not explain it, but pose and registration
quality are different quantities. Pose is the rotation in the affine. Quality is
whether the affine is a sensible one at all.

Three measures, all from the linear part A of the affine, via the polar
decomposition A = R S. R is the rotation, already used for pose. S is the
symmetric stretch, and it is where quality lives:

    det       determinant of A, the global volume scaling. A brain is not twice
              or half the template, so departures flag a failed fit.
    aniso     ratio of largest to smallest singular value. Isotropic scaling is
              near 1; a stretched fit is not.
    shear     off-diagonal magnitude of S relative to its diagonal. A rigid
              anatomical difference produces none.

The within-participant analysis needs one more step. A person whose head is
simply unusual has an odd affine at every visit, which cancels under
within-person centering and cannot confound anything. What can confound it is a
session that registered differently from that person's own others, so each
measure is also expressed as a deviation from the participant's own mean.

    python registration_quality.py --cohort hcpa
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from data_paths import winpath

HERE = Path(__file__).resolve().parent
DIFF = HERE.parent.parent / "diffusion"
OUT = winpath("Q:/dti_output")


def quality(mat: np.ndarray) -> dict:
    """Scale and shear of the affine's linear part, via polar decomposition."""
    A = mat[:3, :3]
    U, sv, Vt = np.linalg.svd(A)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    S = R.T @ A                      # symmetric stretch, A = R S
    S = 0.5 * (S + S.T)              # symmetrize against round-off
    diag = np.abs(np.diag(S)).mean()
    off = np.sqrt((S[np.triu_indices(3, 1)] ** 2).sum())
    return {"det": float(np.linalg.det(A)),
            "aniso": float(sv.max() / max(sv.min(), 1e-9)),
            "shear": float(off / max(diag, 1e-9))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=["hcpa", "dlbs"], default="hcpa")
    args = ap.parse_args()

    if args.cohort == "hcpa":
        src = pd.read_csv(DIFF / "HCP" / "hcpa_alps_spheres_5mm.csv")
    else:
        src = pd.read_csv(DIFF / "DLBS" / "dlbs_alps_spheres_5mm.csv")
        src["Visit"] = src["Session"]
    src = src[src.status == "ok"]

    rows, missing = [], 0
    for r in src.itertuples():
        m = OUT / r.DTI_Session_ID / "processed" / "atlas" / "subject_to_mni_affine.mat"
        if not m.exists():
            missing += 1
            continue
        try:
            A = np.loadtxt(m)
            if A.shape != (4, 4):
                continue
        except Exception:                                       # noqa: BLE001
            continue
        rows.append({"Subject_ID": str(r.Subject_ID), "Visit": str(r.Visit),
                     **quality(A)})
    d = pd.DataFrame(rows)
    print(f"{len(d)} sessions with an affine, {missing} missing\n")

    for c in ("det", "aniso", "shear"):
        print(f"   {c:<7s} median {d[c].median():7.3f}  "
              f"IQR {d[c].quantile(.25):.3f}-{d[c].quantile(.75):.3f}  "
              f"range {d[c].min():.3f} to {d[c].max():.3f}")

    # Deviation from the participant's own mean is the part that can confound a
    # within-person result; the between-person part cancels under centering.
    g = d.groupby("Subject_ID")
    for c in ("det", "aniso", "shear"):
        d[f"{c}_dev"] = (d[c] - g[c].transform("mean")).abs()
    rep = d[d.Subject_ID.isin(g.size()[g.size() > 1].index)]
    print(f"\n   {rep.Subject_ID.nunique()} participants with repeats; "
          f"within-person absolute deviation:")
    for c in ("det", "aniso", "shear"):
        v = rep[f"{c}_dev"]
        print(f"      {c:<7s} median {v.median():.4f}  p95 {v.quantile(.95):.4f}  "
              f"max {v.max():.4f}")

    bad = d[(d.aniso > 1.6) | (d.det < 0.3) | (d.det > 3.0) | (d.shear > 0.3)]
    print(f"\n   sessions failing a plain sanity bound: {len(bad)} of {len(d)}")

    d.to_csv(HERE / f"registration_quality_{args.cohort}.csv", index=False)
    print(f"\n   wrote registration_quality_{args.cohort}.csv")


if __name__ == "__main__":
    main()
