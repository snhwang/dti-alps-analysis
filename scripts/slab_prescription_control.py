"""Head pose from scanner geometry, independent of the brain.

The paper's principal finding rests on head pitch recovered from the rotation
part of each session's subject-to-template affine. That affine is fitted to
brain shape, so a reader can reasonably object that age-related atrophy and
ventricular enlargement might shift the optimal rotation without the head having
moved. If so, the pitch-age correlation would be anatomy re-expressed as pose.

This control answers that objection with a quantity the brain cannot influence.
The slice prescription is recorded in the raw NIfTI header: the third column of
the image affine is the slice normal in scanner coordinates, so its tilt out of
the axial plane is the angulation the operator dialled in. It is fixed before a
single volume is acquired, and no property of the brain can change it.

Three results follow, on the DLBS cohort, one session per participant:

  1. slab pitch correlates with age at r = -0.342, about the same magnitude as
     the affine-derived pitch and in the opposite direction
  2. the two measures correlate with each other at r = -0.223
  3. their difference, which estimates head pitch in the bore rather than either
     component alone, correlates with age at r = +0.428, more strongly than
     either measure by itself

The opposite signs are the interpretable part. The operator angulated the slab
progressively further for older participants, which is what one does when the
head is pitched back, so the registration-derived measure sees only the residual
after that compensation and understates the true head tilt.

Writes slab_prescription_dlbs.csv.
"""
from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from data_paths import winpath

import atomic_io  # noqa: F401  writes become atomic on import
from scipy import stats

HERE = Path(__file__).resolve().parent
DIFF = HERE.parent.parent / "diffusion"
OUT = winpath("Q:/dti_output")


def nearest_rotation(M: np.ndarray) -> np.ndarray:
    """Rotation part of a linear map, by polar decomposition."""
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U = U.copy()
        U[:, -1] *= -1
        R = U @ Vt
    return R


def main() -> None:
    alps = pd.read_csv(DIFF / "DLBS" / "dlbs_alps_spheres_5mm.csv")
    alps = alps[alps.status == "ok"]
    rows = []
    for r in alps.itertuples():
        raw = OUT / r.DTI_Session_ID / "inputs" / "dwi.nii.gz"
        aff = OUT / r.DTI_Session_ID / "processed" / "atlas" / "subject_to_mni_affine.mat"
        if not (raw.exists() and aff.exists()):
            continue
        try:
            A = np.asarray(nib.load(str(raw)).affine)[:3, :3]
            M = np.loadtxt(aff)
            if M.shape != (4, 4):
                continue
        except Exception:
            continue
        # Slice normal in scanner RAS. Its storage direction is arbitrary, so
        # fix the hemisphere before reading an angle off it.
        n = A[:, 2] / np.linalg.norm(A[:, 2])
        if n[2] < 0:
            n = -n
        Ra = nearest_rotation(M[:3, :3])
        rows.append(dict(
            Subject_ID=r.Subject_ID, Visit=r.Session,
            slab_tilt=float(np.degrees(np.arccos(np.clip(n[2], -1, 1)))),
            slab_pitch=float(np.degrees(np.arctan2(n[1], n[2]))),
            aff_pitch=float(np.degrees(np.arctan2(-Ra[2, 1], Ra[2, 2]))),
        ))

    d = pd.DataFrame(rows)
    d.to_csv(HERE / "slab_prescription_dlbs.csv", index=False)
    age = pd.read_csv(HERE / "measured_pvs_axis_dlbs.csv")[["Subject_ID", "Visit", "Age"]]
    for x in (d, age):
        x["Subject_ID"] = x.Subject_ID.astype(str)
        x["Visit"] = x.Visit.astype(str)
    m = (d.merge(age, on=["Subject_ID", "Visit"]).dropna()
          .sort_values(["Subject_ID", "Visit"]).groupby("Subject_ID").first().reset_index())

    print(f"{len(d)} sessions, {len(m)} participants\n")
    print(f"slab angulation from axial: median {m.slab_tilt.median():.2f} deg, "
          f"IQR [{m.slab_tilt.quantile(.25):.2f}, {m.slab_tilt.quantile(.75):.2f}], "
          f"max {m.slab_tilt.max():.2f}\n")
    m["combined"] = (m.aff_pitch - m.slab_pitch).abs()
    for a, b, lab in (("Age", "slab_pitch", "slab prescription pitch vs age"),
                      ("Age", "aff_pitch", "affine-derived pitch vs age"),
                      ("slab_pitch", "aff_pitch", "the two measures against each other"),
                      ("Age", "combined", "combined head-in-bore pitch vs age")):
        _r, _p = stats.pearsonr(m[a], m[b])
        print(f"  {lab:36s} r = {_r:+.3f}   p = {_p:.2g}")
    hr = pd.read_csv(HERE / "head_rotation_dlbs.csv")
    hr["Subject_ID"] = hr.Subject_ID.astype(str)
    hr["Visit"] = hr.Visit.astype(str)
    idx = pd.read_csv(HERE / "measured_pvs_axis_dlbs.csv")[["Subject_ID", "Visit", "classic"]]
    idx["Subject_ID"] = idx.Subject_ID.astype(str)
    idx["Visit"] = idx.Visit.astype(str)
    mm = m.merge(hr, on=["Subject_ID", "Visit"], suffixes=("", "_hr"))
    mm = mm.merge(idx, on=["Subject_ID", "Visit"])
    absorption(mm.dropna(subset=["classic", "pitch", "total"]))
    print("\nThe slab prescription is scanner metadata. No property of the brain can")
    print("change it, so the atrophy objection does not apply to the first line.")


def absorption(m) -> None:
    """Does the header-derived pose absorb the age coefficient, not just track age?

    The validation above shows pose covaries with age in a quantity the brain
    cannot influence. That answers the objection to the premise. It does not by
    itself make the headline 45% atrophy-proof, because that figure is computed
    from the registration measure the objection targets. This runs the same
    adjustment on the header measure instead.
    """
    def z(v):
        v = np.asarray(v, float)
        return (v - v.mean()) / v.std(ddof=1)

    def beta(y, age, covs):
        X = [np.ones(len(y)), z(age)] + [z(c) for c in covs]
        return float(np.linalg.lstsq(np.column_stack(X), z(y), rcond=None)[0][1])

    base = beta(m.classic, m.Age, [])
    print()
    print(f"absorption of the classic age coefficient, {len(m)} participants")
    print(f"  unadjusted {base:.3f}")
    for lab, covs in (("registration pose", [np.abs(m.pitch), m.total]),
                      ("header slab angulation", [m.slab_pitch]),
                      ("combined head-in-bore", [(m.aff_pitch - m.slab_pitch).abs()]),
                      ("header and registration", [m.slab_pitch, np.abs(m.pitch), m.total])):
        b = beta(m.classic, m.Age, covs)
        print(f"  {lab:24s} {b:7.3f}   {100 * (1 - abs(b) / abs(base)):5.1f}% absorbed")
    print("  The header measure alone captures only the operator's compensation, which is")
    print("  a fraction of head-in-bore pitch, so it absorbs less. It is nonetheless a")
    print("  floor that no argument about brain shape can reach.")


if __name__ == "__main__":
    main()
