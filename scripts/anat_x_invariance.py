"""Is the anatomical-axis variant exactly invariant under head rotation?

Every corrected variant in this paper is checked numerically rather than
asserted, and anat_x needs the check more than the others because its invariance
has a condition attached.

The perivascular axis is R'x, with R the rotation of the subject-to-template
affine. If the head is rotated by Q, the tensors become Q D Q' and the new
registration finds R_new = R Q', so the axis becomes

    R_new' x = Q R' x = Q p

The axis rotates with the head, the measurement triad rotates with it, and the
ratio is unchanged. Invariance is therefore exact, but only if the registration
is recomputed for the rotated head. Rotating the tensors while holding the old
affine fixed would make anat_x look badly non-invariant, and that would be an
artifact of the test rather than a property of the method.

Both are run here, so the distinction is visible rather than assumed:

  rotated correctly   tensors by Q, axis by Q       expect exactly flat
  affine held fixed   tensors by Q, axis unchanged  expect drift like classic

The second is not a criticism of anat_x. It is what happens if someone reuses a
registration computed from a different acquisition, which is worth knowing.

    python anat_x_invariance.py

Writes anat_x_invariance.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import atomic_io  # noqa: F401  writes become atomic on import
from data_paths import winpath

sys.path.insert(0, str(Path(__file__).resolve().parent))
from anatomical_x_variant import HEMIS, classic_index, index_for, unit  # noqa: E402
from direction_estimators import X, Y, Z, align, principal, weights_for  # noqa: E402
from registration_aligns_tracts import polar_rotation  # noqa: E402

HERE = Path(__file__).resolve().parent
DIFF = HERE.parent.parent / "diffusion"
OUT = winpath("Q:/dti_output")
FA_MIN, SLAB_MM = 0.2, 8.0
ANGLES = (10.0, 20.0, 30.0)


def rot(axis, deg):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    th = np.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def main() -> None:
    import nibabel as nib

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=25)
    args = ap.parse_args()

    src = pd.read_csv(DIFF / "DLBS" / "dlbs_alps_spheres_5mm.csv")
    src = src[src.status == "ok"].head(args.limit)

    rows = []
    for r in src.itertuples():
        sd = OUT / r.DTI_Session_ID / "processed"
        try:
            limg = nib.load(str(sd / "atlas" / "jhu_labels_registered.nii.gz"))
            lab = limg.get_fdata().astype(int)
            sph = nib.load(str(sd / "atlas" / "sphere_roi"
                               / "sphere_roi_combined.nii.gz")).get_fdata().astype(int)
            evals = nib.load(str(sd / "tensor_eigenvalues.nii.gz")).get_fdata()
            evecs = nib.load(str(sd / "tensor_eigenvectors.nii.gz")).get_fdata()
            M = np.loadtxt(sd / "atlas" / "subject_to_mni_affine.mat")
        except Exception:
            continue
        p0 = unit(polar_rotation(M[:3, :3]).T @ X, X)

        md = evals.mean(-1)
        nu = np.sqrt(((evals - md[..., None]) ** 2).sum(-1))
        de = np.sqrt((evals ** 2).sum(-1))
        fa = np.clip(np.sqrt(1.5) * np.divide(nu, de, out=np.zeros_like(nu), where=de != 0), 0, 1)
        ii, jj, kk = np.indices(lab.shape)
        A = limg.affine
        zc = A[2, 0] * ii + A[2, 1] * jj + A[2, 2] * kk + A[2, 3]
        xw = A[0, 0] * ii + A[0, 1] * jj + A[0, 2] * kk + A[0, 3]
        z0 = float(np.median(zc[sph > 0])) if (sph > 0).any() else 0.0
        band = np.abs(zc - z0) <= SLAB_MM

        for hemi, scr, slf in HEMIS:
            side = xw < 0 if hemi == "L" else xw > 0
            mp_s, ma_s = (sph == 1) & side & (fa >= FA_MIN), (sph == 2) & side & (fa >= FA_MIN)
            mp_l = (lab == scr) & (fa >= FA_MIN) & band
            ma_l = (lab == slf) & (fa >= FA_MIN) & band
            if mp_s.sum() < 4 or ma_s.sum() < 4 or mp_l.sum() < 10 or ma_l.sum() < 10:
                continue

            def pack(m, R=None):
                v1 = evecs[m][:, :, 0]
                n = np.linalg.norm(v1, axis=1, keepdims=True)
                n[n == 0] = 1
                ec = evecs[m]
                if R is not None:
                    ec = np.einsum("ij,vjk->vik", R, ec)
                    v1 = ec[:, :, 0]
                return {"v1": v1 / np.maximum(np.linalg.norm(v1, axis=1, keepdims=True), 1e-12),
                        "fa": fa[m], "evals": evals[m], "evecs": ec}

            def measure(R):
                P, Aa = pack(mp_s, R), pack(ma_s, R)
                vp = align(principal(pack(mp_l, R)["v1"], weights_for("cl", pack(mp_l, R))), Z)
                va = align(principal(pack(ma_l, R)["v1"], weights_for("cl", pack(ma_l, R))), Y)
                return P, Aa, vp, va

            P0, A0, vp0, va0 = measure(None)
            base = index_for(P0, A0, vp0, va0, p0)
            base_classic = classic_index(P0, A0)
            for deg in ANGLES:
                Q = rot([1, 1, 1], deg)
                P1, A1, vp1, va1 = measure(Q)
                # correct: the registration is recomputed, so the axis rotates too
                good = index_for(P1, A1, vp1, va1, unit(Q @ p0, X))
                # wrong: an affine reused from another acquisition
                stale = index_for(P1, A1, vp1, va1, p0)
                rows.append(dict(deg=deg, hemi=hemi,
                                 anat_x_rotated=100 * abs(good - base) / base,
                                 anat_x_stale=100 * abs(stale - base) / base,
                                 classic=100 * abs(classic_index(P1, A1) - base_classic)
                                 / base_classic))

    d = pd.DataFrame(rows)
    d.to_csv(HERE / "anat_x_invariance.csv", index=False)
    print(f"DLBS, {len(d) // len(ANGLES)} region-hemispheres, rotation about (1,1,1)\n")
    print("  rotation   anat_x, axis rotated   anat_x, affine stale   classic")
    for deg, g in d.groupby("deg"):
        print(f"    {deg:4.0f} deg        {g.anat_x_rotated.max():.2e}%"
              f"              {g.anat_x_stale.median():6.2f}%        {g.classic.median():6.2f}%")
    print()
    print("  The first column is machine precision, so the variant is exactly")
    print("  invariant when the registration is recomputed. The second is what")
    print("  reusing a stale affine costs, and it is the same order as classic.")


if __name__ == "__main__":
    main()
