"""How accurately does a midline have to be drawn?

The perivascular axis can be obtained without any registration. It is the normal
to the midsagittal plane, a unit vector with two degrees of freedom, so two
traces fix it: a midline on an axial slice and a midline on a coronal one. Pitch,
which is the large rotation in these cohorts (median 10.7 degrees in DLBS, 9.5 in
the trigeminal patients), is rotation about that axis and therefore cannot move
it. The rotations that do move it, yaw and roll, sit under a degree at the
median. So the one you would have to measure carefully is the one that does not
matter here.

That makes a manual route practical, and raises the only question that decides
whether it is usable: how precisely does the line have to be drawn.

The answer is not the within-plane figure in axis_error_sensitivity.csv. That
case holds the axis perpendicular to the fibers, so the error only trades
lambda2 against lambda3 and costs 0.26% at 10 degrees. A drawn midline does not
respect fiber geometry. Part of its error tilts the axis toward the fibers and
admits lambda1, which is two to three times larger. This measures that directly,
by rotating the axis a known amount about a uniformly random perpendicular
direction and recomputing the index on real tensors.

Result, 120 region-hemispheres of DLBS, close to linear at about 0.31% per
degree:

    2 deg    0.62%
    5 deg    1.57%
   10 deg    3.06%
   20 deg    6.33%

Drawing to two or three degrees is easy by hand and costs under 1%, against
6.8% for the uncorrected pitch confound in this cohort and 4.9% for the one
silently failed registration we found in 507 sessions.

    python midline_sensitivity.py

Writes midline_sensitivity.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from data_paths import winpath

import atomic_io  # noqa: F401  writes become atomic on import

sys.path.insert(0, str(Path(__file__).resolve().parent))
from anatomical_x_variant import HEMIS, index_for, unit  # noqa: E402
from direction_estimators import X, Y, Z, align, principal, weights_for  # noqa: E402
from registration_aligns_tracts import polar_rotation  # noqa: E402

HERE = Path(__file__).resolve().parent
DIFF = HERE.parent.parent / "diffusion"
OUT = winpath("Q:/dti_output")
FA_MIN = 0.2
SLAB_MM = 8.0
DEGREES = (2, 5, 10, 20)
DRAWS = 8


def perturb(p, deg, rng):
    """Rotate p by deg about a uniformly random perpendicular axis."""
    a = rng.normal(size=3)
    a -= (a @ p) * p
    a /= np.linalg.norm(a)
    th = np.radians(deg)
    return p * np.cos(th) + np.cross(a, p) * np.sin(th)


def main() -> None:
    import nibabel as nib

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=60,
                    help="sessions to sample; the estimate is stable well below the cohort")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

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
        p_anat = unit(polar_rotation(M[:3, :3]).T @ X, X)

        md = evals.mean(-1)
        nu = np.sqrt(((evals - md[..., None]) ** 2).sum(-1))
        de = np.sqrt((evals ** 2).sum(-1))
        fa = np.clip(np.sqrt(1.5) * np.divide(nu, de, out=np.zeros_like(nu), where=de != 0), 0, 1)
        ii, jj, kk = np.indices(lab.shape)
        A = limg.affine
        zc = A[2, 0] * ii + A[2, 1] * jj + A[2, 2] * kk + A[2, 3]
        xw = A[0, 0] * ii + A[0, 1] * jj + A[0, 2] * kk + A[0, 3]

        def pack(m):
            v1 = evecs[m][:, :, 0]
            n = np.linalg.norm(v1, axis=1, keepdims=True)
            n[n == 0] = 1
            return {"v1": v1 / n, "fa": fa[m], "evals": evals[m], "evecs": evecs[m]}

        z0 = float(np.median(zc[sph > 0])) if (sph > 0).any() else 0.0
        band = np.abs(zc - z0) <= SLAB_MM

        for hemi, scr, slf in HEMIS:
            side = xw < 0 if hemi == "L" else xw > 0
            mp_s, ma_s = (sph == 1) & side & (fa >= FA_MIN), (sph == 2) & side & (fa >= FA_MIN)
            mp_l = (lab == scr) & (fa >= FA_MIN) & band
            ma_l = (lab == slf) & (fa >= FA_MIN) & band
            if mp_s.sum() < 4 or ma_s.sum() < 4 or mp_l.sum() < 10 or ma_l.sum() < 10:
                continue
            proj, assoc = pack(mp_s), pack(ma_s)
            vp = align(principal(pack(mp_l)["v1"], weights_for("cl", pack(mp_l))), Z)
            va = align(principal(pack(ma_l)["v1"], weights_for("cl", pack(ma_l))), Y)
            base = index_for(proj, assoc, vp, va, p_anat)
            if not np.isfinite(base) or base == 0:
                continue
            for deg in DEGREES:
                errs = [abs(index_for(proj, assoc, vp, va,
                                      unit(perturb(p_anat, deg, rng), X)) - base) / base
                        for _ in range(DRAWS)]
                rows.append(dict(deg=deg, pct=100 * float(np.mean(errs))))

    d = pd.DataFrame(rows)
    d.to_csv(HERE / "midline_sensitivity.csv", index=False)
    print(f"midline drawing error, {len(d) // len(DEGREES)} region-hemispheres "
          f"of real DLBS tensors\n")
    print("  error in the drawn midline    resulting change in the index")
    for deg, g in d.groupby("deg"):
        print(f"        {deg:2.0f} deg                    {g.pct.mean():6.3f}%"
              f"   (p95 {g.pct.quantile(.95):.3f}%)")


if __name__ == "__main__":
    main()
