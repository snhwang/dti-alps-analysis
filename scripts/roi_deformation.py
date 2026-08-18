"""
What the atlas regions become after the inverse warp.

The measurement regions are spheres in template space. They are used in native
space, reached through the inverse of a nonlinear registration, and a nonlinear
map is only locally affine: a sphere acquires whatever anisotropic scaling and
shear the Jacobian carries at that location. So the regions are not spheres
where they are actually applied, and their volume is set per participant by the
registration rather than by the radius.

This measures both. Each warped region is reduced to its equivalent ellipsoid,
whose semi-axes come from the eigenvalues of the coordinate covariance in world
millimetres, and the delivered volume is compared with the nominal one.

Noticed because the regions looked wrong in a figure, which is the only reason
it was checked.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from data_paths import winpath, refined_rois

import atomic_io  # noqa: F401  writes become atomic on import

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
DIFF = HERE.parent.parent / "diffusion"
OUT = winpath("Q:/dti_output")
TEMPLATE = refined_rois() / "L_SCR.nii.gz"
ROIS = ("L_SCR", "R_SCR", "L_SLF", "R_SLF")
COHORTS = (("DLBS", "HCP/lifespan_alps_results.csv", "DTI_Session_ID"),
           ("HCP-A", "HCP/alps_results_2026-02-22.csv", "Session_ID"))


def ellipsoid(path: Path):
    """Voxel count, semi-axes in mm, and volume of one warped region."""
    import nibabel as nib
    im = nib.load(str(path))
    m = im.get_fdata() > 0.5
    if m.sum() < 10:
        return None
    idx = np.argwhere(m).astype(float)
    A = im.affine
    w = (A[:3, :3] @ idx.T).T + A[:3, 3]
    # semi-axes of the ellipsoid with the same second moments as a uniform body
    axes = np.sqrt(5 * np.linalg.eigvalsh(np.cov((w - w.mean(0)).T))[::-1])
    vol = float(m.sum() * abs(np.linalg.det(A[:3, :3])))
    return int(m.sum()), axes, vol


def main() -> None:
    ref = ellipsoid(TEMPLATE)
    print(f"template sphere: axes {ref[1].round(2)} mm, {ref[2]:.0f} mm3, "
          f"elongation {ref[1][0]/ref[1][2]:.2f}\n")

    rows = []
    for tag, src, idc in COHORTS:
        d = pd.read_csv(DIFF / src).dropna(subset=[idc])
        for r in d.itertuples():
            sd = OUT / str(getattr(r, idc)) / "processed" / "atlas" / "sphere_roi"
            for n in ROIS:
                f = sd / f"{n}_native.nii.gz"
                if not f.exists():
                    continue
                e = ellipsoid(f)
                if e is None:
                    continue
                nv, axes, vol = e
                rows.append({"cohort": tag, "sid": str(getattr(r, idc)), "roi": n,
                             "a": axes[0], "b": axes[1], "c": axes[2],
                             "elongation": axes[0] / axes[2], "volume_mm3": vol})

    t = pd.DataFrame(rows)
    t.to_csv(HERE / "roi_deformation.csv", index=False)
    print(f"{len(t)} warped regions across {t.sid.nunique()} sessions\n")
    for c, g in t.groupby("cohort"):
        print(f"{c}")
        print(f"  elongation  median {g.elongation.median():.2f}  "
              f"range {g.elongation.min():.2f} to {g.elongation.max():.2f}")
        print(f"  volume      median {g.volume_mm3.median():.0f} mm3  "
              f"range {g.volume_mm3.min():.0f} to {g.volume_mm3.max():.0f}  "
              f"CV {100*g.volume_mm3.std()/g.volume_mm3.mean():.0f}%")
        print(f"  nominal     {ref[2]:.0f} mm3, so delivered volume is "
              f"{100*g.volume_mm3.median()/ref[2]:.0f}% of nominal\n")
    print(f"wrote {HERE / 'roi_deformation.csv'}")


if __name__ == "__main__":
    main()
