"""How much of a hand-drawn region points away from its own tract?

Figure 3 shows one session, and the caption has to say whether that session is
typical. This supplies the comparison.

The file the caption previously drew on, manual_roi_offtract_dlbs.csv, has no
generator anywhere in the tree, so its definition cannot be recovered and the
manuscript never states one. This defines the quantity explicitly instead.

Definition. Within a hand-drawn region, keep voxels with FA >= 0.2, the
threshold every analysis here applies. Take the region's own tract direction as
the principal eigenvector of the CL-weighted dyadic sum of those voxels'
principal eigenvectors, the same sign-invariant construction the refined index
uses for tract directions. A voxel is off-tract if its principal eigenvector
lies more than 45 degrees from that direction. The region's off-tract fraction
is the share of its voxels that are, and a session's figure is the worse of its
two regions.

The direction comes from the region itself rather than from an atlas label, so
the measure asks whether a hand-drawn region is internally coherent, not whether
it agrees with a template. That is the right question for a caption disclosing
how well placed the displayed session is, and it needs no free parameters beyond
the FA floor and the 45 degree threshold, both stated above.

    python manual_roi_offtract.py

Writes manual_roi_offtract.csv.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import atomic_io  # noqa: F401  writes become atomic on import
from data_paths import winpath

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
DIFF = HERE.parent.parent / "diffusion"
OUT = Path(winpath("Q:/dti_output"))
FA_MIN = 0.2
THRESH = 45.0
MIN_VOX = 10


def principal_direction(v1, cl):
    """Sign-invariant weighted mean direction: principal eigenvector of the dyadic sum."""
    w = np.clip(cl, 0, None)
    if w.sum() <= 0:
        w = np.ones(len(v1))
    T = (w[:, None, None] * np.einsum("ij,ik->ijk", v1, v1)).sum(0) / w.sum()
    return np.linalg.eigh(T)[1][:, -1]


def main() -> None:
    argparse.ArgumentParser().parse_args()
    import nibabel as nib

    src = pd.read_csv(DIFF / "HCP" / "lifespan_alps_results.csv")
    src = src.dropna(subset=["DTI_Session_ID"])
    rows = []
    for r in src.itertuples():
        sid = str(r.DTI_Session_ID)
        sd = OUT / sid / "processed"
        need = ("alps_rois_manual.nii.gz", "fa.nii.gz",
                "tensor_eigenvectors.nii.gz", "tensor_eigenvalues.nii.gz")
        if not all((sd / q).exists() for q in need):
            continue
        try:
            man = np.rint(nib.load(str(sd / "alps_rois_manual.nii.gz"))
                          .get_fdata()).astype(int)
            fa = nib.load(str(sd / "fa.nii.gz")).get_fdata()
            ev = nib.load(str(sd / "tensor_eigenvectors.nii.gz")).get_fdata()
            ea = nib.load(str(sd / "tensor_eigenvalues.nii.gz")).get_fdata()
        except Exception:
            continue

        rec = {"sid": sid}
        for val, name in ((1, "proj_off"), (2, "assoc_off")):
            m = (man == val) & (fa >= FA_MIN)
            if m.sum() < MIN_VOX:
                rec[name] = np.nan
                continue
            v1 = ev[m][:, :, 0]
            n = np.linalg.norm(v1, axis=1, keepdims=True)
            n[n == 0] = 1
            v1 = v1 / n
            lam = np.sort(ea[m], axis=-1)[:, ::-1]
            cl = (lam[:, 0] - lam[:, 1]) / np.maximum(lam[:, 0], 1e-12)
            axis = principal_direction(v1, cl)
            ang = np.degrees(np.arccos(np.clip(np.abs(v1 @ axis), 0, 1)))
            rec[name] = float((ang > THRESH).mean())
        rec["worst"] = float(np.nanmax([rec.get("proj_off", np.nan),
                                        rec.get("assoc_off", np.nan)]))
        rows.append(rec)

    d = pd.DataFrame(rows)
    d.to_csv(HERE / "manual_roi_offtract.csv", index=False)

    print(f"DLBS, {len(d)} sessions with a hand-drawn mask\n")
    for c in ("proj_off", "assoc_off", "worst"):
        print(f"   {c:10s} median {d[c].median()*100:5.2f}%   "
              f"IQR {d[c].quantile(.25)*100:4.2f}-{d[c].quantile(.75)*100:5.2f}%")
    clean = d[d.worst == 0]
    print(f"\n   sessions with no off-tract voxel in either region: {len(clean)}")
    print(f"   worse-region median, the figure the caption quotes: "
          f"{d.worst.median()*100:.1f}%")


if __name__ == "__main__":
    main()
