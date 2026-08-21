"""
Is the ALPS location special for perpendicular anisotropy, or is the ROI doing all the work?

If the index degenerates to lambda2/lambda3, it is a measure of how anisotropic the
plane perpendicular to the fibre is, and that quantity exists in every white
matter voxel. What makes it ALPS is the claim that at this particular location,
lateral to the ventricles where medullary vessels run across the tracts, the
perpendicular anisotropy is perivascular rather than something else.

That claim is testable in a limited way. If perpendicular anisotropy at the ALPS
regions is unremarkable compared with other white matter, then the anatomical
specificity rests entirely on where the region was placed and not on anything
measurable in the signal. If it stands out, the location is doing something.

Compares lambda2/lambda3 and the Westin planar coefficient in the two ALPS
regions against the other JHU tracts, in the same sessions, and also asks whether
the age association at the ALPS location is distinctive or shared.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from data_paths import winpath

import atomic_io  # noqa: F401  writes become atomic on import
from scipy import stats

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
DIFF = HERE.parent.parent / "diffusion"
OUT = winpath("Q:/dti_output")
SHELL = os.environ.get("ALPS_TENSOR_SUFFIX", "_b1500")
FA_MIN = 0.2
SLAB_MM = 8.0

# JHU labels. SCR and SLF are where ALPS is measured; the rest are comparators.
ROIS = {"SCR (ALPS proj)": (25, 26), "SLF (ALPS assoc)": (41, 42),
        "Genu CC": (3,), "Body CC": (4,), "Splenium CC": (5,),
        "ACR": (23, 24), "PCR": (27, 28), "PLIC": (19, 20),
        "ALIC": (17, 18), "External capsule": (33, 34),
        "Cingulum": (35, 36), "Sagittal stratum": (31, 32)}


def main() -> None:
    import nibabel as nib

    src = pd.read_csv(DIFF / "HCP" / "hcpa_alps_spheres_5mm.csv")
    src = src[src.status == "ok"].head(120)

    rows = []
    for i, r in enumerate(src.itertuples(), 1):
        sd = OUT / r.DTI_Session_ID / "processed"
        lab_p = sd / "atlas" / "jhu_labels_registered.nii.gz"
        ev_p = sd / f"tensor_eigenvalues{SHELL}.nii.gz"
        sph_p = sd / "atlas" / "sphere_roi" / "sphere_roi_combined.nii.gz"
        if not (lab_p.exists() and ev_p.exists() and sph_p.exists()):
            continue
        limg = nib.load(str(lab_p)); lab = limg.get_fdata().astype(int)
        ev = nib.load(str(ev_p)).get_fdata()
        sph = nib.load(str(sph_p)).get_fdata().astype(int)

        srt = np.sort(ev, axis=-1)[..., ::-1]
        l1, l2, l3 = srt[..., 0], srt[..., 1], srt[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            # kept per voxel only for the planarity map; the regional ratio
            # below is formed as a ratio of means, as the index is
            CP = np.where(l1 > 0, (l2 - l3) / l1, np.nan)
        md = ev.mean(-1)
        nu = np.sqrt(((ev - md[..., None]) ** 2).sum(-1))
        de = np.sqrt((ev ** 2).sum(-1))
        fa = np.clip(np.sqrt(1.5) * np.divide(nu, de, out=np.zeros_like(nu), where=de != 0), 0, 1)

        ii, jj, kk = np.indices(lab.shape)
        A = limg.affine
        zw = A[2, 0] * ii + A[2, 1] * jj + A[2, 2] * kk + A[2, 3]
        z0 = float(np.median(zw[sph > 0])) if (sph > 0).any() else 0.0
        band = np.abs(zw - z0) <= SLAB_MM

        rec = {"sid": r.DTI_Session_ID, "Age": r.Age}
        for nm, ids in ROIS.items():
            m = np.isin(lab, ids) & (fa >= FA_MIN)
            # the two ALPS regions are measured at the ALPS level, as the index does
            if "ALPS" in nm:
                m = m & band
            if m.sum() < 20:
                continue
            rec[nm] = float(np.nanmean(l2[m]) / np.nanmean(l3[m]))
            rec[nm + " CP"] = float(np.nanmean(CP[m]))
        rows.append(rec)
        if i % 30 == 0:
            print(f"  {i}/{len(src)}", flush=True)

    d = pd.DataFrame(rows)
    d["Age"] = pd.to_numeric(d.Age, errors="coerce")
    d.to_csv(HERE / "alps_location_special.csv", index=False)
    print(f"\n{len(d)} sessions\n")

    print(f"{'region':<20s} {'lambda2/lambda3':>16s} {'CP':>7s} {'r with age':>11s}")
    out = []
    for nm in ROIS:
        if nm not in d:
            continue
        s = d[[nm, "Age", nm + " CP"]].dropna()
        r = stats.pearsonr(s.Age, s[nm])[0] if len(s) > 20 else np.nan
        out.append((nm, s[nm].mean(), s[nm + " CP"].mean(), r))
    for nm, v, cp, r in sorted(out, key=lambda x: -x[1]):
        mark = "  <-- ALPS" if "ALPS" in nm else ""
        print(f"{nm:<20s} {v:16.3f} {cp:7.3f} {r:11.3f}{mark}")

    alps = [v for nm, v, _, _ in out if "ALPS" in nm]
    other = [v for nm, v, _, _ in out if "ALPS" not in nm]
    if alps and other:
        print(f"\nALPS regions mean {np.mean(alps):.3f}, other white matter "
              f"{np.mean(other):.3f}, rank of ALPS regions among "
              f"{len(out)}: "
              f"{[i for i, (nm, *_ ) in enumerate(sorted(out, key=lambda x: -x[1]), 1) if 'ALPS' in nm]}")


if __name__ == "__main__":
    main()
