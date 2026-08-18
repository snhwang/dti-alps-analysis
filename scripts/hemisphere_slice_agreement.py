"""In how many sessions do the two hand-drawn hemispheres share a slice?

The Figure 3 caption states that raters drew each hemisphere on whatever slice
suited it, and that the two sides coincide in most sessions. That count decides
whether a single axial slice can show all four hand-drawn regions at once, which
is what the figure needs.

The criterion is the one build_roi_comparison_figure.py applies when choosing a
session: a slice counts if it carries all four regions, two tracts by two
hemispheres, with at least three voxels each, and a session counts if any slice
does. Hemisphere is decided by the world x coordinate, so the test does not
depend on the atlas.

That count was previously printed to the console by the figure builder and never
recorded, so the caption could not be checked against anything. This writes it
down, together with the intermediate filters, since the figure builder applies
several and the caption refers to only one of them.

    python hemisphere_slice_agreement.py

Writes hemisphere_slice_agreement.csv.
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
EXCLUDE = {"session_20260122_160723"}      # damaged manual mask
MIN_VOX = 3


def four_region_slices(man, affine):
    """Slices carrying all four hand-drawn regions: two tracts, two hemispheres."""
    ii, jj, kk = np.indices(man.shape)
    xw = (affine[0, 0] * ii + affine[0, 1] * jj
          + affine[0, 2] * kk + affine[0, 3])
    out = []
    for k in range(man.shape[2]):
        s = man[:, :, k]
        if not (s > 0).any():
            continue
        got = sum(1 for v in (1, 2) for side in (xw[:, :, k] < 0, xw[:, :, k] > 0)
                  if ((s == v) & side).sum() >= MIN_VOX)
        if got == 4:
            out.append(k)
    return out


def is_hand_drawn(man) -> bool:
    """A programmatic placement is a union of spheres of identical size."""
    sizes = [int((man == v).sum()) for v in (1, 2)]
    return not (sizes[0] == sizes[1] and sizes[0] in (80, 136, 200))


def main() -> None:
    argparse.ArgumentParser().parse_args()
    import nibabel as nib

    src = pd.read_csv(DIFF / "HCP" / "lifespan_alps_results.csv")
    src = src.dropna(subset=["DTI_Session_ID"])
    rows = []
    for r in src.itertuples():
        sid = str(r.DTI_Session_ID)
        sd = OUT / sid / "processed"
        mp = sd / "alps_rois_manual.nii.gz"
        if not mp.exists():
            continue
        try:
            im = nib.load(str(mp))
            man = np.rint(im.get_fdata()).astype(int)
        except Exception:
            continue
        good = four_region_slices(man, im.affine)
        rows.append(dict(
            sid=sid,
            excluded=sid in EXCLUDE,
            hand_drawn=bool(is_hand_drawn(man)),
            has_atlas=(sd / "atlas" / "sphere_roi"
                       / "sphere_roi_combined.nii.gz").exists(),
            n_shared_slices=len(good),
            sides_coincide=len(good) > 0))

    d = pd.DataFrame(rows)
    d.to_csv(HERE / "hemisphere_slice_agreement.csv", index=False)

    n = len(d)
    print(f"sessions with a hand-drawn mask                      {n}")
    print(f"   the two sides share at least one slice            "
          f"{int(d.sides_coincide.sum())}")
    print(f"   and the mask is genuinely hand drawn              "
          f"{int((d.sides_coincide & d.hand_drawn).sum())}")
    print(f"   and an atlas placement exists                     "
          f"{int((d.sides_coincide & d.hand_drawn & d.has_atlas).sum())}")
    print(f"   and the session is not the damaged one            "
          f"{int((d.sides_coincide & d.hand_drawn & d.has_atlas & ~d.excluded).sum())}")
    print(f"\n   median shared slices where they do coincide       "
          f"{d.loc[d.sides_coincide, 'n_shared_slices'].median():.0f}")
    print(f"   sessions carrying an atlas placement               "
          f"{int(d.has_atlas.sum())}")


if __name__ == "__main__":
    main()
