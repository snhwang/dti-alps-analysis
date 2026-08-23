"""Is the stored trigeminal placement QC measuring placement at all?

The manuscript had reported that 43 of 171 sessions carried a region more than
8 mm from its intended position. That never made sense: the spheres are defined
at a fixed atlas coordinate and warped into native space, so they are centered
where they were placed by construction.

This recomputes the centroid of each delivered region in the participant's own
native RAS frame and compares it with the value stored in tn_sphere_qc.csv. If
the stored quantity is a native-space coordinate, it will reproduce exactly, and
the comparison against the JHU coordinates (26 and 38 mm) is then a comparison
between two different coordinate frames. What it would measure is head size and
how far lateral the tract sits in that participant, not placement error.

Result: it reproduces exactly, r = 1.000 with identical values, so the stored
figure is not a placement check. The claim has been removed from the manuscript.
"""
from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from data_paths import winpath

HERE = Path(__file__).resolve().parent
ROOT = winpath("M:/ds005713-derivatives/dti_output/ds005713_preproc")
LIMIT = 25


def main() -> None:
    qc = pd.read_csv(HERE / "tn_sphere_qc.csv").set_index("BIDS_ID")
    rows = []
    for sd in sorted(d for d in ROOT.iterdir() if d.is_dir()):
        meta = sd / "metadata.json"
        if not meta.exists():
            continue
        name = json.load(open(meta)).get("name", "")
        if not name or name.endswith("fu") or name not in qc.index:
            continue
        sp = sd / "processed" / "atlas" / "sphere_roi" / "sphere_roi_combined.nii.gz"
        if not sp.exists():
            continue
        im = nib.load(str(sp))
        lab = np.asarray(im.dataobj).astype(int)
        rec = {"BIDS_ID": name}
        for code, tag in ((1, "scr"), (2, "slf")):
            idx = np.argwhere(lab == code)
            if not len(idx):
                continue
            w = nib.affines.apply_affine(im.affine, idx)   # native RAS mm
            for hemi, sel in (("L", w[:, 0] < 0), ("R", w[:, 0] >= 0)):
                if sel.sum():
                    rec[f"{tag}_{hemi}_x_native"] = float(w[sel, 0].mean())
        rows.append(rec)
        if len(rows) >= LIMIT:
            break

    j = pd.DataFrame(rows).set_index("BIDS_ID").join(qc, how="inner")
    print(f"{len(j)} sessions\n")
    print(f"{'region':10s} {'stored _x':>11s} {'native centroid':>17s} {'|r|':>7s}")
    for c in ("scr_L", "scr_R", "slf_L", "slf_R"):
        a, b = j[f"{c}_x"], j[f"{c}_x_native"]
        print(f"{c:10s} {a.median():11.2f} {b.median():17.2f} "
              f"{abs(np.corrcoef(a, b)[0, 1]):7.3f}")
    print("\n|r| = 1.000 means the stored value IS the native centroid, so comparing")
    print("it against the JHU coordinate compares two different frames.")


if __name__ == "__main__":
    main()
