"""Where does a rater put the region, relative to where the atlas puts it?

manual_placement_age.py asked whether hand-drawn regions drift off the intended
tract with age, and found a marginal effect. That measures distance from a
tract. This measures the thing the rater actually controls: the displacement
between the centre of the hand-drawn region and the centre of the warped atlas
sphere in the same session.

Centroids are taken per hemisphere. A bilateral label has its centroid near the
midline whatever the rater did, so a left-right displacement would cancel
against itself and the test would be blind to exactly the drift it is looking
for.

Displacement is reported in millimetres through the affine, and signed along
each axis as well as in magnitude, because a drift with a consistent direction
means something different from one that merely grows. Superior drift in the
projection region, for instance, would suggest the rater following the corona
radiata upward as the colour fades.

    python manual_centroid_shift.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_paths import winpath                                  # noqa: E402

HERE = Path(__file__).resolve().parent
DIFF = HERE.parent.parent / "diffusion"
OUT = Path(winpath("Q:/dti_output"))


def centroids(mask, aff):
    """Centre of mass in mm, split into left and right by world x."""
    ii, jj, kk = np.nonzero(mask)
    if not len(ii):
        return {}
    xyz = nib.affines.apply_affine(aff, np.column_stack([ii, jj, kk]))
    out = {}
    for side, sel in (("L", xyz[:, 0] < 0), ("R", xyz[:, 0] > 0)):
        if sel.sum() >= 4:
            out[side] = xyz[sel].mean(axis=0)
    return out


def main() -> None:
    argparse.ArgumentParser().parse_args()
    o = pd.read_csv(HERE / "manual_roi_offtract.csv")
    sp = pd.read_csv(DIFF / "DLBS" / "dlbs_alps_spheres_5mm.csv")
    sp["DTI_Session_ID"] = sp.DTI_Session_ID.astype(str)
    age = dict(zip(sp.DTI_Session_ID, pd.to_numeric(sp.Age, errors="coerce")))

    rows = []
    for sid in o.sid.astype(str):
        proc = OUT / sid / "processed"
        hand_p = proc / "alps_rois_manual.nii.gz"
        atlas_p = proc / "atlas" / "sphere_roi" / "sphere_roi_combined.nii.gz"
        if not (hand_p.exists() and atlas_p.exists()):
            continue
        h = nib.load(str(hand_p))
        a = nib.load(str(atlas_p))
        hd = np.asanyarray(h.dataobj)
        ad = np.asanyarray(a.dataobj)
        row = {"sid": sid, "Age": age.get(sid, np.nan)}
        for lab, name in ((1, "proj"), (2, "assoc")):
            ch = centroids(hd == lab, h.affine)
            ca = centroids(ad == lab, a.affine)
            for side in ("L", "R"):
                if side in ch and side in ca:
                    d = ch[side] - ca[side]
                    row[f"{name}_{side}_dx"] = d[0]
                    row[f"{name}_{side}_dy"] = d[1]
                    row[f"{name}_{side}_dz"] = d[2]
                    row[f"{name}_{side}_dist"] = float(np.linalg.norm(d))
        rows.append(row)

    d = pd.DataFrame(rows)
    d.to_csv(HERE / "manual_centroid_shift.csv", index=False)
    print(f"{len(d)} sessions with both a hand-drawn and an atlas region\n")

    dist = [c for c in d.columns if c.endswith("_dist")]
    d["mean_dist"] = d[dist].mean(axis=1)
    print("=== how far the rater sits from the atlas centre, mm ===")
    for c in dist + ["mean_dist"]:
        v = d[c].dropna()
        print(f"   {c:<18s} n={len(v):3d}  median {v.median():5.1f}  "
              f"IQR {v.quantile(.25):4.1f}-{v.quantile(.75):4.1f}  max {v.max():5.1f}")

    print("\n=== displacement against age ===")
    for c in dist + ["mean_dist"]:
        s = d[[c, "Age"]].dropna()
        if len(s) < 10:
            continue
        r, p = stats.pearsonr(s[c], s.Age)
        flag = "  *" if p < 0.05 else ""
        print(f"   {c:<18s} n={len(s):3d}  r={r:+.3f}  p={p:.4f}{flag}")

    print("\n=== signed components against age, is the drift directional? ===")
    for c in sorted(x for x in d.columns if x[-3:] in ("_dx", "_dy", "_dz")):
        s = d[[c, "Age"]].dropna()
        if len(s) < 10:
            continue
        r, p = stats.pearsonr(s[c], s.Age)
        if p < 0.10:
            print(f"   {c:<18s} n={len(s):3d}  r={r:+.3f}  p={p:.4f}"
                  f"{'  *' if p < 0.05 else ''}")
    print(f"\n   wrote manual_centroid_shift.csv")


if __name__ == "__main__":
    main()
