"""
Were the hand-drawn regions placed where the protocol specifies?

The atlas-placed regions are at defined template coordinates by construction:
(+/-26, -16, 27) for the projection region and (+/-38, -16, 27) for the
association region, in JHU-ICBM-FA-1mm. The hand-drawn regions were placed by
eye on each subject's colour map, so where they ended up is an empirical
question rather than a given, and the manual-versus-automated comparison is hard
to interpret without knowing the answer.

This warps each hand-drawn region into template space with the same subject-to-
template transform used elsewhere, takes its centroid, and compares that with
the specified coordinate. Hemispheres are separated by the sign of world x,
which is the convention verified against the atlas labels in decoupled_roi.py.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
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
TARGET = {("proj", "L"): (-26, -16, 27), ("proj", "R"): (26, -16, 27),
          ("assoc", "L"): (-38, -16, 27), ("assoc", "R"): (38, -16, 27)}
LABEL = {"proj": 1, "assoc": 2}      # value in alps_rois_manual.nii.gz


def to_fsl(p) -> str:
    p = str(p).replace("\\", "/")
    return f"/mnt/{p[0].lower()}{p[2:]}" if len(p) > 1 and p[1] == ":" else p


def warp_to_template(src: Path, ref: Path, warp: Path, dst: Path) -> bool:
    if dst.exists():
        return True
    cmd = (f"applywarp --in={to_fsl(src)} --ref={to_fsl(ref)} "
           f"--warp={to_fsl(warp)} --out={to_fsl(dst)} --interp=nn")
    r = subprocess.run(f'wsl -e bash -lc "{cmd}"', shell=True,
                       capture_output=True, text=True, timeout=600)
    return r.returncode == 0 and dst.exists()


def main() -> None:
    import nibabel as nib
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="HCP/alps_results_2026-02-22.csv")
    ap.add_argument("--id-col", default="Session_ID")
    args = ap.parse_args()

    src = pd.read_csv(DIFF / args.source).dropna(subset=[args.id_col])
    fsl = None
    ref_template = None

    rows = []
    for r in src.itertuples():
        sid = str(getattr(r, args.id_col))
        sd = OUT / sid / "processed"
        man = sd / "alps_rois_manual.nii.gz"
        warp = sd / "atlas" / "subject_to_mni_warp.nii.gz"
        jhu = sd / "atlas" / "jhu_fa_registered.nii.gz"
        if not (man.exists() and warp.exists()):
            continue
        # the template grid: use the shipped atlas-space ROI as reference
        if ref_template is None:
            ref_template = refined_rois() / "L_SCR.nii.gz"
        dst = sd / "atlas" / "alps_rois_manual_template.nii.gz"
        if not warp_to_template(man, ref_template, warp, dst):
            continue
        try:
            img = nib.load(str(dst)); lab = np.rint(img.get_fdata()).astype(int)
        except Exception:
            continue
        if (lab > 0).sum() == 0:
            continue
        ii, jj, kk = np.indices(lab.shape); A = img.affine
        w = np.stack([A[a, 0] * ii + A[a, 1] * jj + A[a, 2] * kk + A[a, 3]
                      for a in range(3)])
        rec = {"sid": sid}
        for region, val in LABEL.items():
            for hemi, side in (("L", w[0] < 0), ("R", w[0] > 0)):
                m = (lab == val) & side
                if m.sum() < 3:
                    continue
                c = np.array([w[a][m].mean() for a in range(3)])
                t = np.array(TARGET[(region, hemi)], float)
                rec[f"{region}_{hemi}_dx"] = c[0] - t[0]
                rec[f"{region}_{hemi}_dy"] = c[1] - t[1]
                rec[f"{region}_{hemi}_dz"] = c[2] - t[2]
                rec[f"{region}_{hemi}_dist"] = float(np.linalg.norm(c - t))
        rows.append(rec)
        if len(rows) % 10 == 0:
            print(f"  {len(rows)} sessions", flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(HERE / "manual_roi_placement_check.csv", index=False)
    if d.empty:
        print("no sessions could be warped")
        return

    print(f"\n{len(d)} hand-drawn sessions warped to template space\n")
    print(f"  {'region':<12s} {'dx':>8s} {'dy':>8s} {'dz':>8s} {'distance':>10s}  n")
    for region in ("proj", "assoc"):
        for hemi in ("L", "R"):
            cols = [f"{region}_{hemi}_{a}" for a in ("dx", "dy", "dz", "dist")]
            if cols[0] not in d:
                continue
            s = d[cols].dropna()
            print(f"  {region + ' ' + hemi:<12s} {s.iloc[:,0].median():+8.1f} "
                  f"{s.iloc[:,1].median():+8.1f} {s.iloc[:,2].median():+8.1f} "
                  f"{s.iloc[:,3].median():10.1f}  {len(s)}")
    dist = [c for c in d.columns if c.endswith("_dist")]
    alld = d[dist].to_numpy(float).ravel()
    alld = alld[~np.isnan(alld)]
    print(f"\n  median displacement from the specified coordinate: {np.median(alld):.1f} mm")
    print(f"  regions more than 5 mm away: {100*(alld > 5).mean():.0f}%")
    print(f"  regions more than 10 mm away: {100*(alld > 10).mean():.0f}%")
    print(f"\nwrote {HERE / 'manual_roi_placement_check.csv'}")


if __name__ == "__main__":
    main()
