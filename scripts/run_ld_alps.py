"""Run the reference LD-ALPS implementation on our cohorts.

Until now the paper compared against a per-voxel variant written in the spirit
of LD-ALPS rather than against LD-ALPS itself. That was an honest label but a
weak benchmark, and it is no longer necessary: Burles et al. release their code
under MIT at https://fordburles.com/ld-alps.html, so the actual method can be
run on the same sessions as every other variant.

The two are not the same algorithm, which is why this matters. Our per-voxel
variant crosses each voxel's own principal direction with the opposite region's
ROI-mean direction and reads diffusivity off the fitted tensor. LD-ALPS first
rejects outlier voxels by DBSCAN on the sphere, takes the surviving voxel
nearest the cluster centre as that region's direction, and then computes the
apparent diffusion coefficient by interpolating the measured signal across the
acquired gradient directions with a Clough-Tocher interpolator, never using the
tensor at all. The second difference is the substantive one: LD-ALPS measures
ADC from the data, our variant derives it from a model fitted to the data.

This script only adapts inputs. It does not reimplement anything. Their code is
vendored unmodified in external/ld-alps.py and invoked as published, so the
result is theirs rather than our reading of their paper.

Their loader expects, per subject directory:

    eddy_corrected_data.nii.gz                  4D DWI after eddy
    eddy_corrected_data.eddy_rotated_bvecs      (3, K) rotated gradients
    bvals                                       (K,)
    dti_V1.nii.gz                               principal eigenvector volume
    nativeALPSrois.nii.gz                       labels 1..4, in their order
                                                R_Assoc, R_Proj, L_Assoc, L_Proj

Everything is present in our trees except V1 as a standalone volume and the
four-label ROI file, both of which are derived here. Note the label order is
theirs and differs from ours, where 1 is projection and 2 association pooled
across hemispheres.

    python run_ld_alps.py --cohort dlbs --limit 20
    python run_ld_alps.py --cohort dlbs

Writes ld_alps_<cohort>.csv.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import atomic_io  # noqa: F401  writes become atomic on import
from data_paths import winpath

HERE = Path(__file__).resolve().parent
DIFF = HERE.parent.parent / "diffusion"
OUT = winpath("Q:/dti_output")
VENDORED = HERE / "external" / "ld-alps.py"
WORK = winpath("Q:/ld_alps_work")   # same volume as the sessions, so hardlinks work

# Their label order, mapped onto the ROI files we already warp into native space.
LABEL_ORDER = (("R_SLF", 1), ("R_SCR", 2), ("L_SLF", 3), ("L_SCR", 4))


def prepare(sdir: Path, dest: Path) -> bool:
    """Lay out one session the way their loader expects. Returns False if incomplete."""
    import nibabel as nib

    proc = sdir / "processed"
    atlas = proc / "atlas"
    eddy = proc / "dwi_eddy_corrected.nii.gz"
    bvec = proc / "dwi_eddy_rotated.bvec"
    bval = sdir / "inputs" / "dwi.bval"
    evecs = proc / "tensor_eigenvectors.nii.gz"
    sph_dir = atlas / "sphere_roi"
    if not all(p.exists() for p in (eddy, bvec, bval, evecs)):
        return False
    if not all((sph_dir / f"{n}_native.nii.gz").exists() for n, _ in LABEL_ORDER):
        return False

    dest.mkdir(parents=True, exist_ok=True)
    # Their loader opens these by name, so link or copy rather than pass paths.
    for src, name in ((eddy, "eddy_corrected_data.nii.gz"),
                      (bvec, "eddy_corrected_data.eddy_rotated_bvecs"),
                      (bval, "bvals")):
        tgt = dest / name
        if tgt.exists():
            continue
        try:
            os.link(src, tgt)
        except OSError:
            shutil.copyfile(src, tgt)

    v1_path = dest / "dti_V1.nii.gz"
    if not v1_path.exists():
        img = nib.load(str(evecs))
        v1 = np.asanyarray(img.dataobj)[..., 0]      # principal eigenvector
        nib.save(nib.Nifti1Image(np.ascontiguousarray(v1, np.float32), img.affine),
                 str(v1_path))

    roi_path = dest / "nativeALPSrois.nii.gz"
    if not roi_path.exists():
        ref = nib.load(str(sph_dir / f"{LABEL_ORDER[0][0]}_native.nii.gz"))
        lab = np.zeros(ref.shape[:3], np.uint8)
        for name, value in LABEL_ORDER:
            m = np.asanyarray(nib.load(str(sph_dir / f"{name}_native.nii.gz")).dataobj) > 0.5
            lab[m] = value
        nib.save(nib.Nifti1Image(lab, ref.affine), str(roi_path))
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=["dlbs", "tn"], default="dlbs")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--keep", action="store_true", help="leave the staged inputs on disk")
    args = ap.parse_args()

    if not VENDORED.exists():
        raise SystemExit(f"vendored LD-ALPS not found at {VENDORED}")

    src = pd.read_csv(DIFF / "DLBS" / "dlbs_alps_spheres_5mm.csv")
    src = src[src.status == "ok"].copy()
    src["Visit"] = src["Session"]
    if args.limit:
        src = src.head(args.limit)

    base = WORK / args.cohort
    base.mkdir(parents=True, exist_ok=True)
    staged = []
    for r in src.itertuples():
        dest = base / f"alps_{r.Subject_ID}_{r.Visit}"
        if prepare(OUT / r.DTI_Session_ID, dest):
            staged.append((r.Subject_ID, r.Visit, dest.name))
    print(f"staged {len(staged)} of {len(src)} sessions in {base}", flush=True)
    if not staged:
        raise SystemExit("nothing staged; check that eddy outputs and spheres exist")

    out_csv = base / "ld_alps_raw.csv"
    cmd = [sys.executable, str(VENDORED), str(base), "--subject-prefix", "alps_",
           "--out", str(out_csv)]
    print(" ".join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        print(r.stderr[-3000:])
        raise SystemExit(f"LD-ALPS exited {r.returncode}")
    print(r.stdout[-1500:])

    d = pd.read_csv(out_csv)
    key = pd.DataFrame(staged, columns=["Subject_ID", "Visit", "subject"])
    idc = next((c for c in d.columns if c.lower() in ("subject", "subject_id", "id")), None)
    if idc is not None:
        d = d.rename(columns={idc: "subject"}).merge(key, on="subject", how="left")
    d.to_csv(HERE / f"ld_alps_{args.cohort}.csv", index=False)
    print(f"\n{len(d)} sessions -> ld_alps_{args.cohort}.csv")
    print("columns:", list(d.columns))
    if not args.keep:
        shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    main()
