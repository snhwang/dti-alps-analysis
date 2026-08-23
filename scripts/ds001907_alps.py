"""ALPS variants in Parkinson's disease, and whether orientation correction
changes the clinical group comparison.

The pose result says patients lie with about six degrees more head rotation than
controls. That is the premise of the paper's claim, not the claim itself. The
claim is that a pose difference between clinical groups distorts the group
contrast an ALPS study would report, and testing it needs the index itself.

This builds the same derivative tree the DLBS, HCP-A and trigeminal cohorts
have, then calls the identical variant code, so the numbers are comparable
across every cohort in the project:

    FA -> FLIRT (already done by ds001907_pose) -> FNIRT with the standard
    FA_2_FMRIB58_1mm config -> invwarp -> ALPS spheres and JHU labels warped
    into native space -> estimator_variants

The FNIRT config lives in the FSL tree but the bridge shell has no FSLDIR, so
the absolute path is passed. Without it FNIRT silently falls back to affine,
which would place the ROIs worse for both arms and quietly weaken everything.

    python ds001907_alps.py --jobs 10
    python ds001907_alps.py --report
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DIFFREPO = HERE.parent.parent / "diffusion"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(DIFFREPO))
os.environ.setdefault("DTI_OUTPUT_DIR", str(DIFFREPO / "dti_output"))

from ds001907_common import group, assert_group_mapping, DEST  # noqa: E402
from ds001907_pose import fsl, p, scans  # noqa: E402
from tn_alps import alps_variants, VARIANTS  # noqa: E402

WORK = Path(r"M:\ds001907-derivatives")
ROI_SRC = Path(r"C:\tmp\alps_roi\old_ROIs_JHU_ALPS_5mm_radius")
JHU = DIFFREPO / "atlases" / "JHU-ICBM-labels-1mm.nii.gz"
CNF = "/home/snhwang/fsl/etc/flirtsch/FA_2_FMRIB58_1mm.cnf"
OUT = HERE / "ds001907_alps.csv"


def build(row) -> dict | None:
    """Warp the atlas into native space and assemble the tensor files."""
    import nibabel as nib
    import fastapi_diffusion_processor as fdp

    d = WORK / row.subject / row.session
    proc = d / "processed"
    atlas = proc / "atlas"
    atlas.mkdir(parents=True, exist_ok=True)
    fa = proc / "fa.nii.gz"
    try:
        if not fa.exists():
            shutil.copyfile(d / "dti_FA.nii.gz", fa)
        aff = atlas / "subject_to_mni_affine.mat"
        if not aff.exists():
            shutil.copyfile(d / "subject_to_mni_affine.mat", aff)

        coef = atlas / "subject_to_mni_warp_coef.nii.gz"
        if not coef.exists():
            fsl(f"fnirt --in={p(fa)} --ref={p(fdp.FMRIB58_FA_PATH)} "
                f"--aff={p(aff)} --cout={p(coef)} --config={CNF} "
                f"--inmask={p(d / 'nodif_brain_mask.nii.gz')}", timeout=3600)
        inv = atlas / "atlas_to_subject_warp.nii.gz"
        if not inv.exists():
            fsl(f"invwarp --warp={p(coef)} --ref={p(fa)} --out={p(inv)}",
                timeout=1800)

        lab = atlas / "jhu_labels_registered.nii.gz"
        if not lab.exists():
            fsl(f"applywarp --in={p(JHU)} --ref={p(fa)} --warp={p(inv)} "
                f"--out={p(lab)} --interp=nn")

        comb = atlas / "sphere_roi" / "sphere_roi_combined.nii.gz"
        if not comb.exists():
            comb.parent.mkdir(parents=True, exist_ok=True)
            got = {}
            for nm in ("L_SCR", "R_SCR", "L_SLF", "R_SLF"):
                dst = comb.parent / f"{nm}_native.nii.gz"
                if not dst.exists():
                    fsl(f"applywarp --in={p(ROI_SRC / (nm + '.nii.gz'))} "
                        f"--ref={p(fa)} --warp={p(inv)} --out={p(dst)} --interp=nn")
                got[nm] = nib.load(str(dst)).get_fdata() > 0.5
            ri = nib.load(str(fa))
            l = np.zeros(ri.shape[:3], np.uint8)
            l[got["L_SCR"] | got["R_SCR"]] = 1      # projection, SCR
            l[got["L_SLF"] | got["R_SLF"]] = 2      # association, SLF
            nib.save(nib.Nifti1Image(l, ri.affine), str(comb))

        # dtifit writes sorted eigenvalues and unit eigenvectors separately.
        # The variant code expects them stacked, with eigenvectors indexed
        # [..., component, eigenvector], matching the pipeline's own layout.
        evp, vcp = proc / "tensor_eigenvalues.nii.gz", proc / "tensor_eigenvectors.nii.gz"
        if not (evp.exists() and vcp.exists()):
            ref = nib.load(str(fa))
            L = [nib.load(str(d / f"dti_L{i}.nii.gz")).get_fdata() for i in (1, 2, 3)]
            V = [nib.load(str(d / f"dti_V{i}.nii.gz")).get_fdata() for i in (1, 2, 3)]
            nib.save(nib.Nifti1Image(np.stack(L, -1).astype(np.float32), ref.affine),
                     str(evp))
            nib.save(nib.Nifti1Image(np.stack(V, -1).astype(np.float32), ref.affine),
                     str(vcp))
        return {"subject": row.subject, "session": row.session, "group": row.group,
                "sdir": str(d)}
    except Exception as e:                                      # noqa: BLE001
        print(f"   BUILD FAIL {row.subject}/{row.session}: {repr(e)[:110]}", flush=True)
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    assert_group_mapping()

    if not args.report:
        s = scans()
        use = s[s.exclude == ""]
        if args.limit:
            use = use.head(args.limit)
        print(f"{len(use)} scans, {args.jobs} at a time\n")
        built = []
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            for i, r in enumerate(ex.map(build, list(use.itertuples())), 1):
                if r:
                    built.append(r)
                if i % 5 == 0:
                    print(f"   built {i}/{len(use)} ({len(built)} ok)", flush=True)

        rows = []
        for b in built:
            v = alps_variants(Path(b["sdir"]))
            if v:
                rows.append({**{k: b[k] for k in ("subject", "session", "group")},
                             **v})
        pd.DataFrame(rows).to_csv(OUT, index=False)
        print(f"\n   {len(rows)} scans with variants -> {OUT}")

    r = pd.read_csv(OUT)
    print(f"\n{len(r)} scans, {r.subject.nunique()} subjects")
    have = [v for v in VARIANTS if v in r.columns and r[v].notna().any()]
    print(f"{len(have)} variants computed\n")
    print(r.groupby("group")[have[:6]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
