"""Head pose for every ds001907 diffusion scan, to test whether Parkinson's
patients lie differently in the scanner.

Pose is measured exactly as it is for DLBS, HCP-A and the trigeminal cohort:
FLIRT the subject's FA to FMRIB58_FA_1mm at 12 degrees of freedom with the full
search range, then take the polar decomposition of the linear part and read
pitch, roll and yaw off the nearest rotation. Using a different registration
here would make the numbers incomparable with the rest of the project, so the
flirt line below is copied from fastapi_diffusion_processor.

    flirt -in FA -ref FMRIB58_FA_1mm -omat M -dof 12 -cost corratio
          -searchrx -90 90 -searchry -90 90 -searchrz -90 90

One deliberate deviation. The full pipeline runs eddy before dtifit; this does
not. Eddy on 85 scans of 129 volumes is days of compute and it corrects
within-series distortion and motion, which is not what a gross rigid pose
measurement depends on. Both arms are processed identically, so the deviation
cannot manufacture a group difference. It does leave one real worry, that
Parkinson's tremor makes patient FA blurrier and shifts its registration, which
is why mcflirt is run as well and within-scan motion is carried as a covariate
rather than assumed away.

Excluded, decided by acquisition rather than by result:
  - the two b=1000 65-volume scans, a different sequence from the other 83
  - the two control scans at TE 0.067 and bandwidth 2402, parameters no
    patient scan has

    python ds001907_pose.py --jobs 6
    python ds001907_pose.py --report      # re-read the CSV, no recompute
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DIFFREPO = HERE.parent.parent / "diffusion"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(DIFFREPO))
os.environ.setdefault("DTI_OUTPUT_DIR", str(DIFFREPO / "dti_output"))

from ds001907_common import group, assert_group_mapping, DEST  # noqa: E402

WORK = Path(r"M:\ds001907-derivatives")
OUT = HERE / "ds001907_pose.csv"


def decompose(mat: np.ndarray):
    """Polar decomposition of the linear part, then Euler angles of the rotation.

    Identical to head_rotation_observed.decompose. Shear and scale go into S and
    are discarded, so what is left is head pose relative to the template.
    """
    A = mat[:3, :3]
    U, _, Vt = np.linalg.svd(A)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    total = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
    roll = float(np.degrees(np.arcsin(np.clip(-R[2, 0], -1, 1))))
    if abs(R[2, 0]) < 0.9999:
        pitch = float(np.degrees(np.arctan2(R[2, 1], R[2, 2])))
        yaw = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    else:
        pitch = float(np.degrees(np.arctan2(-R[1, 2], R[1, 1])))
        yaw = 0.0
    return pitch, roll, yaw, total


def scans() -> pd.DataFrame:
    """Every dwi image with its group, shell and echo time, exclusions marked."""
    rows = []
    for f in sorted(DEST.rglob("*_dwi.nii.gz")):
        stem = str(f)[: -len(".nii.gz")]
        bval = Path(stem + ".bval")
        side = Path(stem + ".json")
        b = np.loadtxt(bval)
        shells = sorted({int(round(v / 50) * 50) for v in b})
        j = json.loads(side.read_text()) if side.exists() else {}
        sid = f.parts[len(DEST.parts)]
        ses = next((p for p in f.parts if p.startswith("ses-")), "ses-1")
        rows.append({"subject": sid, "session": ses, "group": group(sid),
                     "nii": str(f), "n_vol": len(b), "max_b": max(shells),
                     "echo_time": j.get("EchoTime"),
                     "bandwidth": j.get("PixelBandwidth")})
    d = pd.DataFrame(rows)
    d["exclude"] = ""
    d.loc[d.max_b != 3000, "exclude"] = "b1000 sequence"
    odd = d.echo_time.round(4) != round(0.09347, 4)
    d.loc[odd & (d.exclude == ""), "exclude"] = "off-protocol TE/bandwidth"
    return d


def fsl(cmd: str, timeout: int = 1800):
    import fastapi_diffusion_processor as fdp
    return fdp._run_fsl(cmd, timeout=timeout)


def p(path) -> str:
    import fastapi_diffusion_processor as fdp
    return fdp._to_fsl_path(Path(path))


def process(row) -> dict | None:
    """b0 -> mask -> dtifit -> flirt, skipping any step already done."""
    import fastapi_diffusion_processor as fdp
    d = WORK / row.subject / row.session
    d.mkdir(parents=True, exist_ok=True)
    stem = row.nii[: -len(".nii.gz")]
    mat = d / "subject_to_mni_affine.mat"
    fa = d / "dti_FA.nii.gz"
    try:
        if not fa.exists():
            nodif, mask = d / "nodif.nii.gz", d / "nodif_brain.nii.gz"
            if not nodif.exists():
                b = np.loadtxt(stem + ".bval")
                i0 = int(np.argmin(b))
                fsl(f"fslroi {p(row.nii)} {p(nodif)} {i0} 1")
            if not (d / "nodif_brain_mask.nii.gz").exists():
                fsl(f"bet {p(nodif)} {p(mask)} -m -f 0.3")
            fsl(f"dtifit -k {p(row.nii)} -o {p(d / 'dti')} "
                f"-m {p(d / 'nodif_brain_mask.nii.gz')} "
                f"-r {p(stem + '.bvec')} -b {p(stem + '.bval')}")
        if not mat.exists():
            fsl(f"flirt -in {p(fa)} -ref {p(fdp.FMRIB58_FA_PATH)} "
                f"-omat {p(mat)} -dof 12 -cost corratio "
                f"-searchrx -90 90 -searchry -90 90 -searchrz -90 90")
        # within-scan motion, so tremor can be a covariate rather than a worry
        rms = np.nan
        par = d / "mc.par"
        if not par.exists():
            fsl(f"mcflirt -in {p(row.nii)} -out {p(d / 'mc')} -plots -refvol 0")
        if par.exists():
            m = np.loadtxt(par)
            rot, tr = np.degrees(m[:, :3]), m[:, 3:]
            rms = float(np.sqrt((np.diff(tr, axis=0) ** 2).sum(1)).mean())
            rot_rms = float(np.sqrt((np.diff(rot, axis=0) ** 2).sum(1)).mean())
        else:
            rot_rms = np.nan
        A = np.loadtxt(mat)
        pitch, roll, yaw, total = decompose(A)
        return {"subject": row.subject, "session": row.session, "group": row.group,
                "pitch": pitch, "roll": roll, "yaw": yaw, "total": total,
                "motion_tr_rms": rms, "motion_rot_rms": rot_rms}
    except Exception as e:                                      # noqa: BLE001
        print(f"   FAIL {row.subject}/{row.session}: {repr(e)[:120]}", flush=True)
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    assert_group_mapping()

    if not args.report:
        d = scans()
        print(f"{len(d)} scans; excluded: "
              f"{d[d.exclude!=''].exclude.value_counts().to_dict()}")
        use = d[d.exclude == ""]
        if args.limit:
            use = use.head(args.limit)
        print(f"{len(use)} to process "
              f"({(use.group=='patient').sum()} patient, "
              f"{(use.group=='control').sum()} control), {args.jobs} at a time\n")
        WORK.mkdir(parents=True, exist_ok=True)
        done = []
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            for i, r in enumerate(ex.map(process, [x for x in use.itertuples()]), 1):
                if r:
                    done.append(r)
                if i % 5 == 0:
                    print(f"   {i}/{len(use)}  ({len(done)} ok)", flush=True)
        pd.DataFrame(done).to_csv(OUT, index=False)
        print(f"\n   wrote {OUT}")

    r = pd.read_csv(OUT)
    print(f"\n{len(r)} scans, {r.subject.nunique()} subjects\n")
    print(f"{'axis':<16s} {'patient':>18s} {'control':>18s}")
    for c in ("pitch", "roll", "yaw", "total", "motion_tr_rms"):
        a, b = (r.loc[r.group == g, c].dropna() for g in ("patient", "control"))
        v = "abs" if c in ("pitch", "roll", "yaw") else "raw"
        A, B = (a.abs(), b.abs()) if v == "abs" else (a, b)
        print(f"{c:<16s} {A.median():8.2f} +- {A.std():5.2f}  "
              f"{B.median():8.2f} +- {B.std():5.2f}")


if __name__ == "__main__":
    main()
