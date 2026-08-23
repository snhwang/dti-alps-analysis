"""Does the Parkinson's pose difference survive a completely different image?

The diffusion result is that patients sit with about six degrees more head
rotation than controls. The threat to it is that pose is measured by registering
the subject's FA to an FA template, and Parkinson's brains differ from control
brains. If white matter degeneration pulls the registration, the group
difference would be an artefact of the measurement rather than a fact about how
people lie in the scanner.

That threat is specific to the FA registration, so the control is to measure the
same quantity from a different image with a different contrast and a different
template. T1 is acquired in the same session as the diffusion, on the same head,
and shares nothing with the FA pipeline except the subject.

  - replicates in T1  -> the measurement is reading the head, not the FA maps
  - absent in T1      -> the diffusion result is a registration artefact

The T1 and diffusion scans are separate acquisitions minutes apart, so the two
pose estimates are not required to be identical. Their agreement is reported as
a second, independent quantity: it says how much of head pose is a stable
property of the person rather than of the individual acquisition, which is the
question the r=0.91 between-session correlation raised.

    python ds001907_t1_pose.py --jobs 6
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
DIFFREPO = HERE.parent.parent / "diffusion"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(DIFFREPO))
os.environ.setdefault("DTI_OUTPUT_DIR", str(DIFFREPO / "dti_output"))

from ds001907_common import group, assert_group_mapping, DEST, demographics  # noqa: E402
from ds001907_pose import decompose, fsl, p  # noqa: E402

WORK = Path(r"M:\ds001907-derivatives")
OUT = HERE / "ds001907_t1_pose.csv"


def _ref() -> str:
    """Locate MNI152_T1_2mm_brain, the T1 counterpart of FMRIB58_FA_1mm.

    FSLDIR is not exported in the non-interactive shell the bridge uses, and a
    flirt given an unresolvable -ref exits zero without writing a matrix, so an
    unchecked path fails silently and looks like a null result. Resolve it once
    and fail loudly if it is missing.
    """
    import fastapi_diffusion_processor as fdp
    for d in ("$FSLDIR/data/standard", "/home/snhwang/fsl/data/standard",
              "/usr/local/fsl/data/standard", "/usr/share/fsl/data/standard"):
        # No shell variable here. Assigning one and testing it came back empty
        # through the WSL bridge even for a file that plainly exists, so the
        # literal path is tested directly.
        got = fdp._run_fsl(
            f"ls {d}/MNI152_T1_2mm_brain.nii.gz 2>/dev/null || true").stdout.strip()
        if got:
            return got.replace(".nii.gz", "")
    raise SystemExit("MNI152_T1_2mm_brain not found; cannot run the T1 control")


REF = None      # resolved in main, so --report needs no FSL


def process(row) -> dict | None:
    d = WORK / row.subject / row.session
    d.mkdir(parents=True, exist_ok=True)
    mat = d / "t1_to_mni_affine.mat"
    brain = d / "t1_brain.nii.gz"
    std = d / "t1_std.nii.gz"
    try:
        if not mat.exists():
            if not std.exists():
                # These T1s are sagittal: their axes run anterior-to-posterior,
                # inferior-to-superior, left-to-right, a full permutation away
                # from the template. Left alone that shows up as a 90 degree
                # rotation that swamps the few degrees of real pose and puts the
                # Euler decomposition in gimbal lock. fslreorient2std applies
                # only 90 degree flips and permutations, so it removes the
                # convention without touching the pose.
                fsl(f"fslreorient2std {p(row.nii)} {p(std)}")
            if not brain.exists():
                # -R is the robust centre estimate; these T1s include neck, and
                # a bad centre is the usual cause of a bet that eats occipital
                # cortex and then tilts the registration.
                fsl(f"bet {p(std)} {p(brain)} -R -f 0.4")
            fsl(f"flirt -in {p(brain)} -ref {REF} -omat {p(mat)} "
                f"-dof 12 -cost corratio "
                f"-searchrx -90 90 -searchry -90 90 -searchrz -90 90")
        A = np.loadtxt(mat)
        pitch, roll, yaw, total = decompose(A)
        sv = np.linalg.svd(A[:3, :3], compute_uv=False)
        return {"subject": row.subject, "session": row.session, "group": row.group,
                "t1_pitch": pitch, "t1_roll": roll, "t1_yaw": yaw,
                "t1_total": total, "t1_scale_ratio": float(sv.max() / sv.min())}
    except Exception as e:                                      # noqa: BLE001
        print(f"   FAIL {row.subject}/{row.session}: {repr(e)[:110]}", flush=True)
        return None


def hedges_g(a, b):
    na, nb = len(a), len(b)
    s = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    g = (a.mean() - b.mean()) / s * (1 - 3 / (4 * (na + nb) - 9))
    se = np.sqrt((na + nb) / (na * nb) + g ** 2 / (2 * (na + nb - 2)))
    return g, g - 1.96 * se, g + 1.96 * se


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    assert_group_mapping()

    if not args.report:
        global REF
        REF = _ref()
        print(f"registration target: {REF}")
        rows = [{"subject": f.parts[len(DEST.parts)],
                 "session": next(x for x in f.parts if x.startswith("ses-")),
                 "group": group(f.parts[len(DEST.parts)]), "nii": str(f)}
                for f in sorted(DEST.rglob("*_T1w.nii.gz"))]
        src = pd.DataFrame(rows)
        print(f"{len(src)} T1 images "
              f"({(src.group=='patient').sum()} patient, "
              f"{(src.group=='control').sum()} control), {args.jobs} at a time\n")
        done = []
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            for i, r in enumerate(ex.map(process, list(src.itertuples())), 1):
                if r:
                    done.append(r)
                if i % 10 == 0:
                    print(f"   {i}/{len(src)}  ({len(done)} ok)", flush=True)
        pd.DataFrame(done).to_csv(OUT, index=False)

    t = pd.read_csv(OUT)
    bad = t.t1_scale_ratio > 1.6
    if bad.any():
        print(f"QC dropped {int(bad.sum())}: {t[bad].groupby('group').size().to_dict()}")
    t = t[~bad]
    for c in ("t1_pitch", "t1_roll", "t1_yaw"):
        t[c] = t[c].abs()
    t["sid"] = t.subject.str.replace("sub-", "", regex=False)
    dem = demographics().rename(columns={"subject": "sid"}).drop(columns=["group"])
    t = t.merge(dem, on="sid", how="left")
    t["patient"] = (t.group == "patient").astype(int)
    t["male"] = (t.sex == "Male").astype(float)

    sub = (t.groupby(["sid", "patient", "male"], as_index=False)
             [["t1_pitch", "t1_roll", "t1_yaw", "t1_total", "age"]].mean())
    print(f"\n{len(t)} T1 scans, {len(sub)} subjects "
          f"({int(sub.patient.sum())} patient, {int((1-sub.patient).sum())} control)\n")

    print("=== does the group difference replicate in T1? ===")
    print(f"{'axis':<10s} {'patient':>9s} {'control':>9s} {'Welch p':>9s} "
          f"{'MW p':>7s} {'Hedges g [95% CI]':>24s}")
    for c in ("t1_pitch", "t1_roll", "t1_yaw", "t1_total"):
        a = sub.loc[sub.patient == 1, c].dropna()
        b = sub.loc[sub.patient == 0, c].dropna()
        _, pv = stats.ttest_ind(a, b, equal_var=False)
        _, pu = stats.mannwhitneyu(a, b)
        g, lo, hi = hedges_g(a, b)
        print(f"{c:<10s} {a.mean():9.2f} {b.mean():9.2f} {pv:9.3f} {pu:7.3f}   "
              f"{g:6.2f} [{lo:5.2f}, {hi:5.2f}]")

    print("\n   adjusted for age and sex:")
    for c in ("t1_pitch", "t1_roll", "t1_yaw", "t1_total"):
        d = sub.dropna(subset=[c, "age", "male"])
        X = np.column_stack([np.ones(len(d)), d.patient.to_numpy(float),
                             d.age.to_numpy(float), d.male.to_numpy(float)])
        y = d[c].to_numpy(float)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        res = y - X @ beta
        dof = len(d) - X.shape[1]
        se = np.sqrt(np.diag(np.linalg.pinv(X.T @ X)) * (res @ res) / dof)
        pv = 2 * stats.t.sf(abs(beta[1] / se[1]), dof)
        print(f"      {c:<10s} beta={beta[1]:5.2f}  p={pv:.3f}")

    # --- do the two modalities agree, scan by scan? -------------------------
    fa = pd.read_csv(HERE / "ds001907_pose.csv")
    for c in ("pitch", "roll", "yaw"):
        fa[c] = fa[c].abs()
    j = fa.merge(t, on=["subject", "session", "group"], how="inner")
    print(f"\n=== T1 against FA, same session, same head ({len(j)} scans) ===")
    for c in ("pitch", "roll", "yaw", "total"):
        r, pv = stats.pearsonr(j[c], j[f"t1_{c}"])
        print(f"   {c:<8s} r={r:5.2f}  p={pv:.4f}")
    j.to_csv(HERE / "ds001907_pose_both.csv", index=False)
    print(f"\n   wrote {HERE / 'ds001907_pose_both.csv'}")


if __name__ == "__main__":
    main()
