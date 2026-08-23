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

Usage. DLBS is already done and its results are in the manuscript. For HCP-A:

    python run_ld_alps.py --cohort hcpa --limit 3 --chunk 3 --keep   # smoke test
    python run_ld_alps.py --cohort hcpa --chunk 40                   # full run

The full run is long. It writes a shell-filtered copy of each session's DWI,
which the single-shell path does not have to do, so budget hours rather than
minutes and expect to restart it at least once. That is safe: results append
after every chunk and sessions already present in ld_alps_<cohort>.csv are
skipped, so re-invoking the same command resumes where it stopped. Staging is
cleared per chunk, so an interrupted chunk cannot leave a truncated volume
behind to be picked up as if it were complete.

Writes ld_alps_<cohort>.csv.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd

import atomic_io  # noqa: F401  writes become atomic on import
from data_paths import winpath

# Staging dominates the runtime, and almost all of it is gzip. Measured on
# HCP-A: 38 s per session, of which about 30 s is writing the filtered series.
# The vendored loader opens "eddy_corrected_data.nii.gz" by that exact name, so
# the file has to stay gzipped and cannot simply be written uncompressed.
# nibabel's compression level is a supported module-level knob, though, and
# level 1 costs a little disk for a large fraction of that time. These files are
# deleted at the end of their chunk, so the disk does not matter.
nib.openers.Opener.default_compresslevel = 1

HERE = Path(__file__).resolve().parent
DIFF = HERE.parent.parent / "diffusion"
OUT = winpath("Q:/dti_output")
VENDORED = HERE / "external" / "ld-alps.py"
WORK = winpath("Q:/ld_alps_work")   # same volume as the sessions, so hardlinks work

# Their label order, mapped onto the ROI files we already warp into native space.
LABEL_ORDER = (("R_SLF", 1), ("R_SCR", 2), ("L_SLF", 3), ("L_SCR", 4))

# Which shell each cohort is analysed on. DLBS has one, b=1000, so nothing is
# filtered. HCP-A has b=1500 and b=3000 and the paper fits b=1500, which is why
# its tensors are written with a _b1500 suffix. LD-ALPS does not use the tensor:
# it computes an apparent diffusion coefficient by interpolating the measured
# signal across the acquired gradient directions. Handing it both shells would
# therefore mix them into the ADC and give a number that is not comparable with
# any other variant in the paper. There is no pre-filtered series on disk, so
# the shell is selected here.
SHELL = {"hcpa": 1500, "dlbs": None, "tn": None}

# A volume at or below this b is a b0. HCP-A labels its b0 volumes b=5 rather
# than b=0, which is common practice, but the vendored code identifies them with
# an exact `bvals == 0` test and raises "bvals must contain both 0 and nonzero
# entries" when none matches. The b0 labels are therefore normalized to 0 on the
# way in. Only the labels: the diffusion-weighted b values are left exactly as
# acquired, because compute_adc_volume uses each volume's own b in the ADC, so
# the 1490 to 1510 spread of a nominal 1500 shell is meaningful and rounding it
# would change every ADC.
B0_MAX = 100.0


def prepare(sdir: Path, dest: Path, shell: int | None = None) -> bool:
    """Lay out one session the way their loader expects. Returns False if incomplete.

    DLBS is eddy-corrected by our pipeline, so it has dwi_eddy_corrected and the
    rotated bvecs. HCP-A does not, because it is distributed already corrected by
    the HCP minimal preprocessing pipeline and our pipeline consumed it as given.
    Its inputs/dwi.nii.gz is therefore the corrected series and inputs/dwi.bvec
    the gradients already rotated to match. Falling back to those is the correct
    equivalent, not a degradation, but it is a substitution and is made
    explicitly here rather than by silently accepting whatever is on disk.
    """
    proc = sdir / "processed"
    atlas = proc / "atlas"
    eddy = proc / "dwi_eddy_corrected.nii.gz"
    bvec = proc / "dwi_eddy_rotated.bvec"
    if not eddy.exists():
        eddy = sdir / "inputs" / "dwi.nii.gz"
        bvec = sdir / "inputs" / "dwi.bvec"
    bval = sdir / "inputs" / "dwi.bval"
    # The principal eigenvector must come from the same shell as everything
    # else. LD-ALPS uses V1 only to pick each region's direction, but a V1 from
    # the b=3000 fit would select different voxels than every other variant in
    # the paper does, so the suffixed tensors are used where they exist.
    evecs = proc / "tensor_eigenvectors.nii.gz"
    if shell is not None:
        suffixed = proc / f"tensor_eigenvectors_b{shell}.nii.gz"
        if not suffixed.exists():
            return False
        evecs = suffixed
    sph_dir = atlas / "sphere_roi"
    if not all(p.exists() for p in (eddy, bvec, bval, evecs)):
        return False
    if not all((sph_dir / f"{n}_native.nii.gz").exists() for n, _ in LABEL_ORDER):
        return False

    dest.mkdir(parents=True, exist_ok=True)
    if shell is None:
        # Single-shell cohort: their loader opens these by name, so link or copy
        # rather than pass paths. Linking keeps this nearly free.
        b_all = np.loadtxt(bval).ravel()
        needs_b0_fix = not (b_all == 0).any()
        for src, name in ((eddy, "eddy_corrected_data.nii.gz"),
                          (bvec, "eddy_corrected_data.eddy_rotated_bvecs"),
                          (bval, "bvals")):
            tgt = dest / name
            if tgt.exists():
                continue
            if name == "bvals" and needs_b0_fix:
                # No exact zeros, so write a normalized copy instead of linking.
                b_fixed = np.where(b_all <= B0_MAX, 0.0, b_all)
                np.savetxt(tgt, b_fixed[None, :], fmt="%g")
                continue
            try:
                os.link(src, tgt)
            except OSError:
                shutil.copyfile(src, tgt)
    else:
        # Multi-shell: keep b=0 and the analysed shell, drop the rest. The
        # filtered series has to be written out, so this costs real time and
        # disk where the single-shell path costs neither.
        tgt = dest / "eddy_corrected_data.nii.gz"
        if not tgt.exists():
            b = np.loadtxt(bval).ravel()
            is_b0 = b <= B0_MAX
            keep = is_b0 | (np.abs(b - shell) <= 100)
            if keep.sum() < 10 or not is_b0.any():
                return False
            img = nib.load(str(eddy))
            data = np.asanyarray(img.dataobj)[..., keep]
            nib.save(nib.Nifti1Image(data, img.affine, img.header), str(tgt))
            # b0 labels to exactly 0; weighted volumes keep their acquired b.
            b_out = np.where(is_b0, 0.0, b)[keep]
            np.savetxt(dest / "bvals", b_out[None, :], fmt="%g")
            v = np.loadtxt(bvec)
            if v.shape[0] != 3:
                v = v.T
            np.savetxt(dest / "eddy_corrected_data.eddy_rotated_bvecs",
                       v[:, keep], fmt="%.6f")

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
    ap.add_argument("--cohort", choices=["dlbs", "hcpa", "tn"], default="dlbs")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--stage-jobs", type=int, default=8, dest="stage_jobs",
                    help="parallel workers for staging, the gzip-bound step")
    ap.add_argument("--chunk", type=int, default=40,
                    help="sessions per batch; results append after each batch")
    ap.add_argument("--keep", action="store_true", help="leave the staged inputs on disk")
    args = ap.parse_args()

    if not VENDORED.exists():
        raise SystemExit(f"vendored LD-ALPS not found at {VENDORED}")

    # The cohort flag used to select only the output directory: the session list
    # was read from DLBS whatever was passed, so --cohort tn silently ran DLBS
    # sessions into a tn folder. Each cohort now selects its own session table.
    if args.cohort == "hcpa":
        src = pd.read_csv(DIFF / "HCP" / "hcpa_alps_spheres_5mm.csv")
        src = src[src.status == "ok"].copy()
        src["Visit"] = src["Visit"].astype(str)
        src["Subject_ID"] = src.Subject_ID.astype(str)
        # 2742 sessions pass sphere placement but only 1706 carry b=1500
        # tensors, and those 1706 are the sample every other variant is
        # reported on. Iterating all of them would stage and then discard a
        # third of the run. The test is the same one prepare() applies, made on
        # the filesystem rather than against a side table, so it holds from a
        # clone of either repository.
        # 2742 sessions pass sphere placement, 2226 carry b=1500 tensors, and
        # 1706 are in the published sample every other variant is reported on.
        # The difference between the last two is sessions where the variant
        # computation itself failed, which no test on the filesystem recovers,
        # so the published table is the authority. Running a different sample
        # silently would make LD-ALPS incomparable with the rest of the paper,
        # so its absence is an error rather than a skipped filter.
        pub = HERE / "measured_pvs_axis_hcpa_b1500_all.csv"
        if not pub.exists():
            raise SystemExit(
                f"{pub.name} is required to select the published HCP-A sample "
                f"and is not beside this script. It is a derived HCP-A file and "
                f"cannot be redistributed under the AABC terms, so copy it in "
                f"from the analysis directory that produced it.")
        p = pd.read_csv(pub)
        keep = {(str(a), str(b)) for a, b in zip(p.Subject_ID, p.Visit)}
        before = len(src)
        src = src[[(a, b) in keep for a, b in zip(src.Subject_ID, src.Visit)]]
        print(f"restricted to the published b=1500 sample: "
              f"{len(src)} of {before} sessions", flush=True)
    else:
        src = pd.read_csv(DIFF / "DLBS" / "dlbs_alps_spheres_5mm.csv")
        src = src[src.status == "ok"].copy()
        src["Visit"] = src["Session"]
    if args.limit:
        src = src.head(args.limit)

    base = WORK / args.cohort
    base.mkdir(parents=True, exist_ok=True)
    final = HERE / f"ld_alps_{args.cohort}.csv"

    # Accumulate in chunks and append after each one. A run over a few thousand
    # sessions will be interrupted sooner or later, and staging everything then
    # writing once at the end means an interruption costs the whole run. Sessions
    # already present in the output are skipped, so re-invoking resumes.
    done = set()
    if final.exists():
        prev = pd.read_csv(final)
        if {"Subject_ID", "Visit"} <= set(prev.columns):
            done = {(str(a), str(b)) for a, b in zip(prev.Subject_ID, prev.Visit)}
        print(f"resuming: {len(done)} sessions already in {final.name}")

    todo = [r for r in src.itertuples()
            if (str(r.Subject_ID), str(r.Visit)) not in done]
    print(f"{len(todo)} sessions to run, in chunks of {args.chunk}", flush=True)

    n_written = 0
    for start in range(0, len(todo), args.chunk):
        batch = todo[start:start + args.chunk]
        cdir = base / f"chunk_{start // args.chunk:04d}"
        # Clear before staging. An interrupted run leaves a half-written
        # eddy_corrected_data.nii.gz behind, and the "skip if it exists" guards
        # below would accept the truncated file and hand it to LD-ALPS. A chunk
        # is cheap to rebuild and is processed as a unit, so start it clean.
        shutil.rmtree(cdir, ignore_errors=True)
        cdir.mkdir(parents=True, exist_ok=True)
        # Stage in parallel. zlib releases the GIL, so threads genuinely overlap
        # the compression that dominates this step, and the reads come off one
        # volume so more workers stop helping fairly quickly.
        def _stage(r):
            dest = cdir / f"alps_{r.Subject_ID}_{r.Visit}"
            if prepare(OUT / r.DTI_Session_ID, dest, SHELL.get(args.cohort)):
                return (str(r.Subject_ID), str(r.Visit), dest.name)
            return None

        with ThreadPoolExecutor(max_workers=args.stage_jobs) as ex:
            staged = [x for x in ex.map(_stage, batch) if x is not None]
        if not staged:
            print(f"   chunk {start // args.chunk}: nothing staged, skipping",
                  flush=True)
            shutil.rmtree(cdir, ignore_errors=True)
            continue

        raw = cdir / "ld_alps_raw.csv"
        proc = subprocess.run(
            [sys.executable, str(VENDORED), str(cdir), "--subject-prefix", "alps_",
             "--out", str(raw)], capture_output=True, text=True)
        if proc.returncode != 0 or not raw.exists():
            print(f"   chunk {start // args.chunk} FAILED (exit {proc.returncode})")
            print(proc.stderr[-800:])
            if not args.keep:
                shutil.rmtree(cdir, ignore_errors=True)
            continue

        d = pd.read_csv(raw)
        key = pd.DataFrame(staged, columns=["Subject_ID", "Visit", "subject"])
        idc = next((c for c in d.columns
                    if c.lower() in ("subject", "subject_id", "id")), None)
        if idc is not None:
            d = d.rename(columns={idc: "subject"}).merge(key, on="subject", how="left")
        # Append rather than overwrite, so earlier chunks survive a later failure.
        d.to_csv(final, mode="a", header=not final.exists(), index=False)
        n_written += len(d)
        print(f"   chunk {start // args.chunk}: +{len(d)} rows "
              f"({n_written} this run) -> {final.name}", flush=True)
        if not args.keep:
            shutil.rmtree(cdir, ignore_errors=True)

    if final.exists():
        allr = pd.read_csv(final)
        print(f"\n{len(allr)} sessions total in {final.name}")
        print("columns:", list(allr.columns))
    if not args.keep:
        shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    main()
