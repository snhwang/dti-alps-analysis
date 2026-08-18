"""Regenerate every cached result and report anything that drifted.

Re-running a script proves it still runs. It does not prove the committed CSV
matches what the script now produces, and that gap is where a wrong number
survives: the manuscript is checked against the CSV, the CSV is trusted because
a script exists, and nobody compares the two. tn_sphere_qc.csv is the worked
example. Its placement claim was wrong for months, and the reason it was never
caught is that it had no script to re-run at all.

So this regenerates each cached result and diffs it numerically against the
version committed in git, column by column. Three outcomes per file:

  identical   byte-for-byte or within 1e-9 on every numeric column
  DRIFT       a numeric column moved, reported with the column and the size
  new         not in git yet

DRIFT is the finding. It means the committed CSV and the current code disagree,
so either the code changed without the results being refreshed, or the results
were edited by hand, or an input moved underneath both. Any of those invalidates
whatever the manuscript says about that file until it is understood.

Order matters. Later steps read earlier outputs, and the ALPS_TENSOR_SUFFIX
variable selects the b1500 shell for HCP-A, which is the cohort-rebuild gotcha:
without it the scripts silently read the legacy unsuffixed tensors.

    python regenerate_and_diff.py --dry-run       show the plan and timings
    python regenerate_and_diff.py                 run everything
    python regenerate_and_diff.py --only pose     run one group
    python regenerate_and_diff.py --skip-slow     omit the multi-minute steps

Nothing is written outside revision/, and git is never touched, so an
interrupted run leaves the tree exactly as a normal re-run would.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from data_paths import winpath

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
B1500 = {"ALPS_TENSOR_SUFFIX": "_b1500"}

# (group, label, argv, env, outputs, approx_minutes)
STEPS = [
    ("pose", "head rotation, HCP-A",
     ["head_rotation_observed.py", "--cohort", "hcpa"], {}, ["head_rotation_hcpa.csv"], 2),
    ("pose", "head rotation, DLBS",
     ["head_rotation_observed.py", "--cohort", "dlbs"], {}, ["head_rotation_dlbs.csv"], 1),
    ("pose", "head rotation, trigeminal",
     ["head_rotation_observed.py", "--cohort", "tn"], {}, ["head_rotation_tn.csv"], 1),
    ("pose", "slab prescription control",
     ["slab_prescription_control.py"], {}, ["slab_prescription_dlbs.csv"], 2),

    ("regions", "placement quality, HCP-A",
     ["roi_placement_quality.py", "--cohort", "hcpa"], B1500,
     ["roi_placement_quality_hcpa_b1500.csv"], 17),
    ("regions", "placement quality, DLBS all sessions",
     ["roi_placement_quality.py", "--cohort", "dlbs", "--all-sessions",
      "--out", "roi_placement_quality_dlbs_all.csv"], {},
     ["roi_placement_quality_dlbs_all.csv"], 6),

    ("directions", "tract direction variation",
     ["tract_direction_variation.py"], {},
     ["tract_direction_variation.csv", "tract_direction_age.csv"], 1),
    ("directions", "does registration align tracts",
     ["registration_aligns_tracts.py", "--affine"], {},
     ["registration_aligns_tracts_affine.csv"], 3),

    ("variants", "anatomical x variant, DLBS",
     ["anatomical_x_variant.py", "--cohort", "dlbs"], {},
     ["anatomical_x_variant_dlbs.csv"], 3),
    ("variants", "anatomical x variant, HCP-A",
     ["anatomical_x_variant.py", "--cohort", "hcpa"], B1500,
     ["anatomical_x_variant_hcpa.csv"], 15),
    ("variants", "paired ICC bootstrap",
     ["variant_icc_bootstrap.py"], {}, ["variant_icc_bootstrap.csv"], 1),
    ("variants", "midline tolerance",
     ["midline_sensitivity.py"], {}, ["midline_sensitivity.csv"], 1),

    ("adjust", "joint geometry adjustment",
     ["joint_geometry_adjustment.py"], B1500, ["joint_geometry_adjustment.csv"], 2),
    ("adjust", "registration age dependence",
     ["registration_age_dependence.py"], {}, ["registration_age_dependence.csv"], 1),
    ("adjust", "phenotype, pose adjusted",
     ["phenotype_pose_adjusted.py"], B1500, ["phenotype_pose_adjusted.csv"], 2),
    ("adjust", "sorting bias against data quality",
     ["sorting_bias_check.py"], B1500, ["sorting_bias_check.csv"], 1),
    ("adjust", "sorting bias noise floor, simulation only",
     ["sorting_bias_floor.py"], {}, ["sorting_bias_floor.csv"], 1),
    ("variants", "shortfall decomposition against measured angles",
     ["shortfall_decomposition.py"], {}, ["shortfall_decomposition.csv"], 1),
    ("pose", "is the AC-PC alignment the method assumes actually done",
     ["acpc_assumption.py"], {}, ["acpc_assumption.csv"], 1),
    ("regions", "do the two hand-drawn hemispheres share a slice",
     ["hemisphere_slice_agreement.py"], {}, ["hemisphere_slice_agreement.csv"], 3),
    ("regions", "off-tract fraction of the hand-drawn regions",
     ["manual_roi_offtract.py"], {}, ["manual_roi_offtract.csv"], 3),
    ("regions", "placement reliability, one estimator for both region sets",
     ["manual_vs_atlas_icc.py"], {}, ["manual_vs_atlas_icc.csv"], 1),
]
SLOW = 5


REQUIREMENTS = "requirements.txt"   # uv pip install -r revision/requirements.txt


def check_environment() -> None:
    """Fail before the first script can touch its output, not several frames in.

    Windows and WSL are both fine now that paths go through winpath, so the only
    thing worth checking is whether the data is actually reachable. If it is not,
    the scripts would each open an output for writing, truncate it, and only then
    discover the missing input.
    """
    import sys

    data = winpath("Q:/dti_output")
    if data.exists():
        return
    print(f"Cannot see the processed sessions.\n")
    print(f"  looked in: {data}    (platform {sys.platform})")
    print()
    if sys.platform.startswith("linux"):
        print("  Under WSL the drives appear beneath /mnt. Check that Q: is mounted,")
        print("  or set WSL_DRIVE_ROOT if your wsl.conf uses a different prefix.")
    else:
        print("  Check that the Q: drive is mounted.")
    raise SystemExit(2)


def committed(rel: str):
    """The version of a CSV in HEAD, or None if it is not tracked."""
    r = subprocess.run(["git", "show", f"HEAD:revision/{rel}"],
                       capture_output=True, text=True, cwd=REPO)
    if r.returncode != 0 or not r.stdout.strip():
        return None
    from io import StringIO
    try:
        return pd.read_csv(StringIO(r.stdout))
    except Exception:
        return None


def compare(rel: str) -> tuple[str, str]:
    """Diff the on-disk CSV against HEAD. Returns (verdict, detail)."""
    p = HERE / rel
    if not p.exists():
        return "MISSING", "script did not write it"
    old = committed(rel)
    if old is None:
        return "new", "not tracked in git yet"
    new = pd.read_csv(p)
    # Report every kind of difference found, not the first one. Returning on a
    # column mismatch once masked a 736 to 1525 row change, which is the single
    # most important thing this tool exists to surface.
    notes = []
    if list(old.columns) != list(new.columns):
        added = [c for c in new.columns if c not in old.columns]
        dropped = [c for c in old.columns if c not in new.columns]
        bits = []
        if added:
            bits.append(f"columns added {added}")
        if dropped:
            bits.append(f"columns DROPPED {dropped}")
        notes.append("; ".join(bits))
    if len(old) != len(new):
        notes.append(f"rows {len(old)} -> {len(new)}")
    if notes:
        return "DRIFT", "; ".join(notes)
    worst, where = 0.0, ""
    for c in new.columns:
        if not pd.api.types.is_numeric_dtype(new[c]):
            if not old[c].astype(str).equals(new[c].astype(str)):
                return "DRIFT", f"non-numeric column {c} changed"
            continue
        a, b = old[c].to_numpy(float), new[c].to_numpy(float)
        both_nan = np.isnan(a) & np.isnan(b)
        d = np.where(both_nan, 0.0, np.abs(a - b))
        if np.isnan(d).any():
            return "DRIFT", f"{c}: NaN pattern changed"
        if d.max() > worst:
            worst, where = float(d.max()), c
    if worst > 1e-9:
        return "DRIFT", f"{where}: max abs change {worst:.6g}"
    return "identical", ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="run one group: pose regions directions variants adjust")
    ap.add_argument("--skip-slow", action="store_true",
                    help=f"omit steps estimated over {SLOW} minutes")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not args.dry_run:
        check_environment()

    steps = [s for s in STEPS if not args.only or s[0] == args.only]
    if args.skip_slow:
        steps = [s for s in steps if s[5] <= SLOW]
    total = sum(s[5] for s in steps)
    print(f"{len(steps)} steps, roughly {total} minutes\n")
    if args.dry_run:
        for g, lab, argv, env, outs, mins in steps:
            e = " ".join(f"{k}={v}" for k, v in env.items())
            print(f"  [{g:10s}] {lab:38s} ~{mins:2d} min   {e} python {' '.join(argv)}")
        return

    results = []
    t0 = time.time()
    for g, lab, argv, env, outs, mins in steps:
        print(f"[{g}] {lab} ...", flush=True)
        t = time.time()
        r = subprocess.run(["python"] + argv, cwd=HERE, capture_output=True, text=True,
                           env={**os.environ, **env})
        el = time.time() - t
        if r.returncode != 0:
            print(f"    FAILED rc={r.returncode}: {(r.stderr or r.stdout).strip()[-300:]}")
            results.append((lab, "FAILED", f"rc={r.returncode}"))
            continue
        for o in outs:
            verdict, detail = compare(o)
            mark = "  " if verdict == "identical" else ">>"
            print(f"  {mark} {o:44s} {verdict:10s} {detail}")
            results.append((o, verdict, detail))
        print(f"    {el:.0f}s", flush=True)

    print(f"\ntotal {(time.time() - t0) / 60:.1f} min\n")
    drift = [r for r in results if r[1] in ("DRIFT", "FAILED", "MISSING")]
    ident = [r for r in results if r[1] == "identical"]
    new = [r for r in results if r[1] == "new"]
    print(f"identical {len(ident)}   new {len(new)}   NEEDS ATTENTION {len(drift)}")
    for name, verdict, detail in drift:
        print(f"  {verdict:8s} {name}: {detail}")
    if not drift:
        print("\nEvery committed result matches what the current code produces.")


if __name__ == "__main__":
    main()
