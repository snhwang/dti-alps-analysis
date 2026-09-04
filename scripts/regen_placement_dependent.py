"""Regenerate every placement-dependent output and report what moves.

The revert from 5 mm spheres back to the warped masks restored the source
files by renaming backups into place. A rename keeps the old modification
time, so anything computed from the canonical filename while it briefly held
sphere data still looks current. Timestamps cannot detect this. Only
regenerating and diffing can.

phenotype_sweep.csv was found that way: every one of its 219 rows changed, a
survivor count moved, and the ordering statistic went from p=0.005 to p=0.32.
diagnostic_worked_example.csv and angle_beyond_ratio.csv moved too. Others,
including beyond_eigenvalue_ratio.csv, regenerate identically, which is what
makes this worth doing exhaustively rather than by guess.

Each script is run, its outputs compared against a copy taken first, and the
largest absolute change in any numeric column reported. Nothing is reverted:
the regenerated file is the correct one, and the point of the report is to say
which manuscript numbers now need checking.

    python regen_placement_dependent.py --list
    python regen_placement_dependent.py --run
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent

# script -> outputs it writes, for the analyses whose values depend on where
# the measurement region sits. Ordered so that anything feeding another
# analysis runs before it.
TARGETS = [
    ("beyond_eigenvalue_ratio.py", ["beyond_eigenvalue_ratio.csv"]),
    ("phenotype_sweep.py", ["phenotype_sweep.csv"]),
    ("phenotype_pose_adjusted.py", ["phenotype_pose_adjusted.csv"]),
    ("phenotype_sweep_dlbs.py", ["phenotype_sweep_dlbs.csv"]),
    ("diagnostic_worked_example.py", ["diagnostic_worked_example.csv"]),
    ("angle_beyond_ratio.py", ["angle_beyond_ratio.csv"]),
    ("cardiometabolic.py", ["cardiometabolic.csv"]),
    ("normalization_comparison.py", ["normalization_comparison.csv"]),
    ("reconciliation_table.py", ["reconciliation_table.csv"]),
    ("ratio_bound_proof.py", ["ratio_bound_proof.csv"]),
    ("sorting_bias_check.py", ["sorting_bias_check.csv"]),
    ("registration_age_dependence.py", ["registration_age_dependence.csv"]),
    ("shortfall_decomposition.py", ["shortfall_decomposition.csv"]),
    ("manual_vs_atlas_icc.py", ["manual_vs_atlas_icc.csv"]),
    ("slab_prescription_control.py", ["slab_prescription_dlbs.csv"]),
]


def biggest_change(a: Path, b: Path) -> float | str:
    try:
        x, y = pd.read_csv(a), pd.read_csv(b)
    except Exception as e:
        return f"unreadable ({str(e)[:30]})"
    if x.shape != y.shape:
        return f"shape {x.shape} -> {y.shape}"
    num = [c for c in x.columns
           if c in y.columns and x[c].dtype.kind in "fi" and y[c].dtype.kind in "fi"]
    if not num:
        return 0.0
    return max(float((x[c] - y[c]).abs().max()) for c in num)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()

    if not args.run:
        print(f"{len(TARGETS)} scripts, "
              f"{sum(len(o) for _, o in TARGETS)} outputs. Pass --run.")
        for s, outs in TARGETS:
            print(f"  {s:34s} -> {', '.join(outs)}")
        return

    tmp = Path(tempfile.mkdtemp(prefix="regen_"))
    results = []
    for script, outs in TARGETS:
        sp = HERE / script
        if not sp.exists():
            results.append((script, "-", "script missing"))
            continue
        saved = {}
        for o in outs:
            p = HERE / o
            if p.exists():
                saved[o] = tmp / o
                shutil.copy2(p, saved[o])
        print(f"running {script} ...", flush=True)
        r = subprocess.run([sys.executable, str(sp)], cwd=str(HERE),
                           capture_output=True, text=True, timeout=args.timeout)
        if r.returncode != 0:
            tail = (r.stderr or "").strip().splitlines()[-1:] or ["(no stderr)"]
            results.append((script, "-", f"FAILED: {tail[0][:60]}"))
            continue
        for o in outs:
            if o not in saved:
                results.append((script, o, "created (no prior)"))
                continue
            results.append((script, o, biggest_change(saved[o], HERE / o)))

    print(f"\n{'output':<40s} {'largest change':>16s}  script")
    for script, out, delta in results:
        if isinstance(delta, float):
            tag = f"{delta:.6f}"
            mark = "" if delta < 1e-9 else "   <-- CHANGED"
        else:
            tag, mark = str(delta)[:16], "   <-- see note"
        print(f"  {out:<40s} {tag:>16s}  {script}{mark}")
    print(f"\ncopies of the previous files are in {tmp}")
    print("Anything CHANGED means a manuscript number computed from it needs "
          "rechecking. Run verify_manuscript.py next.")


if __name__ == "__main__":
    main()
