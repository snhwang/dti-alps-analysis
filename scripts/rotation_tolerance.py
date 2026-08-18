"""
How much head rotation can classic DTI-ALPS tolerate before it is too wrong?

HCP-A is anatomically aligned during preprocessing, so head positioning has
already been removed and the unrotated value is a valid reference: rotating a
head cannot change its perivascular diffusivity. Imposing known rotations
therefore gives an error curve with a meaningful zero.

The earlier version of this asked only where the classic and refined curves
cross. That answers "which method is better" but not the question an author
actually has, which is "my cohort has this much head rotation, can I use the
standard index". This inverts the curve: for a stated error tolerance, the
rotation at which classic exceeds it.

Rotations are applied as exact coordinate transformations of the fitted tensors,
D' = R D R^T, which introduces no interpolation. It does not reproduce the
acquisition-stage effects of actually imaging a tilted head (field of view and
slice angle change partial-volume mixing), so these are lower bounds on the real
error.

Reported per axis as well as isotropically, because pitch is both the largest
rotation that occurs in practice and the most damaging one, so a single
isotropic number understates the risk for the rotation people actually have.

Usage:
    python rotation_tolerance.py --repeats 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import atomic_io  # noqa: F401  writes become atomic on import

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rotation_study_slab import load, evaluate, euler_rotation

HERE = Path(__file__).resolve().parent
METHODS = ["classic", "refined", "refined+", "ALPS-PAS", "per-voxel"]
TOL = [1.0, 2.0, 5.0, 10.0]
GRID = np.arange(0, 31, 1.0)


def curve(sessions, truth, mode, rng, repeats):
    """Mean |error| % against the unrotated classic reference, per method."""
    out = {m: [] for m in METHODS}
    for sig in GRID:
        errs = {m: [] for m in METHODS}
        reps = 1 if sig == 0 else repeats
        for _ in range(reps):
            for i, s in enumerate(sessions):
                if sig == 0:
                    R = np.eye(3)
                elif mode == "isotropic":
                    R = euler_rotation(*(rng.normal(0, 1, 3) * sig))
                else:
                    a3 = np.zeros(3); a3[mode] = rng.normal(0, sig)
                    R = euler_rotation(*a3)
                v = evaluate(s, R)
                for m in METHODS:
                    errs[m].append(100 * abs(v[m] - truth[i]) / truth[i])
        for m in METHODS:
            out[m].append(float(np.mean(errs[m])))
    return {m: np.array(v) for m, v in out.items()}


def crossing(x, y, level):
    """First x at which y rises through level, linearly interpolated."""
    for a, b in zip(range(len(x) - 1), range(1, len(x))):
        if y[a] <= level < y[b]:
            t = (level - y[a]) / (y[b] - y[a])
            return float(x[a] + t * (x[b] - x[a]))
    return np.nan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=8)
    args = ap.parse_args()

    sessions = load()
    print(f"{len(sessions)} sessions, "
          f"{len({s['Subject_ID'] for s in sessions})} participants\n")
    base = [evaluate(s) for s in sessions]
    truth = np.array([b["classic"] for b in base])

    rng = np.random.default_rng(20260810)
    modes = [("isotropic", "isotropic"), (0, "pitch (x)"), (1, "roll (y)"), (2, "yaw (z)")]
    rows, tol_rows = [], []
    for mode, nm in modes:
        c = curve(sessions, truth, mode, rng, args.repeats)
        for k, sig in enumerate(GRID):
            rows.append({"mode": nm, "sigma": sig, **{m: c[m][k] for m in METHODS}})
        print(f"=== {nm} ===")
        print("  rotation at which each method's mean error first exceeds:")
        print("  " + " ".join(f"{('%g%%' % t):>9s}" for t in TOL))
        for m in METHODS:
            xs = [crossing(GRID, c[m], t) for t in TOL]
            tol_rows.append({"mode": nm, "method": m,
                             **{f"tol_{t:g}pct": x for t, x in zip(TOL, xs)}})
            cells = " ".join(("     n/a" if np.isnan(x) else f"{x:8.1f}d") for x in xs)
            print(f"  {m:<10s} {cells}")
        # where the corrected variants overtake classic
        for m in METHODS[1:]:
            cross = crossing(GRID, c["classic"] - c[m], 0.0)
            if not np.isnan(cross):
                print(f"    classic exceeds {m} beyond {cross:.1f} deg")
        print()

    pd.DataFrame(rows).to_csv(HERE / "rotation_tolerance_curves.csv", index=False)
    pd.DataFrame(tol_rows).to_csv(HERE / "rotation_tolerance.csv", index=False)
    print("wrote rotation_tolerance{,_curves}.csv")


if __name__ == "__main__":
    main()
