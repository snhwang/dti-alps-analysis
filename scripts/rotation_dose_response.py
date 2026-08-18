"""
How much head rotation must be present before orientation correction pays off?

The refined index removes orientation-driven error but pays for it in
estimation variance, since its measurement axes are estimated from noisy
eigenvectors rather than fixed. On HCP-A that cost is about 20 percent of the
within-participant variance. There is therefore a crossover: below some amount
of between-session head rotation the fixed-axis index is more reproducible, and
above it the refined index is.

HCP-A is the right cohort for locating that crossover precisely because its
diffusion data is already AC-PC aligned, with an axis-aligned image affine. It
supplies a clean zero-rotation baseline carrying real repeated-measures
variance (genuine re-acquisitions, real physiological and measurement noise),
onto which a known amount of rotation can be added.

Design: each session receives its own independently drawn rotation, which is
what varying head positioning between visits actually looks like. Tensors
transform as D' = R D R^T, implemented by rotating the eigenvectors and leaving
the eigenvalues unchanged. Within-participant variance is then recomputed for
each variant as a function of rotation magnitude, and the crossover is the
rotation at which the classic index becomes less reproducible than the refined
one.

This is a coordinate-space simulation. It does not reproduce the acquisition
stage effects of imaging a differently oriented head, in particular the changes
in partial-volume mixing and in-plane averaging that a different slice
prescription would produce, so it isolates the geometric component only. That
limitation is the reason it is used to calibrate a threshold rather than to
claim empirical robustness.

Usage:
    python rotation_dose_response.py
    python rotation_dose_response.py --model realistic
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import atomic_io  # noqa: F401  writes become atomic on import

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from estimator_variants import CACHE, alps_from_rois, variance_components

HERE = Path(__file__).resolve().parent

# Burles et al. measured head pose against MNI in ADNI and reported SDs of
# 7.89 deg pitch, 2.23 roll and 3.34 yaw. The realistic model keeps those
# proportions; the isotropic model gives all three axes equal weight.
REALISTIC = np.array([1.0, 2.23 / 7.89, 3.34 / 7.89])
ISOTROPIC = np.array([1.0, 1.0, 1.0])

SIGMAS = [0, 2, 4, 6, 8, 10, 12, 15, 20, 25, 30]
N_REPEATS = 20
ESTIMATOR = "dyadic"


def euler_rotation(pitch: float, roll: float, yaw: float) -> np.ndarray:
    """Rotation matrix from intrinsic pitch (about x), roll (y), yaw (z), in degrees."""
    p, r, y = np.radians([pitch, roll, yaw])
    Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    Ry = np.array([[np.cos(r), 0, np.sin(r)], [0, 1, 0], [-np.sin(r), 0, np.cos(r)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def load_sessions() -> list[dict]:
    out = []
    for f in sorted(CACHE.glob("*.npz")):
        sub, visit = f.stem.rsplit("_", 1)
        z = np.load(f)
        rec = {"Subject_ID": sub, "Visit": visit, "hemis": {}}
        ok = True
        for hemi in ("L", "R"):
            try:
                proj = {k: z[f"proj_{hemi}_{k}"].astype(np.float64)
                        for k in ("v1", "fa", "evals", "evecs")}
                assoc = {k: z[f"assoc_{hemi}_{k}"].astype(np.float64)
                         for k in ("v1", "fa", "evals", "evecs")}
            except KeyError:
                ok = False
                break
            if len(proj["v1"]) < 4 or len(assoc["v1"]) < 4:
                ok = False
                break
            rec["hemis"][hemi] = (proj, assoc)
        if ok:
            out.append(rec)
    return out


def rotate_roi(roi: dict, R: np.ndarray) -> dict:
    """Apply D' = R D R^T by rotating eigenvectors; eigenvalues are unchanged."""
    return {
        "v1": roi["v1"] @ R.T,
        "fa": roi["fa"],
        "evals": roi["evals"],
        "evecs": np.einsum("ij,njk->nik", R, roi["evecs"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["isotropic", "realistic"], default="isotropic")
    ap.add_argument("--repeats", type=int, default=N_REPEATS)
    args = ap.parse_args()
    weights = ISOTROPIC if args.model == "isotropic" else REALISTIC

    sessions = load_sessions()
    subs = pd.Series([s["Subject_ID"] for s in sessions]).value_counts()
    keep = set(subs[subs >= 2].index)
    sessions = [s for s in sessions if s["Subject_ID"] in keep]
    print(f"sessions {len(sessions)}, participants {len({s['Subject_ID'] for s in sessions})}")
    print(f"rotation model: {args.model}  (per-axis SD weights {np.round(weights,2)})")
    print(f"estimator: {ESTIMATOR}, {args.repeats} random draws per level\n")

    rng = np.random.default_rng(20260728)
    rows = []
    for sigma in SIGMAS:
        cw, rw = [], []
        for rep in range(args.repeats if sigma > 0 else 1):
            recs = []
            for s in sessions:
                if sigma > 0:
                    ang = rng.normal(0, 1, 3) * weights * sigma
                    R = euler_rotation(*ang)
                else:
                    R = np.eye(3)
                vals = []
                for hemi, (proj, assoc) in s["hemis"].items():
                    vals.append(alps_from_rois(rotate_roi(proj, R),
                                               rotate_roi(assoc, R), ESTIMATOR))
                recs.append({
                    "Subject_ID": s["Subject_ID"],
                    "classic": float(np.mean([v["classic"] for v in vals])),
                    "refined": float(np.mean([v["refined"] for v in vals])),
                })
            d = pd.DataFrame(recs)
            cw.append(variance_components(d, "classic")["var_within"])
            rw.append(variance_components(d, "refined")["var_within"])
        rows.append({"sigma": sigma,
                     "classic_var": float(np.mean(cw)),
                     "refined_var": float(np.mean(rw)),
                     "classic_sd": float(np.std(cw)),
                     "refined_sd": float(np.std(rw))})
        r = rows[-1]
        print(f"  sigma {sigma:>2}deg   classic {r['classic_var']:.6f}   "
              f"refined {r['refined_var']:.6f}   ratio {r['classic_var']/r['refined_var']:.3f}")

    df = pd.DataFrame(rows)
    df["ratio"] = df.classic_var / df.refined_var
    df.to_csv(HERE / f"rotation_dose_response_{args.model}.csv", index=False)

    cross = None
    for a, b in zip(df.itertuples(), df.iloc[1:].itertuples()):
        if a.ratio <= 1.0 < b.ratio:
            t = (1.0 - a.ratio) / (b.ratio - a.ratio)
            cross = a.sigma + t * (b.sigma - a.sigma)
            break
    print()
    if cross is not None:
        print(f"crossover: classic becomes less reproducible than refined at about "
              f"{cross:.1f} deg per-axis rotation SD")
    else:
        print("no crossover within the range tested")
    print(f"Wrote rotation_dose_response_{args.model}.csv")


if __name__ == "__main__":
    main()
