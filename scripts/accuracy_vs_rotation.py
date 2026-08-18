"""
How much rotation before the refined index is more ACCURATE than the classic one?

Earlier work here compared methods on reliability, which is precision, not
accuracy. A method can be perfectly reproducible and consistently wrong. This
measures error against a reference instead.

The reference is classic ALPS computed on unrotated HCP-A data. HCP-A diffusion
is already AC-PC aligned, with an axis-aligned image affine, so the classic
index is being evaluated in the frame where its own assumptions hold best.
Treating that as the gold standard deliberately gives classic every advantage:
at zero rotation its error is zero by definition, while the refined index is
scored on how far it departs from classic's aligned answer.

Known rotations are then imposed. Rotating a head cannot change its
perivascular diffusivity, so the reference value stays valid and any deviation
is error. Classic accumulates rotation-induced error; the refined index carries
a fixed offset from the reference, arising because it measures along each
participant's actual tract axes rather than the scanner axes, which differ by
about 12 degrees of anatomy even in aligned data. The crossover is the rotation
at which classic's growing error exceeds the refined index's constant offset.

This is a coordinate-space simulation. It does not reproduce acquisition-stage
partial-volume effects, so it calibrates a threshold rather than claiming
empirical robustness.

Usage:
    python accuracy_vs_rotation.py
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
from estimator_variants import directional_diffusivity
from direction_estimators import weights_for, principal, align, X, Y, Z
from rotation_dose_response import euler_rotation, rotate_roi, load_sessions, REALISTIC, ISOTROPIC

HERE = Path(__file__).resolve().parent
SIGMAS = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30]


def classic(proj, assoc) -> float:
    return ((directional_diffusivity(proj["evals"], proj["evecs"], X)
             + directional_diffusivity(assoc["evals"], assoc["evecs"], X))
            / (directional_diffusivity(proj["evals"], proj["evecs"], Y)
               + directional_diffusivity(assoc["evals"], assoc["evecs"], Z)))


def refined(proj, assoc) -> float:
    vp = align(principal(proj["v1"], weights_for("cl", proj)), Z)
    va = align(principal(assoc["v1"], weights_for("cl", assoc)), Y)
    p = np.cross(vp, va)
    p = p / max(np.linalg.norm(p), 1e-12)
    op = np.cross(p, vp); op /= max(np.linalg.norm(op), 1e-12)
    oa = np.cross(p, va); oa /= max(np.linalg.norm(oa), 1e-12)
    return ((directional_diffusivity(proj["evals"], proj["evecs"], p)
             + directional_diffusivity(assoc["evals"], assoc["evecs"], p))
            / (directional_diffusivity(proj["evals"], proj["evecs"], op)
               + directional_diffusivity(assoc["evals"], assoc["evecs"], oa)))


def hemi_mean(s, fn, R=None):
    vals = []
    for proj, assoc in s["hemis"].values():
        if R is not None:
            proj, assoc = rotate_roi(proj, R), rotate_roi(assoc, R)
        vals.append(fn(proj, assoc))
    return float(np.mean(vals))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["isotropic", "realistic"], default="isotropic")
    ap.add_argument("--repeats", type=int, default=8)
    ap.add_argument("--limit", type=int, default=400)
    args = ap.parse_args()
    w = ISOTROPIC if args.model == "isotropic" else REALISTIC

    sessions = load_sessions()[: args.limit]
    print(f"sessions {len(sessions)}, rotation model {args.model}, "
          f"{args.repeats} draws per level")
    print("reference: classic ALPS on unrotated, AC-PC aligned data\n")

    # Reference value and the refined index's fixed offset from it.
    ref = np.array([hemi_mean(s, classic) for s in sessions])
    ref_ref = np.array([hemi_mean(s, refined) for s in sessions])
    offset = 100 * np.abs(ref_ref - ref) / ref
    print(f"refined offset from reference at zero rotation: "
          f"mean {offset.mean():.2f}%, median {np.median(offset):.2f}%\n")

    rng = np.random.default_rng(20260728)
    rows = []
    print(f"{'rotation':>9s} {'classic err %':>14s} {'refined err %':>14s} {'winner':>9s}")
    for sigma in SIGMAS:
        ce, re = [], []
        for _ in range(args.repeats if sigma > 0 else 1):
            for i, s in enumerate(sessions):
                R = np.eye(3) if sigma == 0 else euler_rotation(*(rng.normal(0, 1, 3) * w * sigma))
                ce.append(100 * abs(hemi_mean(s, classic, R) - ref[i]) / ref[i])
                re.append(100 * abs(hemi_mean(s, refined, R) - ref[i]) / ref[i])
        c, r = float(np.mean(ce)), float(np.mean(re))
        rows.append({"sigma": sigma, "classic_err": c, "refined_err": r})
        print(f"{sigma:>7}deg {c:14.3f} {r:14.3f} {'refined' if r < c else 'classic':>9s}")

    df = pd.DataFrame(rows)
    cross = None
    for a, b in zip(df.itertuples(), df.iloc[1:].itertuples()):
        if a.classic_err <= a.refined_err and b.classic_err > b.refined_err:
            t = (a.refined_err - a.classic_err) / \
                ((b.classic_err - a.classic_err) - (b.refined_err - a.refined_err))
            cross = a.sigma + t * (b.sigma - a.sigma)
            break
    print()
    if cross is not None:
        print(f"accuracy crossover: refined becomes more accurate than classic at "
              f"about {cross:.1f} deg rotation SD")
    else:
        print("no crossover within the range tested")
    df.to_csv(HERE / f"accuracy_vs_rotation_{args.model}.csv", index=False)
    print(f"Wrote accuracy_vs_rotation_{args.model}.csv")


if __name__ == "__main__":
    main()
