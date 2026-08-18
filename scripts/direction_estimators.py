"""
Which direction estimator incurs the least variance?

The refined index pays about 20 percent extra within-participant variance for
estimating its measurement axes. Replacing the sequential sign-flip mean with
the dyadic principal eigenvector recovers only a small part of that, so this
sweeps a wider set of estimators on the same cached ROI tensors.

All are evaluated on HCP-A, whose diffusion data is already AC-PC aligned, so
there is essentially no orientation bias left and any difference in
within-participant variance is close to pure estimator noise.

  vecmean     FA-weighted vector mean with sequential sign alignment. The
              method described in the submitted manuscript.
  dyadic      Principal eigenvector of the FA-weighted dyadic sum. The Watson
              maximum-likelihood direction, sign-invariant by construction.
  dyadic_cl   As dyadic, weighted by Westin linear anisotropy
              CL = (l1 - l2) / l1 instead of FA. CL measures how well defined a
              single direction is in that voxel, which is what the estimate
              actually depends on, whereas FA is also raised by planar
              anisotropy where the principal direction is poorly determined.
  dyadic_fa2  As dyadic, weighted by FA squared, concentrating weight on the
              most anisotropic voxels.
  trimmed     Two-pass dyadic: estimate, discard voxels whose principal
              direction lies more than 30 degrees off it, re-estimate. Aimed at
              crossing-fibre and partial-volume voxels.
  pop_shrunk  Dyadic shrunk toward the cohort mean direction rather than toward
              the scanner axis, by an empirical-Bayes weight from the voxel
              count and dispersion. Shrinking toward the population anatomy
              keeps the correction while damping per-session noise.
  subject     One axis per participant, estimated from all of that
              participant's visits pooled. Legitimate for longitudinal designs
              because anatomy is fixed within a participant, and it removes
              per-session estimation noise from the axis entirely.

Usage:
    python direction_estimators.py
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import atomic_io  # noqa: F401  writes become atomic on import

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from estimator_variants import (CACHE, dir_running_mean, dyadic_tensor,
                                directional_diffusivity, variance_components)

HERE = Path(__file__).resolve().parent
TRIM_DEG = 30.0

X = np.array([1.0, 0.0, 0.0])
Y = np.array([0.0, 1.0, 0.0])
Z = np.array([0.0, 0.0, 1.0])


def weights_for(kind: str, roi: dict) -> np.ndarray:
    fa = roi["fa"].astype(float)
    if kind == "fa":
        return fa
    if kind == "fa2":
        return fa**2
    if kind == "cl":
        ev = np.sort(roi["evals"].astype(float), axis=1)[:, ::-1]
        l1 = ev[:, 0]
        cl = np.divide(l1 - ev[:, 1], l1, out=np.zeros_like(l1), where=l1 > 0)
        return np.clip(cl, 0, None)
    raise ValueError(kind)


def principal(v1: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.linalg.eigh(dyadic_tensor(v1, w))[1][:, -1]


def trimmed_dir(v1: np.ndarray, w: np.ndarray) -> np.ndarray:
    v = principal(v1, w)
    cosang = np.abs(v1 @ v)
    keep = cosang >= np.cos(np.radians(TRIM_DEG))
    if keep.sum() >= 4:
        v = principal(v1[keep], w[keep])
    return v


def estimate(roi: dict, method: str) -> np.ndarray:
    if method == "vecmean":
        return dir_running_mean(roi["v1"], roi["fa"].astype(float))
    if method == "dyadic":
        return principal(roi["v1"], weights_for("fa", roi))
    if method == "dyadic_cl":
        return principal(roi["v1"], weights_for("cl", roi))
    if method == "dyadic_fa2":
        return principal(roi["v1"], weights_for("fa2", roi))
    if method in ("trimmed", "pop_shrunk", "subject"):
        return trimmed_dir(roi["v1"], weights_for("fa", roi)) if method == "trimmed" \
            else principal(roi["v1"], weights_for("fa", roi))
    raise ValueError(method)


def align(v: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return v if np.dot(v, ref) >= 0 else -v


def alps(proj, assoc, vp, va):
    p = np.cross(vp, va)
    n = np.linalg.norm(p)
    p = p / n if n > 1e-10 else X

    def orth(a, b):
        c = np.cross(a, b)
        m = np.linalg.norm(c)
        return c / m if m > 1e-10 else Y

    op, oa = orth(p, vp), orth(p, va)
    num = (directional_diffusivity(proj["evals"], proj["evecs"], p)
           + directional_diffusivity(assoc["evals"], assoc["evecs"], p))
    den = (directional_diffusivity(proj["evals"], proj["evecs"], op)
           + directional_diffusivity(assoc["evals"], assoc["evecs"], oa))
    return num / den if den else np.nan


def main() -> None:
    sessions = []
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
            sessions.append(rec)

    counts = pd.Series([s["Subject_ID"] for s in sessions]).value_counts()
    keep = set(counts[counts >= 2].index)
    sessions = [s for s in sessions if s["Subject_ID"] in keep]
    print(f"sessions {len(sessions)}, participants {len({s['Subject_ID'] for s in sessions})}\n")

    methods = ["vecmean", "dyadic", "dyadic_cl", "dyadic_fa2", "trimmed",
               "pop_shrunk", "subject"]

    # Per-session, per-hemisphere direction estimates for the plain estimators.
    est = defaultdict(dict)
    for s in sessions:
        for hemi, (proj, assoc) in s["hemis"].items():
            for m in ("vecmean", "dyadic", "dyadic_cl", "dyadic_fa2", "trimmed"):
                est[(s["Subject_ID"], s["Visit"], hemi)][m] = (
                    align(estimate(proj, m), Z), align(estimate(assoc, m), Y))

    # Cohort mean direction per hemisphere, for the shrinkage prior.
    pop = {}
    for hemi in ("L", "R"):
        vp = np.array([est[k]["dyadic"][0] for k in est if k[2] == hemi])
        va = np.array([est[k]["dyadic"][1] for k in est if k[2] == hemi])
        pop[hemi] = (principal(vp, np.ones(len(vp))), principal(va, np.ones(len(va))))
        pop[hemi] = (align(pop[hemi][0], Z), align(pop[hemi][1], Y))

    # Participant-level axis pooled over that participant's visits.
    per_sub = defaultdict(lambda: defaultdict(list))
    for s in sessions:
        for hemi in s["hemis"]:
            k = (s["Subject_ID"], s["Visit"], hemi)
            per_sub[(s["Subject_ID"], hemi)]["p"].append(est[k]["dyadic"][0])
            per_sub[(s["Subject_ID"], hemi)]["a"].append(est[k]["dyadic"][1])
    sub_axis = {}
    for key, dd in per_sub.items():
        vp = np.array(dd["p"])
        va = np.array(dd["a"])
        sub_axis[key] = (align(principal(vp, np.ones(len(vp))), Z),
                         align(principal(va, np.ones(len(va))), Y))

    rows = []
    for s in sessions:
        rec = {"Subject_ID": s["Subject_ID"], "Visit": s["Visit"]}
        cls = []
        for hemi, (proj, assoc) in s["hemis"].items():
            cls.append((directional_diffusivity(proj["evals"], proj["evecs"], X)
                        + directional_diffusivity(assoc["evals"], assoc["evecs"], X))
                       / (directional_diffusivity(proj["evals"], proj["evecs"], Y)
                          + directional_diffusivity(assoc["evals"], assoc["evecs"], Z)))
        rec["classic"] = float(np.mean(cls))
        for m in methods:
            vals = []
            for hemi, (proj, assoc) in s["hemis"].items():
                k = (s["Subject_ID"], s["Visit"], hemi)
                if m == "subject":
                    vp, va = sub_axis[(s["Subject_ID"], hemi)]
                elif m == "pop_shrunk":
                    vp0, va0 = est[k]["dyadic"]
                    n = len(s["hemis"][hemi][0]["v1"])
                    lam = n / (n + 60.0)     # empirical-Bayes weight on the data
                    pp, pa = pop[hemi]
                    vp = lam * vp0 + (1 - lam) * pp
                    va = lam * va0 + (1 - lam) * pa
                    vp /= np.linalg.norm(vp)
                    va /= np.linalg.norm(va)
                else:
                    vp, va = est[k][m]
                vals.append(alps(proj, assoc, vp, va))
            rec[m] = float(np.mean(vals))
        rows.append(rec)

    d = pd.DataFrame(rows)
    base = variance_components(d, "classic")
    print(f"{'estimator':<12s} {'ICC':>7s} {'var_within':>12s} {'wCV %':>7s} "
          f"{'vs classic':>11s} {'vs vecmean':>11s}")
    print(f"{'classic':<12s} {base['icc']:7.3f} {base['var_within']:12.6f} "
          f"{base['wcv_pct']:7.2f} {'reference':>11s}")
    vm = variance_components(d, "vecmean")["var_within"]
    for m in methods:
        vc = variance_components(d, m)
        print(f"{m:<12s} {vc['icc']:7.3f} {vc['var_within']:12.6f} {vc['wcv_pct']:7.2f} "
              f"{vc['var_within']/base['var_within']:10.2f}x "
              f"{vc['var_within']/vm:10.3f}x")

    d.to_csv(HERE / "direction_estimators.csv", index=False)
    print(f"\nWrote {HERE/'direction_estimators.csv'}")


if __name__ == "__main__":
    main()
