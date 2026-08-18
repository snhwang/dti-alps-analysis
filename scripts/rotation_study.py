"""
Controlled head-rotation experiment on HCP-A.

HCP-A diffusion data is already AC-PC aligned, with an axis-aligned image
affine, so it provides something no natively oblique cohort can: a population
with head orientation effectively removed. Imposing known rotations on it gives
a reference value that stays valid by construction, because rotating a head
cannot change its perivascular diffusivity. Every method can then be scored
against the same target, which is a stronger basis for a head-to-head than a
reliability comparison with no ground truth.

Three analyses:

  A  Accuracy and reliability against rotation, for every ALPS variant on
     identical ROIs and voxels. Accuracy is absolute error against the
     unrotated classic value, which grants the classic index its own best case.
     Reliability is within-participant variance when each visit of a
     participant receives its own independent rotation, which is what varying
     head positioning between sessions actually looks like.

  B  Per-axis breakdown. Classic ALPS assumes projection fibres lie along z,
     association fibres along y and perivascular spaces along x, so the three
     rotation axes are not interchangeable. Rotation about x moves both of the
     axes that form the denominator. Burles et al. found pitch was both the
     largest real-world rotation and the one that correlated with the index.

  C  Group-difference simulation. If two groups are positioned differently, an
     orientation-dependent index reports a difference that is not there. This
     quantifies the apparent group effect produced by a given positioning
     difference, against the 5 to 20 percent disease effects reported in the
     literature. This is the failure mode that matters, because unlike
     variance it does not diminish with sample size.

Registration-based reorientation is not simulated separately. With a known
rotation it would be exact by construction, so its error is simply the classic
curve evaluated at the residual registration error rather than at the imposed
rotation, which can be read directly off analysis A.

This is a coordinate-space simulation. It does not reproduce acquisition-stage
partial-volume or in-plane averaging effects, so it calibrates thresholds
rather than demonstrating empirical robustness.

Usage:
    python rotation_study.py --limit 400 --repeats 8
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
from estimator_variants import directional_diffusivity, variance_components
from direction_estimators import weights_for, principal, align, X, Y, Z
from rotation_dose_response import euler_rotation, rotate_roi, load_sessions

HERE = Path(__file__).resolve().parent
SIGMAS = [0, 2, 5, 8, 10, 12, 15, 20, 25, 30]


# ---------------------------------------------------------------------------
# ALPS variants, all sharing ROIs, voxels and the diffusivity calculation
# ---------------------------------------------------------------------------


def _dd(roi, u):
    return directional_diffusivity(roi["evals"], roi["evecs"], u)


def m_classic(proj, assoc):
    return (_dd(proj, X) + _dd(assoc, X)) / (_dd(proj, Y) + _dd(assoc, Z))


def _axes(proj, assoc, weight="cl"):
    vp = align(principal(proj["v1"], weights_for(weight, proj)), Z)
    va = align(principal(assoc["v1"], weights_for(weight, assoc)), Y)
    p = np.cross(vp, va)
    p = p / max(np.linalg.norm(p), 1e-12)
    return vp, va, p


def m_refined(proj, assoc):
    vp, va, p = _axes(proj, assoc)
    op = np.cross(p, vp); op /= max(np.linalg.norm(op), 1e-12)
    oa = np.cross(p, va); oa /= max(np.linalg.norm(oa), 1e-12)
    return (_dd(proj, p) + _dd(assoc, p)) / (_dd(proj, op) + _dd(assoc, oa))


def m_refined_plus(proj, assoc):
    """PVS axis projected onto each voxel's transverse plane, then pooled."""
    vp, va, p = _axes(proj, assoc)
    acc, wts = [], []
    for roi in (proj, assoc):
        v1 = roi["v1"]
        proj_p = p - (v1 @ p)[:, None] * v1
        n = np.linalg.norm(proj_p, axis=1, keepdims=True)
        good = n[:, 0] > 1e-8
        acc.append(proj_p[good] / n[good])
        wts.append(roi["fa"][good])
    allv = np.vstack(acc)
    allw = np.concatenate(wts)
    pp = align(principal(allv, allw), p)
    op = np.cross(pp, vp); op /= max(np.linalg.norm(op), 1e-12)
    oa = np.cross(pp, va); oa /= max(np.linalg.norm(oa), 1e-12)
    return (_dd(proj, pp) + _dd(assoc, pp)) / (_dd(proj, op) + _dd(assoc, oa))


def m_alps_pas(proj, assoc):
    """Eigenvalues 2 and 3 sorted by the x-alignment of their eigenvectors."""
    num, den = [], []
    for roi in (proj, assoc):
        ev, vc = roi["evals"], roi["evecs"]
        order = np.argsort(ev, axis=1)[:, ::-1]
        l2 = np.take_along_axis(ev, order[:, 1:2], 1)[:, 0]
        l3 = np.take_along_axis(ev, order[:, 2:3], 1)[:, 0]
        v2x = np.abs(vc[np.arange(len(vc)), 0, order[:, 1]])
        v3x = np.abs(vc[np.arange(len(vc)), 0, order[:, 2]])
        pick2 = v2x > v3x
        num.append(np.where(pick2, l2, l3).mean())
        den.append(np.where(pick2, l3, l2).mean())
    return (num[0] + num[1]) / (den[0] + den[1])


def m_per_voxel(proj, assoc):
    """Each voxel's own principal direction crossed with the other ROI's mean."""
    vp, va, _ = _axes(proj, assoc)
    num, den = [], []
    for roi, other in ((proj, va), (assoc, vp)):
        v1 = roi["v1"]
        p = np.cross(v1, other)
        n = np.linalg.norm(p, axis=1, keepdims=True)
        good = n[:, 0] > 1e-8
        if not good.any():
            continue
        p = p[good] / n[good]
        o = np.cross(p, v1[good])
        o /= np.maximum(np.linalg.norm(o, axis=1, keepdims=True), 1e-12)
        ev, vc = roi["evals"][good], roi["evecs"][good]
        dp = np.einsum("nkj,nj->nk", np.transpose(vc, (0, 2, 1)), p)
        do = np.einsum("nkj,nj->nk", np.transpose(vc, (0, 2, 1)), o)
        num.append((ev * dp**2).sum(axis=1).mean())
        den.append((ev * do**2).sum(axis=1).mean())
    if not num:
        return np.nan
    return float(np.sum(num) / np.sum(den))


METHODS = {
    "classic": m_classic,
    "refined": m_refined,
    "refined+": m_refined_plus,
    "ALPS-PAS": m_alps_pas,
    "per-voxel": m_per_voxel,
}


def evaluate(s, fn, R=None):
    vals = []
    for proj, assoc in s["hemis"].values():
        if R is not None:
            proj, assoc = rotate_roi(proj, R), rotate_roi(assoc, R)
        vals.append(fn(proj, assoc))
    return float(np.mean(vals))


# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--repeats", type=int, default=8)
    args = ap.parse_args()

    sessions = load_sessions()[: args.limit]
    subs = pd.Series([s["Subject_ID"] for s in sessions]).value_counts()
    print(f"sessions {len(sessions)}, participants {len(subs)}\n")

    ref = {m: np.array([evaluate(s, fn) for s in sessions])
           for m, fn in METHODS.items()}
    truth = ref["classic"]
    for m in METHODS:
        off = 100 * np.abs(ref[m] - truth) / truth
        print(f"  {m:<10s} offset from reference at 0 deg: {off.mean():5.2f}%")

    rng = np.random.default_rng(20260728)

    # ---- A. accuracy against rotation -------------------------------------
    print("\nA. ACCURACY  (mean |error| vs unrotated classic, %)")
    print(f"{'sigma':>6s} " + " ".join(f"{m:>10s}" for m in METHODS))
    accA = []
    for sig in SIGMAS:
        errs = {m: [] for m in METHODS}
        for _ in range(args.repeats if sig else 1):
            for i, s in enumerate(sessions):
                R = np.eye(3) if sig == 0 else euler_rotation(*(rng.normal(0, 1, 3) * sig))
                for m, fn in METHODS.items():
                    errs[m].append(100 * abs(evaluate(s, fn, R) - truth[i]) / truth[i])
        row = {"sigma": sig, **{m: float(np.mean(errs[m])) for m in METHODS}}
        accA.append(row)
        print(f"{sig:>4}deg " + " ".join(f"{row[m]:10.3f}" for m in METHODS))
    pd.DataFrame(accA).to_csv(HERE / "rotation_study_accuracy.csv", index=False)

    # ---- B. per-axis ------------------------------------------------------
    print("\nB. PER-AXIS  (mean |error| %, single-axis rotation at 15 deg)")
    print(f"{'axis':>10s} " + " ".join(f"{m:>10s}" for m in METHODS))
    rowsB = []
    for axis, name in ((0, "pitch (x)"), (1, "roll (y)"), (2, "yaw (z)")):
        errs = {m: [] for m in METHODS}
        for _ in range(args.repeats):
            for i, s in enumerate(sessions):
                ang = np.zeros(3)
                ang[axis] = rng.normal(0, 15)
                R = euler_rotation(*ang)
                for m, fn in METHODS.items():
                    errs[m].append(100 * abs(evaluate(s, fn, R) - truth[i]) / truth[i])
        row = {"axis": name, **{m: float(np.mean(errs[m])) for m in METHODS}}
        rowsB.append(row)
        print(f"{name:>10s} " + " ".join(f"{row[m]:10.3f}" for m in METHODS))
    pd.DataFrame(rowsB).to_csv(HERE / "rotation_study_peraxis.csv", index=False)

    # ---- C. spurious group difference -------------------------------------
    print("\nC. SPURIOUS GROUP DIFFERENCE")
    print("   one group tilted, the other not; apparent group difference (%)")
    print(f"{'tilt':>6s} " + " ".join(f"{m:>10s}" for m in METHODS))
    # Both arms use the same participants, one arm tilted, so the true group
    # difference is exactly zero and anything reported is artefact.
    rowsC = []
    for tilt in (2, 5, 8, 10, 15, 20):
        R = euler_rotation(tilt, 0, 0)
        row = {"tilt_deg": tilt}
        for m, fn in METHODS.items():
            a = np.array([evaluate(s, fn) for s in sessions])
            b = np.array([evaluate(s, fn, R) for s in sessions])
            row[m] = float(100 * (b.mean() - a.mean()) / a.mean())
        rowsC.append(row)
        print(f"{tilt:>4}deg " + " ".join(f"{row[m]:+10.3f}" for m in METHODS))
    pd.DataFrame(rowsC).to_csv(HERE / "rotation_study_group.csv", index=False)

    print("\nReported disease-related ALPS differences are typically 5 to 20 percent.")
    print("Wrote rotation_study_{accuracy,peraxis,group}.csv")


if __name__ == "__main__":
    main()
