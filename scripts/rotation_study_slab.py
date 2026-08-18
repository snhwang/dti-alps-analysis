"""
Rotation experiments, recomputed with slab-derived directions.

Everything previously reported estimated the tract directions from the 5 mm
spheres themselves. Those sit 12 mm apart, so each contains tissue belonging to
the other tract, which pulled the two estimates together and produced an
apparent inter-fibre angle of 67 degrees where the anatomical labels give 83 to
86. The method now takes directions from the tract label in an 8 mm axial band
and measures in the sphere, so these results are recomputed on that basis.

Reported: accuracy against known rotation for every variant, error by rotation
axis, and the group difference produced purely by a positioning offset.

The reference is the unrotated classic value, which grants the classic index its
own best case, since its error is zero by definition at zero rotation while
every corrected variant is scored on how far it departs from that answer.

Usage:
    python rotation_study_slab.py --repeats 6
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
from rotation_dose_response import euler_rotation

HERE = Path(__file__).resolve().parent
CACHE = HERE / "slab_cache_b1500"
SIGMAS = [0, 2, 5, 8, 10, 12, 15, 20, 25, 30]


def load():
    out = []
    for f in sorted(CACHE.glob("*.npz")):
        sub, visit = f.stem.rsplit("_", 1)
        z = np.load(f)
        rec = {"Subject_ID": sub, "Visit": visit,
               "Age": float(z["Age"][0]), "hemis": {}}
        ok = True
        for hemi in ("L", "R"):
            try:
                d = {}
                for tag in ("sph", "slab"):
                    for nm in ("proj", "assoc"):
                        k = f"{tag}_{nm}_{hemi}"
                        d[f"{tag}_{nm}"] = {
                            "v1": z[f"{k}_v1"].astype(np.float64),
                            "fa": z[f"{k}_fa"].astype(np.float64),
                            "evals": z[f"{k}_evals"].astype(np.float64),
                        }
                        if tag == "sph":
                            d[f"{tag}_{nm}"]["evecs"] = z[f"{k}_evecs"].astype(np.float64)
            except KeyError:
                ok = False
                break
            rec["hemis"][hemi] = d
        if ok:
            out.append(rec)
    return out


def rot(d, R, with_evecs):
    o = {"v1": d["v1"] @ R.T, "fa": d["fa"], "evals": d["evals"]}
    if with_evecs:
        o["evecs"] = np.einsum("ij,njk->nik", R, d["evecs"])
    return o


def _dd(roi, u):
    return directional_diffusivity(roi["evals"], roi["evecs"], u)


def variants(h, R=None):
    """All ALPS variants for one hemisphere; directions from the slab."""
    sp = rot(h["sph_proj"], R, True) if R is not None else h["sph_proj"]
    sa = rot(h["sph_assoc"], R, True) if R is not None else h["sph_assoc"]
    lp = rot(h["slab_proj"], R, False) if R is not None else h["slab_proj"]
    la = rot(h["slab_assoc"], R, False) if R is not None else h["slab_assoc"]

    out = {"classic": (_dd(sp, X) + _dd(sa, X)) / (_dd(sp, Y) + _dd(sa, Z))}

    vp = align(principal(lp["v1"], weights_for("cl", lp)), Z)
    va = align(principal(la["v1"], weights_for("cl", la)), Y)
    p = np.cross(vp, va); p /= max(np.linalg.norm(p), 1e-12)
    op = np.cross(p, vp); op /= max(np.linalg.norm(op), 1e-12)
    oa = np.cross(p, va); oa /= max(np.linalg.norm(oa), 1e-12)
    out["refined"] = (_dd(sp, p) + _dd(sa, p)) / (_dd(sp, op) + _dd(sa, oa))
    out["angle"] = float(np.degrees(np.arccos(np.clip(abs(vp @ va), 0, 1))))

    # Refined+ : the slab-derived PVS axis projected onto the transverse plane of
    # each measurement voxel, then pooled. The axes still come from the slab; only
    # the per-voxel perpendicularity correction uses the sphere.
    acc, wts = [], []
    for roi in (sp, sa):
        v1 = roi["v1"]
        pr = p - (v1 @ p)[:, None] * v1
        n = np.linalg.norm(pr, axis=1, keepdims=True)
        good = n[:, 0] > 1e-8
        acc.append(pr[good] / n[good])
        wts.append(roi["fa"][good])
    if sum(len(a) for a in acc):
        pp = align(principal(np.vstack(acc), np.concatenate(wts)), p)
        opp = np.cross(pp, vp); opp /= max(np.linalg.norm(opp), 1e-12)
        oap = np.cross(pp, va); oap /= max(np.linalg.norm(oap), 1e-12)
        out["refined+"] = (_dd(sp, pp) + _dd(sa, pp)) / (_dd(sp, opp) + _dd(sa, oap))
    else:
        out["refined+"] = np.nan

    # Per-voxel: each measurement voxel's own principal direction crossed with the
    # opposite tract's slab-derived mean direction.
    num, den = [], []
    for roi, other in ((sp, va), (sa, vp)):
        v1 = roi["v1"]
        pv = np.cross(v1, other)
        n = np.linalg.norm(pv, axis=1, keepdims=True)
        good = n[:, 0] > 1e-8
        if not good.any():
            continue
        pv = pv[good] / n[good]
        ov = np.cross(pv, v1[good])
        ov /= np.maximum(np.linalg.norm(ov, axis=1, keepdims=True), 1e-12)
        ev, vc = roi["evals"][good], roi["evecs"][good]
        dp = np.einsum("nkj,nj->nk", np.transpose(vc, (0, 2, 1)), pv)
        do = np.einsum("nkj,nj->nk", np.transpose(vc, (0, 2, 1)), ov)
        num.append((ev * dp ** 2).sum(axis=1).mean())
        den.append((ev * do ** 2).sum(axis=1).mean())
    out["per-voxel"] = float(np.sum(num) / np.sum(den)) if num else np.nan

    # ALPS-PAS, unchanged: it uses no estimated axes, only the scanner x-component
    num, den = [], []
    for roi in (sp, sa):
        ev, vc = roi["evals"], roi["evecs"]
        order = np.argsort(ev, axis=1)[:, ::-1]
        idx = np.arange(len(ev))
        l2, l3 = ev[idx, order[:, 1]], ev[idx, order[:, 2]]
        v2x = np.abs(np.array([vc[i][0, order[i, 1]] for i in idx]))
        v3x = np.abs(np.array([vc[i][0, order[i, 2]] for i in idx]))
        pick = v2x > v3x
        num.append(np.where(pick, l2, l3).mean())
        den.append(np.where(pick, l3, l2).mean())
    out["ALPS-PAS"] = (num[0] + num[1]) / (den[0] + den[1])
    return out


def evaluate(s, R=None):
    vals = [variants(h, R) for h in s["hemis"].values()]
    return {k: float(np.mean([v[k] for v in vals])) for k in vals[0]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=6)
    args = ap.parse_args()

    sessions = load()
    print(f"{len(sessions)} sessions, "
          f"{len({s['Subject_ID'] for s in sessions})} participants\n")

    base = [evaluate(s) for s in sessions]
    truth = np.array([b["classic"] for b in base])
    ang = np.array([b["angle"] for b in base])
    METHODS = ["classic", "refined", "refined+", "ALPS-PAS", "per-voxel"]
    print(f"inter-fibre angle, slab-derived: median {np.median(ang):.1f} deg "
          f"(spheres gave 67.5)")
    for m in METHODS:
        off = 100 * np.abs(np.array([b[m] for b in base]) - truth) / truth
        print(f"  {m:<10s} offset from reference at 0 deg: {off.mean():5.2f}%")

    rng = np.random.default_rng(20260802)
    print(f"\nACCURACY  (mean |error| vs unrotated classic, %)")
    print(f"{'sigma':>6s} " + " ".join(f"{m:>10s}" for m in METHODS))
    rows = []
    for sig in SIGMAS:
        errs = {m: [] for m in METHODS}
        for _ in range(args.repeats if sig else 1):
            for i, s in enumerate(sessions):
                R = np.eye(3) if sig == 0 else euler_rotation(*(rng.normal(0, 1, 3) * sig))
                v = evaluate(s, R)
                for m in METHODS:
                    errs[m].append(100 * abs(v[m] - truth[i]) / truth[i])
        row = {"sigma": sig, **{m: float(np.mean(errs[m])) for m in METHODS}}
        rows.append(row)
        print(f"{sig:>4}deg " + " ".join(f"{row[m]:10.3f}" for m in METHODS))
    acc = pd.DataFrame(rows)
    acc.to_csv(HERE / "rotation_slab_accuracy.csv", index=False)

    cross = None
    for a, b in zip(acc.itertuples(), acc.iloc[1:].itertuples()):
        if a.classic <= a.refined and b.classic > b.refined:
            t = (a.refined - a.classic) / ((b.classic - a.classic) - (b.refined - a.refined))
            cross = a.sigma + t * (b.sigma - a.sigma)
            break
    print(f"\naccuracy crossover: {cross:.1f} deg" if cross else "\nno crossover")

    print(f"\nPER-AXIS  (mean |error| %, single-axis rotation at 15 deg)")
    print(f"{'axis':>10s} " + " ".join(f"{m:>10s}" for m in METHODS))
    rowsB = []
    for axis, nm in ((0, "pitch (x)"), (1, "roll (y)"), (2, "yaw (z)")):
        errs = {m: [] for m in METHODS}
        for _ in range(args.repeats):
            for i, s in enumerate(sessions):
                a3 = np.zeros(3); a3[axis] = rng.normal(0, 15)
                v = evaluate(s, euler_rotation(*a3))
                for m in METHODS:
                    errs[m].append(100 * abs(v[m] - truth[i]) / truth[i])
        row = {"axis": nm, **{m: float(np.mean(errs[m])) for m in METHODS}}
        rowsB.append(row)
        print(f"{nm:>10s} " + " ".join(f"{row[m]:10.3f}" for m in METHODS))
    pd.DataFrame(rowsB).to_csv(HERE / "rotation_slab_peraxis.csv", index=False)

    print(f"\nSPURIOUS GROUP DIFFERENCE  (same participants, one arm tilted)")
    print(f"{'tilt':>6s} " + " ".join(f"{m:>10s}" for m in METHODS))
    rowsC = []
    for tilt in (2, 5, 8, 10, 15, 20):
        R = euler_rotation(tilt, 0, 0)
        tv = [evaluate(s, R) for s in sessions]
        row = {"tilt_deg": tilt}
        for m in METHODS:
            a = np.array([b[m] for b in base]); b_ = np.array([t[m] for t in tv])
            row[m] = float(100 * (b_.mean() - a.mean()) / a.mean())
        rowsC.append(row)
        print(f"{tilt:>4}deg " + " ".join(f"{row[m]:+10.3f}" for m in METHODS))
    pd.DataFrame(rowsC).to_csv(HERE / "rotation_slab_group.csv", index=False)
    print("\nWrote rotation_slab_{accuracy,peraxis,group}.csv")


if __name__ == "__main__":
    main()
