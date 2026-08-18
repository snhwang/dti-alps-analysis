"""
The measured-perpendicular axis, evaluated on the hand-drawn regions.

Everything so far has used atlas-placed regions. Those are reproducible but they
spill past the tract, and a human rater does not put a region where the atlas
does. Running the same comparison on the hand-drawn regions asks whether the
measured-v2 axis is doing something real or is an artefact of automated
placement.

Region file: alps_rois_manual.nii.gz, value 1 projection, value 2 association,
bilateral, in native diffusion space. This file reproduces the published
Traditional_Avg values exactly, which alps_rois.nii.gz does not, so it is the
authoritative one.

session_20260122_160723 is excluded. Its mask was overwritten early in this
project and neither surviving copy reproduces its stored value, so the region is
unrecoverable even though the published number survives in the results table.

Direction estimation still uses the JHU tract band, which is warped into the same
native space, so only the measurement region differs from the automated run.

Usage:
    python manual_pvs_axis.py
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from data_paths import winpath

import atomic_io  # noqa: F401  writes become atomic on import
from scipy import stats

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from estimator_variants import directional_diffusivity as dd, variance_components
from direction_estimators import weights_for, principal, align, X, Y, Z
from alps_common import parse_age

HERE = Path(__file__).resolve().parent
DIFF = HERE.parent.parent / "diffusion"
OUT = winpath("Q:/dti_output")
SLAB_MM, FA_MIN = 8.0, 0.2
VARIANTS = ["classic", "cross", "v2_roi", "v2_slab",
            "auto_classic", "auto_cross", "auto_v2_roi", "auto_v2_slab"]
EXCLUDE = {"session_20260122_160723"}   # mask destroyed, see module docstring


def acute(u, v):
    return float(np.degrees(np.arccos(np.clip(abs(float(np.dot(u, v))), 0, 1))))


def main() -> None:
    import nibabel as nib

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    src = pd.read_csv(DIFF / "HCP" / "lifespan_alps_results.csv")
    src["Age"] = parse_age(src["Age"])
    src = src.dropna(subset=["Age"])
    src = src[~src.DTI_Session_ID.isin(EXCLUDE)]
    if args.limit:
        src = src.head(args.limit)
    print(f"{len(src)} hand-drawn sessions, {src.Subject_ID.nunique()} participants "
          f"({len(EXCLUDE)} excluded for a damaged mask)\n")

    rows, skipped = [], 0
    for i, r in enumerate(src.itertuples(), 1):
        sd = OUT / r.DTI_Session_ID / "processed"
        roi_p = sd / "alps_rois_manual.nii.gz"
        lab_p = sd / "atlas" / "jhu_labels_registered.nii.gz"
        sph_p = sd / "atlas" / "sphere_roi" / "sphere_roi_combined.nii.gz"
        if not (roi_p.exists() and lab_p.exists() and sph_p.exists()):
            skipped += 1
            continue
        try:
            rimg = nib.load(str(roi_p)); roi = rimg.get_fdata().astype(int)
            lab = nib.load(str(lab_p)).get_fdata().astype(int)
            sph = nib.load(str(sph_p)).get_fdata().astype(int)
            ev = nib.load(str(sd / "tensor_eigenvalues.nii.gz")).get_fdata()
            vc = nib.load(str(sd / "tensor_eigenvectors.nii.gz")).get_fdata()
        except Exception:
            skipped += 1
            continue
        if roi.shape != lab.shape or roi.shape != ev.shape[:3]:
            skipped += 1
            continue

        srt = np.argsort(ev, axis=-1)[..., ::-1]
        l1 = np.take_along_axis(ev, srt[..., 0:1], -1)[..., 0]
        l2 = np.take_along_axis(ev, srt[..., 1:2], -1)[..., 0]
        l3 = np.take_along_axis(ev, srt[..., 2:3], -1)[..., 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            CP = np.where(l1 > 0, (l2 - l3) / l1, 0.0)
        md = ev.mean(-1)
        nu = np.sqrt(((ev - md[..., None]) ** 2).sum(-1))
        de = np.sqrt((ev ** 2).sum(-1))
        fa = np.clip(np.sqrt(1.5) * np.divide(nu, de, out=np.zeros_like(nu), where=de != 0), 0, 1)

        ii, jj, kk = np.indices(roi.shape)
        Af = rimg.affine
        xw = Af[0, 0] * ii + Af[0, 1] * jj + Af[0, 2] * kk + Af[0, 3]
        zw = Af[2, 0] * ii + Af[2, 1] * jj + Af[2, 2] * kk + Af[2, 3]

        def evec(m, w):
            V = vc[m]; o = srt[m]
            v = np.take_along_axis(V, o[:, None, w:w + 1], 2)[:, :, 0]
            return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)

        acc = {k: [] for k in VARIANTS}
        ang = {k: [] for k in ("v2_to_x", "v2_to_cross", "cross_to_x")}
        for side, scr, slf in ((xw < 0, 26, 42), (xw > 0, 25, 41)):
            mp = (roi == 1) & side & (fa >= FA_MIN)
            ma = (roi == 2) & side & (fa >= FA_MIN)
            if mp.sum() < 4 or ma.sum() < 4:
                continue
            z0 = float(np.median(zw[roi > 0])) if (roi > 0).any() else 0.0
            band = np.abs(zw - z0) <= SLAB_MM
            bp = (lab == scr) & (fa >= FA_MIN) & band
            ba = (lab == slf) & (fa >= FA_MIN) & band
            if bp.sum() < 10 or ba.sum() < 10:
                continue

            vp = align(principal(evec(bp, 0), weights_for("cl", {"fa": fa[bp], "evals": ev[bp]})), Z)
            va = align(principal(evec(ba, 0), weights_for("cl", {"fa": fa[ba], "evals": ev[ba]})), Y)
            p_cross = np.cross(vp, va); p_cross /= max(np.linalg.norm(p_cross), 1e-12)

            def v2_axis(masks):
                v2 = np.vstack([evec(m, 1) for m in masks])
                w = np.concatenate([CP[m] for m in masks])
                ok = w > 0
                if ok.sum() < 6:
                    return None
                a = principal(v2[ok], w[ok])
                return align(a / max(np.linalg.norm(a), 1e-12), X)

            p_roi = v2_axis([mp, ma])
            p_slab = v2_axis([bp, ba])
            if p_roi is None or p_slab is None:
                continue

            def alps(p):
                op = np.cross(p, vp); op /= max(np.linalg.norm(op), 1e-12)
                oa = np.cross(p, va); oa /= max(np.linalg.norm(oa), 1e-12)
                return ((dd(ev[mp], vc[mp], p) + dd(ev[ma], vc[ma], p))
                        / (dd(ev[mp], vc[mp], op) + dd(ev[ma], vc[ma], oa)))

            acc["classic"].append((dd(ev[mp], vc[mp], X) + dd(ev[ma], vc[ma], X))
                                  / (dd(ev[mp], vc[mp], Y) + dd(ev[ma], vc[ma], Z)))
            acc["cross"].append(alps(p_cross))
            acc["v2_roi"].append(alps(p_roi))
            acc["v2_slab"].append(alps(p_slab))

            # same sessions, same directions, atlas-placed measurement region
            amp = (sph == 1) & side & (fa >= FA_MIN)
            ama = (sph == 2) & side & (fa >= FA_MIN)
            if amp.sum() >= 4 and ama.sum() >= 4:
                ap_roi = v2_axis([amp, ama])

                def aalps(p):
                    op = np.cross(p, vp); op /= max(np.linalg.norm(op), 1e-12)
                    oa = np.cross(p, va); oa /= max(np.linalg.norm(oa), 1e-12)
                    return ((dd(ev[amp], vc[amp], p) + dd(ev[ama], vc[ama], p))
                            / (dd(ev[amp], vc[amp], op) + dd(ev[ama], vc[ama], oa)))

                acc["auto_classic"].append(
                    (dd(ev[amp], vc[amp], X) + dd(ev[ama], vc[ama], X))
                    / (dd(ev[amp], vc[amp], Y) + dd(ev[ama], vc[ama], Z)))
                acc["auto_cross"].append(aalps(p_cross))
                if ap_roi is not None:
                    acc["auto_v2_roi"].append(aalps(ap_roi))
                acc["auto_v2_slab"].append(aalps(p_slab))
            ang["v2_to_x"].append(acute(p_slab, X))
            ang["v2_to_cross"].append(acute(p_slab, p_cross))
            ang["cross_to_x"].append(acute(p_cross, X))

        if not acc["classic"]:
            skipped += 1
            continue
        rec = {"Subject_ID": r.Subject_ID, "Session": r.Session, "Age": r.Age,
               "stored_classic": pd.to_numeric(r.Traditional_Avg, errors="coerce")}
        for k, v in {**acc, **ang}.items():
            if v:
                rec[k] = float(np.mean(v))
        rows.append(rec)

    d = pd.DataFrame(rows)
    d.to_csv(HERE / "manual_pvs_axis.csv", index=False)
    print(f"{len(d)} usable sessions, {skipped} skipped, "
          f"{d.Subject_ID.nunique()} participants")

    ok = d.dropna(subset=["stored_classic", "classic"])
    if len(ok) > 3:
        err = 100 * (ok.classic - ok.stored_classic).abs() / ok.stored_classic
        print(f"reproduction of published manual values: median |error| {err.median():.3f}%"
              f"  max {err.max():.3f}%\n")

    print("axis geometry (degrees, mean)")
    for k in ("v2_to_x", "v2_to_cross", "cross_to_x"):
        if k in d:
            print(f"  {k:<14s} {d[k].mean():5.1f}")

    counts = d.Subject_ID.value_counts()
    lon = d[d.Subject_ID.isin(counts[counts >= 2].index)]
    print(f"\nlongitudinal subset: {len(lon)} sessions, {lon.Subject_ID.nunique()} participants")
    base = variance_components(lon.dropna(subset=["classic"]), "classic") if len(lon) > 10 else None
    print(f"{'variant':<12s} {'ICC':>7s} {'var/classic':>12s} {'r age':>8s} {'p':>10s}")
    for k in VARIANTS:
        if k not in d:
            continue
        ds = d.dropna(subset=[k])
        r, pv = stats.pearsonr(ds.Age, ds[k])
        if base is not None and len(lon.dropna(subset=[k])) > 10:
            vc = variance_components(lon.dropna(subset=[k]), k)
            print(f"{k:<12s} {vc['icc']:7.3f} {vc['var_within']/base['var_within']:12.2f} "
                  f"{r:8.3f} {pv:10.2e}")
        else:
            print(f"{k:<12s} {'--':>7s} {'--':>12s} {r:8.3f} {pv:10.2e}")


if __name__ == "__main__":
    main()
