"""
Can the orientation correction be done without magnifying estimation noise?

Classic ALPS evaluates along fixed axes, so it carries no estimation error at
all. The refined index estimates two tract directions per hemisphere and takes
their cross product, trading orientation bias for estimation variance. In HCP-A
that shows up directly: within-participant variance is about 18 percent higher
for Refined than for Classic.

HCP-A is the right testbed for this specific question. Its diffusion data is
already AC-PC aligned by the HCP pipeline, with a perfectly axis-aligned image
affine, so there is essentially no orientation bias left to remove. Any
difference in within-participant variance there is therefore close to pure
estimator noise, uncontaminated by the bias the correction exists to fix.

Estimators compared, all sharing the same ROIs, the same voxels and the same
directional-diffusivity calculation:

  current   FA-weighted vector mean with a sequential sign fix, then the PVS
            axis as the normalised cross product of the two tract directions.
            This is what the submitted manuscript describes.

  dyadic    Tract direction as the principal eigenvector of the FA-weighted
            dyadic sum, sum_i FA_i v_i v_i^T. Sign-invariant by construction
            and the maximum-likelihood direction under a Watson distribution,
            so it avoids the order dependence of a sequential sign fix.

  dyadic+   As dyadic, but the PVS axis is taken directly as the minor
            eigenvector of the combined SCR and SLF dyadic tensor rather than
            as a cross product. A cross product inherits error from both tract
            estimates and is ill-conditioned as the inter-fiber angle departs
            from 90 degrees, since the cross-product norm goes to zero.

  shrunk    As dyadic+, but each estimated axis is shrunk toward the scanner
            axis it replaces, by a factor derived from the concentration of the
            eigenvector distribution. Where a deviation is not resolvable above
            its own uncertainty the estimator falls back toward fixed axes, so
            no noise is injected when there is nothing to correct.

Pass 1 caches the small per-ROI tensor arrays so estimators can be re-tested
without re-reading the full eigenvector volumes.

Usage:
    python estimator_variants.py --extract --max-subjects 300
    python estimator_variants.py --compare
"""

from __future__ import annotations

import argparse
import glob
import os
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from data_paths import winpath

import atomic_io  # noqa: F401  writes become atomic on import

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
DIFF = HERE.parent.parent / "diffusion"
OUT = winpath("Q:/dti_output")
CACHE = HERE / os.environ.get("ALPS_CACHE_DIR", "roi_tensor_cache")
# "" uses the default multi-shell tensor; "_b1500" the conventional
# single-shell fit. HCP-A has no b=1000, so b=1500 is the closest to the
# DTI-ALPS convention available in that cohort.
SHELL = os.environ.get("ALPS_TENSOR_SUFFIX", "")


# ---------------------------------------------------------------------------
# Direction estimators
# ---------------------------------------------------------------------------


def dir_running_mean(v1: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Current method: sequential sign alignment, then FA-weighted vector mean."""
    if len(v1) == 1:
        return v1[0]
    aligned = np.zeros_like(v1)
    aligned[0] = v1[0]
    running = v1[0].copy()
    for i in range(1, len(v1)):
        aligned[i] = -v1[i] if np.dot(v1[i], running) < 0 else v1[i]
        running = aligned[: i + 1].mean(axis=0)
        n = np.linalg.norm(running)
        if n > 0:
            running /= n
    s = (aligned * w[:, None]).sum(axis=0)
    n = np.linalg.norm(s)
    return s / n if n > 0 else np.array([0.0, 0.0, 1.0])


def dyadic_tensor(v1: np.ndarray, w: np.ndarray) -> np.ndarray:
    """FA-weighted outer-product (dyadic) sum, normalised by total weight."""
    W = w.sum()
    if W <= 0:
        W = 1.0
    return np.einsum("i,ij,ik->jk", w, v1, v1) / W


def dir_dyadic(v1: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Principal eigenvector of the dyadic sum. Sign-invariant."""
    T = dyadic_tensor(v1, w)
    vals, vecs = np.linalg.eigh(T)
    return vecs[:, -1]


def concentration(v1: np.ndarray, w: np.ndarray) -> float:
    """
    Dispersion of the axial distribution, on [0, 1].

    The leading eigenvalue of the normalised dyadic tensor is 1 when every
    vector is parallel and 1/3 when they are uniform on the sphere. Rescaled
    here so that 0 means no information about direction and 1 means perfect
    agreement.
    """
    T = dyadic_tensor(v1, w)
    lam = np.linalg.eigvalsh(T)[-1]
    return float(np.clip((lam - 1.0 / 3.0) / (1.0 - 1.0 / 3.0), 0.0, 1.0))


def shrink_toward(v: np.ndarray, axis: np.ndarray, kappa: float) -> np.ndarray:
    """
    Shrink an estimated direction toward the scanner axis it replaces.

    kappa in [0, 1] is the concentration: at kappa = 1 the estimate is kept,
    at kappa = 0 it collapses to the fixed axis. Signs are matched first so the
    interpolation does not cross the antipode.
    """
    v = v if np.dot(v, axis) >= 0 else -v
    out = kappa * v + (1.0 - kappa) * axis
    n = np.linalg.norm(out)
    return out / n if n > 0 else axis


# ---------------------------------------------------------------------------
# ALPS from cached ROI tensors
# ---------------------------------------------------------------------------


def directional_diffusivity(evals: np.ndarray, evecs: np.ndarray, u: np.ndarray) -> float:
    """Mean over voxels of D(u) = sum_k lambda_k (v_k . u)^2."""
    if len(evals) == 0:
        return np.nan
    dots = np.einsum("nkj,j->nk", np.transpose(evecs, (0, 2, 1)), u)
    return float(np.nanmean((evals * dots**2).sum(axis=1)))


def alps_from_rois(proj: dict, assoc: dict, method: str) -> dict:
    """Classic and refined ALPS for one hemisphere under the named estimator."""
    x = np.array([1.0, 0.0, 0.0])
    y = np.array([0.0, 1.0, 0.0])
    z = np.array([0.0, 0.0, 1.0])

    out = {}
    dp, ap = proj["evals"], proj["evecs"]
    da, aa = assoc["evals"], assoc["evecs"]

    # Classic is identical under every estimator; it uses no estimated axes.
    num = directional_diffusivity(dp, ap, x) + directional_diffusivity(da, aa, x)
    den = directional_diffusivity(dp, ap, y) + directional_diffusivity(da, aa, z)
    out["classic"] = num / den if den else np.nan

    if method == "current":
        vp = dir_running_mean(proj["v1"], proj["fa"])
        va = dir_running_mean(assoc["v1"], assoc["fa"])
    else:
        vp = dir_dyadic(proj["v1"], proj["fa"])
        va = dir_dyadic(assoc["v1"], assoc["fa"])

    if method in ("dyadic+", "shrunk"):
        v_all = np.vstack([proj["v1"], assoc["v1"]])
        w_all = np.concatenate([proj["fa"], assoc["fa"]])
        T = dyadic_tensor(v_all, w_all)
        p = np.linalg.eigh(T)[1][:, 0]          # minor eigenvector
    else:
        p = np.cross(vp, va)
        n = np.linalg.norm(p)
        p = p / n if n > 1e-10 else x

    if method == "shrunk":
        kp = concentration(proj["v1"], proj["fa"])
        ka = concentration(assoc["v1"], assoc["fa"])
        v_all = np.vstack([proj["v1"], assoc["v1"]])
        w_all = np.concatenate([proj["fa"], assoc["fa"]])
        vp = shrink_toward(vp, z, kp)
        va = shrink_toward(va, y, ka)
        p = shrink_toward(p, x, concentration(v_all, w_all))

    def orth(a, b):
        c = np.cross(a, b)
        n = np.linalg.norm(c)
        return c / n if n > 1e-10 else np.array([0.0, 1.0, 0.0])

    op, oa = orth(p, vp), orth(p, va)
    num = directional_diffusivity(dp, ap, p) + directional_diffusivity(da, aa, p)
    den = directional_diffusivity(dp, ap, op) + directional_diffusivity(da, aa, oa)
    out["refined"] = num / den if den else np.nan
    out["theta_scr"] = float(np.degrees(np.arccos(np.clip(abs(np.dot(vp, z)), 0, 1))))
    return out


# ---------------------------------------------------------------------------
# Pass 1: extract and cache
# ---------------------------------------------------------------------------


def extract(max_subjects: int | None) -> None:
    import nibabel as nib

    CACHE.mkdir(parents=True, exist_ok=True)
    rel = pd.read_csv(DIFF / "HCP" / "hcpa_alps_spheres_5mm.csv")
    rel = rel[rel.status == "ok"]
    mot = pd.read_csv(DIFF / "HCP" / "hcpa_motion.csv")
    rel = rel.merge(mot[["Subject_ID", "Visit", "Eddy_Mean_RMS"]],
                    on=["Subject_ID", "Visit"], how="left")
    thr = np.nanpercentile(pd.to_numeric(rel.Eddy_Mean_RMS, errors="coerce").dropna(), 76.4)
    rel = rel[pd.to_numeric(rel.Eddy_Mean_RMS, errors="coerce") <= thr]

    counts = rel.Subject_ID.value_counts()
    subs = sorted(counts[counts >= 2].index)
    if max_subjects:
        rng = np.random.default_rng(20260727)
        subs = list(rng.choice(subs, size=min(max_subjects, len(subs)), replace=False))
    sel = rel[rel.Subject_ID.isin(subs)]
    print(f"caching {len(sel)} sessions from {len(subs)} participants")

    done = skipped = 0
    for i, r in enumerate(sel.itertuples(), 1):
        out = CACHE / f"{r.Subject_ID}_{r.Visit}.npz"
        if out.exists():
            done += 1
            continue
        sdir = OUT / r.DTI_Session_ID
        mask_p = sdir / "processed" / "atlas" / "sphere_roi" / "sphere_roi_combined.nii.gz"
        ev_p = sdir / "processed" / f"tensor_eigenvalues{SHELL}.nii.gz"
        vc_p = sdir / "processed" / f"tensor_eigenvectors{SHELL}.nii.gz"
        if not (mask_p.exists() and ev_p.exists() and vc_p.exists()):
            skipped += 1
            continue
        try:
            mask = nib.load(str(mask_p)).get_fdata().astype(int)
            evals = nib.load(str(ev_p)).get_fdata()
            evecs = nib.load(str(vc_p)).get_fdata()
        except Exception:
            skipped += 1
            continue

        md = evals.mean(axis=-1)
        num = np.sqrt(((evals - md[..., None]) ** 2).sum(axis=-1))
        den = np.sqrt((evals**2).sum(axis=-1))
        fa = np.clip(np.sqrt(1.5) * np.divide(num, den, out=np.zeros_like(num),
                                              where=den != 0), 0, 1)

        xs = np.arange(mask.shape[0])[:, None, None] * np.ones_like(mask)
        mid = mask.shape[0] // 2
        blocks = {}
        for label, code in (("proj", 1), ("assoc", 2)):
            for hemi, sel_h in (("L", xs < mid), ("R", xs >= mid)):
                m = (mask == code) & sel_h & (fa >= 0.2)
                v1 = evecs[m][:, :, 0]
                n = np.linalg.norm(v1, axis=1, keepdims=True)
                n[n == 0] = 1
                blocks[f"{label}_{hemi}_v1"] = (v1 / n).astype(np.float32)
                blocks[f"{label}_{hemi}_fa"] = fa[m].astype(np.float32)
                blocks[f"{label}_{hemi}_evals"] = evals[m].astype(np.float32)
                blocks[f"{label}_{hemi}_evecs"] = evecs[m].astype(np.float32)
        np.savez_compressed(out, **blocks)
        done += 1
        if i % 50 == 0:
            print(f"  {i}/{len(sel)} cached={done} skipped={skipped}", flush=True)
    print(f"cached {done}, skipped {skipped}")


# ---------------------------------------------------------------------------
# Pass 2: compare estimators
# ---------------------------------------------------------------------------


def variance_components(d: pd.DataFrame, col: str) -> dict:
    d = d[["Subject_ID", col]].dropna().rename(columns={col: "y"})
    k = d.Subject_ID.nunique()
    n_i = d.groupby("Subject_ID")["y"].size().to_numpy(float)
    m_i = d.groupby("Subject_ID")["y"].mean().to_numpy(float)
    N = n_i.sum()
    ybar = (n_i * m_i).sum() / N
    ms_b = (n_i * (m_i - ybar) ** 2).sum() / (k - 1)
    ms_w = d.groupby("Subject_ID")["y"].transform(lambda s: s - s.mean()).pow(2).sum() / (N - k)
    n0 = (N - (n_i**2).sum() / N) / (k - 1)
    vb = max((ms_b - ms_w) / n0, 0.0)
    return {"var_within": ms_w, "var_between": vb, "icc": vb / (vb + ms_w),
            "wcv_pct": 100 * np.sqrt(ms_w) / d.y.mean(), "n": len(d), "k": k}


def compare() -> None:
    files = sorted(CACHE.glob("*.npz"))
    print(f"cached sessions: {len(files)}")
    methods = ["current", "dyadic", "dyadic+", "shrunk"]
    rows = []
    for f in files:
        sub, visit = f.stem.rsplit("_", 1)
        z = np.load(f)
        rec = {"Subject_ID": sub, "Visit": visit}
        ok = True
        for m in methods:
            vals = []
            for hemi in ("L", "R"):
                try:
                    proj = {k: z[f"proj_{hemi}_{k}"] for k in ("v1", "fa", "evals", "evecs")}
                    assoc = {k: z[f"assoc_{hemi}_{k}"] for k in ("v1", "fa", "evals", "evecs")}
                except KeyError:
                    ok = False
                    break
                if len(proj["v1"]) < 4 or len(assoc["v1"]) < 4:
                    ok = False
                    break
                vals.append(alps_from_rois(proj, assoc, m))
            if not ok:
                break
            rec[f"classic"] = np.mean([v["classic"] for v in vals])
            rec[f"refined_{m}"] = np.mean([v["refined"] for v in vals])
            rec[f"theta_{m}"] = np.mean([v["theta_scr"] for v in vals])
        if ok:
            rows.append(rec)

    d = pd.DataFrame(rows)
    counts = d.Subject_ID.value_counts()
    d = d[d.Subject_ID.isin(counts[counts >= 2].index)]
    print(f"usable: {len(d)} sessions, {d.Subject_ID.nunique()} participants\n")

    print(f"{'estimator':<12s} {'ICC':>7s} {'var_within':>12s} {'wCV %':>7s} "
          f"{'vs classic':>12s}")
    base = variance_components(d, "classic")
    print(f"{'classic':<12s} {base['icc']:7.3f} {base['var_within']:12.6f} "
          f"{base['wcv_pct']:7.2f} {'reference':>12s}")
    for m in methods:
        vc = variance_components(d, f"refined_{m}")
        ratio = vc["var_within"] / base["var_within"]
        print(f"{'refined:'+m:<12s} {vc['icc']:7.3f} {vc['var_within']:12.6f} "
              f"{vc['wcv_pct']:7.2f} {ratio:11.2f}x")

    print("\nwithin-participant variance ratio against the current estimator:")
    cur = variance_components(d, "refined_current")["var_within"]
    for m in methods[1:]:
        vc = variance_components(d, f"refined_{m}")
        print(f"  {m:<9s} {vc['var_within']/cur:6.3f}x  "
              f"({100*(vc['var_within']/cur-1):+.1f}%)")

    d.to_csv(HERE / "estimator_variants.csv", index=False)
    print(f"\nWrote {HERE/'estimator_variants.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--max-subjects", type=int, default=None)
    a = ap.parse_args()
    if a.extract:
        extract(a.max_subjects)
    if a.compare:
        compare()
