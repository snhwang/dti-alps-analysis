"""
Registration-based reorientation against the registration-free refined index.

Reviewer 4 point 2 asks for a head-to-head against the reorientation approach
of Tatekawa et al., which Burles et al. call VECREG-ALPS: register the subject
to a template, reorient the diffusion tensors along that transform, then
evaluate the ordinary fixed-axis ALPS formula in the aligned frame.

Reorienting the tensor and rotating the measurement axis are equivalent, since

    u^T (R D R^T) u = (R^T u)^T D (R^T u)

so the reoriented index can be evaluated in native space by rotating the fixed
axes by R^T. That is mathematically identical to warping the tensors but
involves no resampling, so it avoids the interpolation smoothing that would
otherwise confound a variance comparison. R is the rotation part of the
subject-to-template affine, recovered by polar decomposition to strip scale and
shear. Head positioning is a rigid effect, so the affine rotation is the part
that matters; the nonlinear warp absorbs anatomy.

DLBS is the cohort for this. Its acquisitions are natively oblique, so real
head-positioning variance survives, unlike HCP-A whose diffusion data is
already AC-PC aligned.

Variants compared on identical ROIs and voxels:

  classic       fixed scanner axes, no correction
  reoriented    fixed axes rotated into the template frame, the vecreg
                equivalent, requires a structural scan and a registration
  refined       subject-specific axes from the CL-weighted dyadic estimator,
                requires neither
  refined_pv    per-voxel directions rather than ROI-mean, to test whether
                using each voxel's own principal direction helps or hurts

Usage:
    python vecreg_comparison.py --extract
    python vecreg_comparison.py --compare
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from data_paths import winpath

import atomic_io  # noqa: F401  writes become atomic on import

warnings.filterwarnings("ignore")
# Radius of the native-space sphere drawn at each warped region centre,
# matching the default in measured_pvs_axis. Zero restores the warped mask.
SPHERE_MM = float(os.environ.get("ALPS_SPHERE_MM", "5"))


import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from estimator_variants import dyadic_tensor, directional_diffusivity, variance_components
from direction_estimators import weights_for, principal, align, X, Y, Z

HERE = Path(__file__).resolve().parent
DIFF = HERE.parent.parent / "diffusion"
OUT = winpath("Q:/dti_output")
CACHE = HERE / "dlbs_tensor_cache"


def rotation_from_affine(mat_path: Path) -> np.ndarray | None:
    """Rotation part of a FLIRT affine, via polar decomposition."""
    try:
        A = np.loadtxt(mat_path)
    except Exception:
        return None
    if A.shape != (4, 4):
        return None
    M = A[:3, :3]
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:            # guard against a reflection
        U[:, -1] *= -1
        R = U @ Vt
    return R


def extract() -> None:
    import nibabel as nib

    CACHE.mkdir(parents=True, exist_ok=True)
    alps = pd.read_csv(DIFF / "DLBS" / "dlbs_alps_spheres_5mm.csv")
    alps = alps[alps.status == "ok"]
    mot = pd.read_csv(DIFF / "DLBS" / "dlbs_motion.csv")
    alps = alps.merge(mot[["DTI_Session_ID", "Eddy_Mean_RMS"]],
                      on="DTI_Session_ID", how="left")
    alps = alps[pd.to_numeric(alps.Eddy_Mean_RMS, errors="coerce") <= 0.5]
    print(f"sessions to cache: {len(alps)}")

    done = skipped = 0
    for i, r in enumerate(alps.itertuples(), 1):
        out = CACHE / f"{r.Subject_ID}_{r.Session}.npz"
        if out.exists():
            done += 1
            continue
        sd = OUT / r.DTI_Session_ID / "processed"
        mask_p = sd / "atlas" / "sphere_roi" / "sphere_roi_combined.nii.gz"
        aff_p = sd / "atlas" / "subject_to_mni_affine.mat"
        if not (mask_p.exists() and aff_p.exists()
                and (sd / "tensor_eigenvalues.nii.gz").exists()):
            skipped += 1
            continue
        R = rotation_from_affine(aff_p)
        if R is None:
            skipped += 1
            continue
        try:
            mask = nib.load(str(mask_p)).get_fdata().astype(int)
            evals = nib.load(str(sd / "tensor_eigenvalues.nii.gz")).get_fdata()
            evecs = nib.load(str(sd / "tensor_eigenvectors.nii.gz")).get_fdata()
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

        # The same placement rule measured_pvs_axis applies. Without it this
        # comparison would measure the reoriented and closed-form indices
        # inside the warped mask while every index it is compared against was
        # measured inside a redrawn sphere, so the contrast would carry a
        # placement difference as well as a correction difference.
        ii, jj, kk = np.indices(mask.shape)
        aff = nib.load(str(mask_p)).affine
        xw = aff[0, 0] * ii + aff[0, 1] * jj + aff[0, 2] * kk + aff[0, 3]
        yw = aff[1, 0] * ii + aff[1, 1] * jj + aff[1, 2] * kk + aff[1, 3]
        zw = aff[2, 0] * ii + aff[2, 1] * jj + aff[2, 2] * kk + aff[2, 3]

        def resphere(sel_mask, radius):
            if radius <= 0 or not sel_mask.any():
                return sel_mask
            cx, cy, cz = xw[sel_mask].mean(), yw[sel_mask].mean(), zw[sel_mask].mean()
            d2 = (xw - cx) ** 2 + (yw - cy) ** 2 + (zw - cz) ** 2
            return d2 <= radius ** 2

        blocks = {"R": R}
        for label, code in (("proj", 1), ("assoc", 2)):
            for hemi, sel in (("L", xs < mid), ("R", xs >= mid)):
                m = resphere((mask == code) & sel, SPHERE_MM) & (fa >= 0.2)
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
            print(f"  {i}/{len(alps)} cached={done} skipped={skipped}", flush=True)
    print(f"cached {done}, skipped {skipped}")


def alps_axes(proj, assoc, ax_pvs, ax_op, ax_oa) -> float:
    num = (directional_diffusivity(proj["evals"], proj["evecs"], ax_pvs)
           + directional_diffusivity(assoc["evals"], assoc["evecs"], ax_pvs))
    den = (directional_diffusivity(proj["evals"], proj["evecs"], ax_op)
           + directional_diffusivity(assoc["evals"], assoc["evecs"], ax_oa))
    return num / den if den else np.nan


def per_voxel_alps(proj, assoc) -> float:
    """
    Each voxel supplies its own principal direction; the perpendicular axis for
    that voxel comes from crossing it with the other ROI's mean direction, in
    the spirit of LD-ALPS. Reported to test whether per-voxel axes help.
    """
    out_num, out_den = [], []
    mp = align(principal(proj["v1"], weights_for("cl", proj)), Z)
    ma = align(principal(assoc["v1"], weights_for("cl", assoc)), Y)
    for roi, other in ((proj, ma), (assoc, mp)):
        for i in range(len(roi["v1"])):
            v = roi["v1"][i]
            p = np.cross(v, other)
            n = np.linalg.norm(p)
            if n < 1e-8:
                continue
            p /= n
            o = np.cross(p, v)
            n2 = np.linalg.norm(o)
            if n2 < 1e-8:
                continue
            o /= n2
            ev, vc = roi["evals"][i], roi["evecs"][i]
            out_num.append(float((ev * (vc.T @ p) ** 2).sum()))
            out_den.append(float((ev * (vc.T @ o) ** 2).sum()))
    if not out_num or not out_den:
        return np.nan
    return float(np.mean(out_num) / np.mean(out_den))


def compare() -> None:
    rows = []
    for f in sorted(CACHE.glob("*.npz")):
        sub, ses = f.stem.rsplit("_", 1)
        z = np.load(f)
        R = z["R"]
        rec = {"Subject_ID": sub, "Session": ses}
        vals = {k: [] for k in ("classic", "reoriented", "refined", "refined_pv")}
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
            vals["classic"].append(alps_axes(proj, assoc, X, Y, Z))
            # vecreg equivalent: fixed template axes pulled back by R^T
            vals["reoriented"].append(
                alps_axes(proj, assoc, R.T @ X, R.T @ Y, R.T @ Z))
            vp = align(principal(proj["v1"], weights_for("cl", proj)), Z)
            va = align(principal(assoc["v1"], weights_for("cl", assoc)), Y)
            p = np.cross(vp, va)
            n = np.linalg.norm(p)
            p = p / n if n > 1e-10 else X
            op = np.cross(p, vp); op /= max(np.linalg.norm(op), 1e-12)
            oa = np.cross(p, va); oa /= max(np.linalg.norm(oa), 1e-12)
            vals["refined"].append(alps_axes(proj, assoc, p, op, oa))
            vals["refined_pv"].append(per_voxel_alps(proj, assoc))
        if not ok:
            continue
        for k, v in vals.items():
            rec[k] = float(np.mean(v))
        # head rotation relative to template, for stratification
        ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
        rec["head_rot_deg"] = float(ang)
        rows.append(rec)

    d = pd.DataFrame(rows).dropna()
    counts = d.Subject_ID.value_counts()
    lon = d[d.Subject_ID.isin(counts[counts >= 2].index)]
    print(f"sessions {len(d)}, participants {d.Subject_ID.nunique()}")
    print(f"longitudinal {len(lon)} sessions, {lon.Subject_ID.nunique()} participants")
    print(f"head rotation vs template: median {d.head_rot_deg.median():.1f} deg, "
          f"IQR [{d.head_rot_deg.quantile(.25):.1f}, {d.head_rot_deg.quantile(.75):.1f}]")

    within = lon.groupby("Subject_ID")["head_rot_deg"].std(ddof=1).dropna()
    print(f"within-participant SD of head rotation: median {within.median():.2f} deg\n")

    print(f"{'variant':<12s} {'ICC':>7s} {'var_within':>12s} {'wCV %':>7s} {'vs classic':>11s}")
    base = variance_components(lon, "classic")
    for m in ("classic", "reoriented", "refined", "refined_pv"):
        vc = variance_components(lon, m)
        rel = "reference" if m == "classic" else f"{vc['var_within']/base['var_within']:.2f}x"
        print(f"{m:<12s} {vc['icc']:7.3f} {vc['var_within']:12.6f} "
              f"{vc['wcv_pct']:7.2f} {rel:>11s}")

    # Stratify by how much the head actually moved between visits.
    print()
    hi = set(within[within >= within.median()].index)
    for lbl, sel in (("low between-visit rotation", lon[~lon.Subject_ID.isin(hi)]),
                     ("high between-visit rotation", lon[lon.Subject_ID.isin(hi)])):
        sel = sel[sel.Subject_ID.isin(
            sel.Subject_ID.value_counts()[lambda s: s >= 2].index)]
        b = variance_components(sel, "classic")
        print(f"{lbl}  (n={b['n']}, k={b['k']})")
        for m in ("classic", "reoriented", "refined"):
            vc = variance_components(sel, m)
            print(f"    {m:<11s} ICC {vc['icc']:.3f}  var_within {vc['var_within']:.6f}"
                  f"  {vc['var_within']/b['var_within']:.2f}x")

    d.to_csv(HERE / "vecreg_comparison.csv", index=False)
    print(f"\nWrote {HERE/'vecreg_comparison.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--compare", action="store_true")
    a = ap.parse_args()
    if a.extract:
        extract()
    if a.compare:
        compare()
