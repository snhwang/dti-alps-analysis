#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LD-ALPS Post-Processor (refactored)
===================================
Compute LD-ALPS (Local Diffusion ALPS) and related ALPS metrics from diffusion MRI
volumes and region-of-interest labels, with robust handling, logging, and a
reproducible CLI.

Motivation
----------
The conventional DTI-ALPS index is sensitive to head orientation during acquisition.
LD-ALPS mitigates this bias by computing local, voxelwise orthogonal diffusion
directions within each ROI before estimating ADC along the putative perivascular
(glymphatic) axis. See the accompanying manuscript for methodological rationale.

Author: (original) fordb  |  Refactor: public release prep
License: MIT (suggested; change as needed)
Python: 3.9+
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

try:
    import nibabel as nib  # type: ignore
except Exception as e:  # pragma: no cover - import-time diagnostics
    raise SystemExit("nibabel is required. Install with: pip install nibabel") from e

try:
    from sklearn.cluster import DBSCAN  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit("scikit-learn is required. Install with: pip install scikit-learn") from e

try:
    from scipy.interpolate import CloughTocher2DInterpolator, griddata  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit("scipy is required. Install with: pip install scipy") from e


# -------------------------
# Dataclasses & constants
# -------------------------

ROI_LABELS = ("R_Association", "R_Projection", "L_Association", "L_Projection")
# Map each ROI index to its partner within the same hemisphere for cross-product
# 0<->1 (Right), 2<->3 (Left)
ROI_PARTNER = {0: 1, 1: 0, 2: 3, 3: 2}

EPS = 1e-12  # Numerical floor for divisions & logs


@dataclass
class SubjectMetrics:
    subject: str
    ALPS_overall: float
    L_ALPS: float
    R_ALPS: float
    L_Association_x: float
    L_Projection_x: float
    L_Association_i: float
    L_Projection_i: float
    R_Association_x: float
    R_Projection_x: float
    R_Association_i: float
    R_Projection_i: float


# -------------------------
# Utility functions
# -------------------------

def setup_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def load_nifti(path: Path) -> np.ndarray:
    """Load a NIfTI file into a float32 numpy array."""
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    return np.asarray(data, dtype=np.float32)


def normalize_vector(v: np.ndarray) -> np.ndarray:
    """Return the unit-length version of v (or v if zero)."""
    n = float(np.linalg.norm(v))
    if n == 0.0:
        return v
    return v / n


def great_circle_angle(
    v1: np.ndarray, v2: np.ndarray, polarity_invariant: bool = True, degrees: bool = False
) -> float:
    """Great-circle angle between two vectors on the unit sphere.

    If *polarity_invariant* is True (default), flips one vector polarity to respect
    antipodal symmetry (i.e., ±v are equivalent). Returns radians by default.
    """
    v1n = normalize_vector(v1)
    v2n = normalize_vector(v2)
    dot = float(np.clip(np.dot(v1n, v2n), -1.0, 1.0))
    if polarity_invariant:
        dot = abs(dot)
    ang = float(np.arccos(dot))
    return float(np.rad2deg(ang)) if degrees else ang


def pairwise_great_circle_matrix(vecs_a: np.ndarray, vecs_b: np.ndarray | None = None) -> np.ndarray:
    """Return an (N, M) matrix of great-circle angles (radians) between vecs_a and vecs_b.

    Shapes:
        vecs_a: (N, 3)
        vecs_b: (M, 3) or None -> uses vecs_a
    """
    a = np.asarray(vecs_a, dtype=np.float64)
    b = a if vecs_b is None else np.asarray(vecs_b, dtype=np.float64)
    # Normalize
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), EPS)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), EPS)
    # Polarity invariance: use absolute dot product -> angle in [0, pi/2]
    dots = a @ b.T
    dots = np.clip(np.abs(dots), 0.0, 1.0)
    return np.arccos(dots)


def find_tangent_frame(n: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return orthonormal frame (n, u, w) where u,w span the tangent plane at n on S^2."""
    n = normalize_vector(n)
    # Choose arbitrary v not collinear with n
    if not (np.isclose(n[0], 1.0) and np.isclose(n[1], 0.0) and np.isclose(n[2], 0.0)):
        v = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        v = np.array([0.0, 1.0, 0.0], dtype=float)
    u = np.cross(n, v); u = normalize_vector(u)
    w = np.cross(n, u); w = normalize_vector(w)
    return n, u, w


def orthographic_project(v: np.ndarray, origin: np.ndarray) -> Tuple[float, float]:
    """Orthographic projection of v onto the tangent plane at *origin* on S^2.

    If v is on the same hemisphere as origin (dot>0), reflect to the opposite hemisphere
    to handle half-shell acquisitions.
    """
    n, u, w = find_tangent_frame(origin)
    vv = normalize_vector(v.astype(float))
    if float(np.dot(vv, n)) > 0.0:
        vv = -vv  # reflect to far hemisphere
    return float(np.dot(vv, u)), float(np.dot(vv, w))


def batch_orthographic_project(vectors: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Project (K,3) *vectors* to (K,2) orthographic coordinates about *origin*."""
    return np.array([orthographic_project(v, origin) for v in vectors], dtype=float)


def adc_from_signal(s_dwi: np.ndarray, s0_mean: np.ndarray, b: float) -> np.ndarray:
    """Compute ADC: (-1/b) * ln(s_dwi / s0_mean), with numerical guards."""
    ratio = np.maximum(s_dwi / np.maximum(s0_mean, EPS), EPS)
    return (-1.0 / float(b)) * np.log(ratio)


def robust_interpolate_at_origin(
    proj_points: np.ndarray, values: np.ndarray, *, prefer: str = "ct", allow_nearest: bool = True
) -> float:
    """Interpolate *values* defined at 2D *proj_points* and return value at (0,0).

    Strategy:
    1) Deduplicate near-identical points (mean-aggregate values).
    2) Try Clough-Tocher (C1) interpolation.
    3) If CT fails or returns NaN, fall back to griddata('linear'), then 'nearest'.
    """
    assert proj_points.ndim == 2 and proj_points.shape[1] == 2, "proj_points must be (N,2)"
    assert values.ndim == 1 and values.shape[0] == proj_points.shape[0], "values must align with proj_points"

    # Deduplicate by rounding coordinates to reduce high-frequency artifacts
    rounded = np.round(proj_points, 3)
    _, inv, counts = np.unique(rounded, axis=0, return_inverse=True, return_counts=True)
    agg = np.zeros_like(values, dtype=float)
    for k in range(len(counts)):
        agg[inv == k] = np.mean(values[inv == k])

    pts_unique = np.unique(rounded, axis=0)
    vals_unique = np.array([np.mean(values[inv == k]) for k in range(len(counts))])

    # Try preferred method
    def try_ct() -> float:
        interp = CloughTocher2DInterpolator(pts_unique, vals_unique)
        val = float(interp((0.0, 0.0)))
        return val

    def try_grid(kind: str) -> float:
        val = griddata(pts_unique, vals_unique, (0.0, 0.0), method=kind)
        return float(val) if val is not None else np.nan

    estimate = np.nan
    try:
        if prefer == "ct":
            estimate = try_ct()
        else:
            estimate = try_grid(prefer)
    except Exception:
        estimate = np.nan

    if not np.isfinite(estimate):
        for method in ("linear", "nearest"):
            try:
                estimate = try_grid(method)
            except Exception:
                estimate = np.nan
            if np.isfinite(estimate):
                break

    if not np.isfinite(estimate) and allow_nearest:
        # fallback to nearest neighbor in Euclidean sense
        d2 = np.sum(pts_unique**2, axis=1)
        j = int(np.argmin(d2))
        estimate = float(vals_unique[j])

    return float(estimate)


# -------------------------
# Core computation
# -------------------------

def compute_adc_volume(eddy_path: Path, bvals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute ADC per nonzero b-value volume.

    Returns:
        adc_data: 4D array (X,Y,Z,K) of ADC for K nonzero b volumes
        b0_mean: 3D array (X,Y,Z) mean S0 across b=0 volumes
    """
    logging.info("Loading eddy-corrected DWI: %s", eddy_path)
    eddy = load_nifti(eddy_path)  # (X,Y,Z,T)
    if eddy.ndim != 4:
        raise ValueError("Expected 4D DWI at %s" % eddy_path)

    if len(bvals) != eddy.shape[3]:
        raise ValueError(f"bvals length ({len(bvals)}) != DWI volumes ({eddy.shape[3]})")

    mask_b0 = (bvals == 0)
    mask_b = ~mask_b0
    if not np.any(mask_b0) or not np.any(mask_b):
        raise ValueError("bvals must contain both 0 and nonzero entries.")

    b0_mean = np.mean(eddy[..., mask_b0], axis=-1)

    # Vectorized ADC computation across all nonzero b volumes
    nonzero_bs = bvals[mask_b]
    adc_data = np.empty((*eddy.shape[:3], int(np.sum(mask_b))), dtype=np.float32)
    for c, idx in enumerate(np.where(mask_b)[0]):
        adc_data[..., c] = adc_from_signal(eddy[..., idx], b0_mean, nonzero_bs[c]).astype(np.float32)

    return adc_data, b0_mean


def load_subject_files(sub_dir: Path, *, bvecs_rotated: bool = True) -> Tuple[Path, Path, Path, np.ndarray, np.ndarray]:
    """Locate required subject files and load bvecs/bvals."""
    eddy_path = sub_dir / "eddy_corrected_data.nii.gz"
    v1_path = sub_dir / "dti_V1.nii.gz"
    rois_path = sub_dir / "nativeALPSrois.nii.gz"

    if bvecs_rotated:
        bvecs_path = sub_dir / "eddy_corrected_data.eddy_rotated_bvecs"
    else:
        # Conventional FSL naming for original bvecs
        bvecs_path = sub_dir / "dwi.bvec"

    # Common names for bvals; allow either naming
    candidates = [sub_dir / "bvals", sub_dir / "bval", sub_dir / "bval1"]
    bvals_path = next((p for p in candidates if p.exists()), None)
    if bvals_path is None:
        raise FileNotFoundError(f"Could not find a bvals file in {sub_dir} (tried: {', '.join(str(c) for c in candidates)})")

    for p in (eddy_path, v1_path, rois_path, bvecs_path, bvals_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    bvecs = np.loadtxt(bvecs_path, dtype=float)  # (3, K)
    bvals = np.loadtxt(bvals_path, dtype=float)  # (K,)

    if bvecs.ndim != 2 or bvecs.shape[0] != 3:
        raise ValueError(f"bvecs must be shaped (3,K) at {bvecs_path}")
    if bvals.ndim != 1:
        raise ValueError(f"bvals must be 1D at {bvals_path}")

    return eddy_path, v1_path, rois_path, bvecs, bvals


def roi_indices_by_label(rois: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return per-ROI coordinate index triplets for labels 1..4 (R_Assoc, R_Proj, L_Assoc, L_Proj)."""
    idxs: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for label in (1.0, 2.0, 3.0, 4.0):
        wh = np.where(np.isclose(rois, label))
        idxs.append(wh)
    return idxs


def collect_roi_v1s(v1_vol: np.ndarray, roi_idxs: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> List[np.ndarray]:
    """Extract V1 vectors for each ROI (list of arrays of shape (N_i, 3))."""
    out: List[np.ndarray] = []
    for (xi, yi, zi) in roi_idxs:
        vecs = v1_vol[xi, yi, zi, :]  # (N_i, 3)
        out.append(vecs.reshape(-1, 3))
    return out


def cluster_primary_direction(vectors: np.ndarray, *, eps: float = 0.5, min_samples: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Cluster *vectors* on the unit sphere using DBSCAN with great-circle distances.

    Returns:
        keep_mask: boolean mask for vectors belonging to the largest DBSCAN cluster (noise rejected)
        medoid: (3,) vector with minimal mean angle to all vectors in the kept set
    """
    if vectors.size == 0:
        return np.zeros((0,), dtype=bool), np.array([np.nan, np.nan, np.nan])

    V = np.asarray(vectors, dtype=float)
    V = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), EPS)
    D = pairwise_great_circle_matrix(V)  # radians

    # DBSCAN over precomputed distances
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed").fit(D)
    labels = db.labels_
    # Find the largest non-noise cluster id
    ids, counts = np.unique(labels[labels >= 0], return_counts=True)
    if ids.size == 0:
        # No cluster found -> keep everything
        keep = np.ones(V.shape[0], dtype=bool)
        medoid = V[np.argmin(np.mean(D, axis=0))]
        return keep, medoid

    major = int(ids[np.argmax(counts)])
    keep = (labels == major)
    V_keep = V[keep]
    if V_keep.shape[0] == 0:
        keep = np.ones(V.shape[0], dtype=bool)
        V_keep = V

    D_keep = pairwise_great_circle_matrix(V_keep)
    # Medoid index = minimal mean angular distance
    medoid_idx = int(np.argmin(np.mean(D_keep, axis=0)))
    medoid = V_keep[medoid_idx]
    return keep, medoid


def compute_ld_alps_for_subject(
    sub_dir: Path,
    *,
    bvecs_rotated: bool = True,
    cluster_eps: float = 0.5,
    cluster_min_samples: int = 5,
    diagnostics: bool = False,
) -> SubjectMetrics:
    """Compute LD-ALPS metrics for a single subject directory."""
    subject = sub_dir.name
    logging.info("Subject: %s", subject)

    eddy_path, v1_path, rois_path, bvecs, bvals = load_subject_files(sub_dir, bvecs_rotated=bvecs_rotated)
    adc_data, b0_mean = compute_adc_volume(eddy_path, bvals)

    # Use only nonzero b-vectors for interpolation domain
    mask_b = (bvals != 0)
    bvecs_nonzero = bvecs[:, mask_b].T  # (K,3)

    v1_vol = load_nifti(v1_path)        # (X,Y,Z,3)
    rois_vol = load_nifti(rois_path)    # (X,Y,Z)

    roi_idxs = roi_indices_by_label(rois_vol)
    roi_v1s = collect_roi_v1s(v1_vol, roi_idxs)

    # Clean vectors & get medoid per ROI
    roi_keep_masks: List[np.ndarray] = []
    roi_medoids: List[np.ndarray] = []
    clean_roi_idxs: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    for i in range(4):
        V = roi_v1s[i]
        keep, medoid = cluster_primary_direction(V, eps=cluster_eps, min_samples=cluster_min_samples)
        roi_keep_masks.append(keep)
        roi_medoids.append(medoid)

        xi, yi, zi = roi_idxs[i]
        # Apply keep mask to coordinates
        clean_roi_idxs.append((xi[keep], yi[keep], zi[keep]))

    # Compute per-ROI ADC means along X and I axes
    metrics: Dict[str, float] = {}
    for i in range(4):
        xi, yi, zi = clean_roi_idxs[i]
        if xi.size == 0:
            logging.warning("ROI %d has no retained voxels; metrics will be NaN.", i+1)
            metrics[f"{ROI_LABELS[i]}_x"] = np.nan
            metrics[f"{ROI_LABELS[i]}_i"] = np.nan
            continue

        partner = ROI_PARTNER[i]
        partner_medoid = normalize_vector(roi_medoids[partner])

        x_vals: List[float] = []
        i_vals: List[float] = []

        for vx, vy, vz in zip(xi, yi, zi):
            v1 = normalize_vector(v1_vol[vx, vy, vz, :])

            x_axis = normalize_vector(np.cross(v1, partner_medoid))
            i_axis = normalize_vector(np.cross(v1, x_axis))

            adc_vec = adc_data[vx, vy, vz, :]  # (K,)

            # Interpolate ADC at origin after orthographic projection about the target axis
            proj_x = batch_orthographic_project(bvecs_nonzero, x_axis)  # (K,2)
            proj_i = batch_orthographic_project(bvecs_nonzero, i_axis)

            adc_x = robust_interpolate_at_origin(proj_x, adc_vec)
            adc_i = robust_interpolate_at_origin(proj_i, adc_vec)

            # Guard against negative ADCs from interpolation artifacts
            adc_x = float(np.clip(adc_x, 0.0, None))
            adc_i = float(np.clip(adc_i, 0.0, None))

            x_vals.append(adc_x)
            i_vals.append(adc_i)

        metrics[f"{ROI_LABELS[i]}_x"] = float(np.mean(x_vals))
        metrics[f"{ROI_LABELS[i]}_i"] = float(np.mean(i_vals))

    # Hemisphere ALPS and overall ALPS
    L_ALPS = float(
        np.mean([metrics["L_Association_x"], metrics["L_Projection_x"]])
        / np.mean([metrics["L_Association_i"], metrics["L_Projection_i"]])
    )
    R_ALPS = float(
        np.mean([metrics["R_Association_x"], metrics["R_Projection_x"]])
        / np.mean([metrics["R_Association_i"], metrics["R_Projection_i"]])
    )
    ALPS_overall = float(np.mean([L_ALPS, R_ALPS]))

    return SubjectMetrics(
        subject=subject,
        ALPS_overall=ALPS_overall,
        L_ALPS=L_ALPS,
        R_ALPS=R_ALPS,
        L_Association_x=metrics["L_Association_x"],
        L_Projection_x=metrics["L_Projection_x"],
        L_Association_i=metrics["L_Association_i"],
        L_Projection_i=metrics["L_Projection_i"],
        R_Association_x=metrics["R_Association_x"],
        R_Projection_x=metrics["R_Projection_x"],
        R_Association_i=metrics["R_Association_i"],
        R_Projection_i=metrics["R_Projection_i"],
    )


# -------------------------
# CLI
# -------------------------

def discover_subjects(base_dir: Path, prefix: str = "alps_") -> List[Path]:
    """Return subject directories in *base_dir* that start with *prefix* and contain NIfTI files."""
    out: List[Path] = []
    for p in sorted(base_dir.iterdir()):
        if p.is_dir() and p.name.startswith(prefix):
            out.append(p)
    return out


def write_csv(results: Sequence[SubjectMetrics], out_path: Path) -> None:
    import csv
    fieldnames = list(asdict(results[0]).keys()) if results else [
        "subject",
        "ALPS_overall",
        "L_ALPS",
        "R_ALPS",
        "L_Association_x",
        "L_Projection_x",
        "L_Association_i",
        "L_Projection_i",
        "R_Association_x",
        "R_Projection_x",
        "R_Association_i",
        "R_Projection_i",
    ]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow(asdict(r))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="LD-ALPS post-processor: compute LD-ALPS metrics for each subject directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("base_dir", type=Path, help="Base directory containing per-subject subdirectories.")
    ap.add_argument("--subject-prefix", default="alps_", help="Subject folder name prefix to include.")
    ap.add_argument("--bvecs-rotated", action="store_true", default=True, help="Use eddy-rotated bvecs if present.")
    ap.add_argument("--no-bvecs-rotated", dest="bvecs_rotated", action="store_false", help="Use original (unrotated) bvecs.")
    ap.add_argument("--eps", type=float, default=0.5, help="DBSCAN eps (in radians) for V1 clustering.")
    ap.add_argument("--min-samples", type=int, default=5, help="DBSCAN min_samples for V1 clustering.")
    ap.add_argument("-o", "--output", type=Path, default=Path("ld_alps_metrics.csv"), help="Output CSV path.")
    ap.add_argument("-v", "--verbose", action="count", default=0, help="Increase logging verbosity (-v, -vv).")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)

    base_dir: Path = args.base_dir
    if not base_dir.exists():
        logging.error("Base directory does not exist: %s", base_dir)
        return 2

    subs = discover_subjects(base_dir, prefix=args.subject_prefix)
    if not subs:
        logging.error("No subject directories found in %s with prefix '%s'", base_dir, args.subject_prefix)
        return 1

    results: List[SubjectMetrics] = []
    for sub in subs:
        try:
            m = compute_ld_alps_for_subject(
                sub,
                bvecs_rotated=args.bvecs_rotated,
                cluster_eps=args.eps,
                cluster_min_samples=args.min_samples,
            )
            results.append(m)
            logging.info("OK: %s  ALPS=%.3f (L=%.3f R=%.3f)", m.subject, m.ALPS_overall, m.L_ALPS, m.R_ALPS)
        except Exception as e:
            logging.exception("Failed subject %s: %s", sub.name, e)

    if results:
        write_csv(results, args.output)
        logging.info("Wrote %d records to %s", len(results), args.output)
    else:
        logging.warning("No results written.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())