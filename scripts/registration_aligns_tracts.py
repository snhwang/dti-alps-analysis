"""Does registering to a template actually align the tracts?

The manuscript claims a structural registration can bring two brains into gross
correspondence while leaving their tracts running at different angles within it.
That claim is currently argued from the mechanism, that a T1 carries no
directional information about white matter, rather than measured. It is
measurable here, because DLBS carries real head rotation and a full
subject-to-template registration was computed for every session.

The test transforms each session's measured tract direction into template space
and asks how much of the between-participant spread survives. Two stages of the
registration are separable and are reported separately.

  affine     the rotation part of the subject-to-template affine, recovered by
             polar decomposition. This is the part that removes head posture.

  nonlinear  the local Jacobian of the FNIRT warp at the region itself, which is
             what could rotate an individual's tract toward the template's. The
             warp field is stored in subject space and gives, for each subject
             voxel, the corresponding template coordinate, so its Jacobian is
             the local linear map from subject to template. Polar decomposition
             again supplies the rotation.

Nothing is resampled. Rotating the direction and reorienting the tensor are
equivalent, so this measures the registration's effect on orientation without
the interpolation smoothing that warping the data would introduce.

Usage:
    python registration_aligns_tracts.py --affine     (from the cache, fast)
    python registration_aligns_tracts.py --nonlinear  (reads the warp fields)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from data_paths import winpath

import atomic_io  # noqa: F401  writes become atomic on import

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from direction_estimators import weights_for, principal, align, X, Y, Z  # noqa: E402

DIFF = HERE.parent.parent / "diffusion"
OUT = winpath("Q:/dti_output")
CACHE = HERE / "dlbs_tensor_cache"


def axis_angle(v, axis) -> float:
    """Angle in degrees between an axial direction and a reference axis."""
    c = abs(float(np.dot(v / np.linalg.norm(v), axis)))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def polar_rotation(M):
    """Rotation part of a 3x3 linear map, guarding against a reflection."""
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U = U.copy()
        U[:, -1] *= -1
        R = U @ Vt
    return R


def session_directions(z, hemi):
    """The two measured tract directions, exactly as the paper estimates them."""
    proj = {k: z[f"proj_{hemi}_{k}"].astype(np.float64)
            for k in ("v1", "fa", "evals", "evecs")}
    assoc = {k: z[f"assoc_{hemi}_{k}"].astype(np.float64)
             for k in ("v1", "fa", "evals", "evecs")}
    if len(proj["v1"]) < 4 or len(assoc["v1"]) < 4:
        return None
    vp = align(principal(proj["v1"], weights_for("cl", proj)), Z)
    va = align(principal(assoc["v1"], weights_for("cl", assoc)), Y)
    return vp, va


def store_vec(rec, tag, stage, v):
    v = v / np.linalg.norm(v)
    for i, c in enumerate("xyz"):
        rec[f"{tag}_{stage}_{c}"] = float(v[i])


def dispersion(d, tag, stage):
    """
    Angular spread of an axial direction about the cohort mean axis.

    This is the quantity the claim is about. An angle to a fixed axis also
    carries wherever the template's own tract happens to point, which is a
    constant offset and not variability between people. The mean axis is the
    principal eigenvector of the dyadic tensor, since eigenvectors are
    sign-ambiguous and a vector mean would be wrong.
    """
    V = d[[f"{tag}_{stage}_{c}" for c in "xyz"]].dropna().to_numpy()
    if len(V) < 3:
        return None
    T = (V[:, :, None] * V[:, None, :]).mean(axis=0)
    m = np.linalg.eigh(T)[1][:, -1]
    ang = np.degrees(np.arccos(np.clip(np.abs(V @ m), 0, 1)))
    return ang


def run_affine() -> pd.DataFrame:
    rows = []
    for f in sorted(CACHE.glob("*.npz")):
        sub, ses = f.stem.rsplit("_", 1)
        z = np.load(f)
        R = z["R"]
        for hemi in ("L", "R"):
            try:
                got = session_directions(z, hemi)
            except KeyError:
                continue
            if got is None:
                continue
            vp, va = got
            rec = dict(
                Subject_ID=sub, Session=ses, hemi=hemi,
                proj_native=axis_angle(vp, Z), assoc_native=axis_angle(va, Y),
                proj_affine=axis_angle(R @ vp, Z), assoc_affine=axis_angle(R @ va, Y),
            )
            for tag, v in (("proj", vp), ("assoc", va)):
                store_vec(rec, tag, "native", v)
                store_vec(rec, tag, "affine", R @ v)
            rows.append(rec)
    return pd.DataFrame(rows)


def jacobian_rotation(field, voxels, zooms):
    """
    Local subject-to-template rotation, averaged over a region.

    FSL stores this warp as a relative field: it holds the displacement, in mm,
    from each subject voxel to its template counterpart, so the local linear map
    is I + grad(d) rather than grad(d) alone. Omitting the identity gives a
    near-singular matrix and a meaningless rotation, which is worth stating
    because the two differ by a term that looks like a rounding detail.

    The field produced by FNIRT composes the initial affine, so this rotation is
    the whole subject-to-template linear map and is not combined with the affine
    separately. Central differences at one voxel are noisy, so the linear maps
    are averaged across the region before the rotation is taken.
    """
    s = field.shape[:3]
    acc, n = np.zeros((3, 3)), 0
    for i, j, k in voxels:
        if not all(1 <= c < dd - 1 for c, dd in zip((i, j, k), s)):
            continue
        J = np.empty((3, 3))
        for ax in range(3):
            hi, lo = [i, j, k], [i, j, k]
            hi[ax] += 1
            lo[ax] -= 1
            J[:, ax] = (field[tuple(hi)] - field[tuple(lo)]) / (2.0 * zooms[ax])
        J += np.eye(3)
        if np.isfinite(J).all():
            acc += J
            n += 1
    if n == 0:
        return None, np.nan
    J = acc / n
    if abs(np.linalg.det(J)) < 1e-6:
        return None, np.nan
    return polar_rotation(J), float(np.linalg.det(J))


def run_nonlinear(limit=None) -> pd.DataFrame:
    import nibabel as nib

    alps = pd.read_csv(DIFF / "DLBS" / "dlbs_alps_spheres_5mm.csv")
    alps = alps[alps.status == "ok"]
    rows, skipped = [], 0
    todo = list(alps.itertuples())
    if limit:
        todo = todo[:limit]
    for n, r in enumerate(todo, 1):
        cf = CACHE / f"{r.Subject_ID}_{r.Session}.npz"
        sd = OUT / r.DTI_Session_ID / "processed"
        wp = sd / "atlas" / "atlas_to_subject_warp.nii.gz"
        mp = sd / "atlas" / "sphere_roi" / "sphere_roi_combined.nii.gz"
        if not (cf.exists() and wp.exists() and mp.exists()):
            skipped += 1
            continue
        try:
            wim = nib.load(str(wp))
            field = np.asarray(wim.dataobj, dtype=np.float32)
            mask = np.asarray(nib.load(str(mp)).dataobj).astype(int)
        except Exception:
            skipped += 1
            continue
        zooms = wim.header.get_zooms()[:3]
        z = np.load(cf)
        Raff = z["R"]
        mid = mask.shape[0] // 2
        xs = np.arange(mask.shape[0])[:, None, None] * np.ones_like(mask)
        for hemi, sel in (("L", xs < mid), ("R", xs >= mid)):
            try:
                got = session_directions(z, hemi)
            except KeyError:
                got = None
            if got is None:
                continue
            vp, va = got
            rec = dict(Subject_ID=r.Subject_ID, Session=r.Session, hemi=hemi,
                       proj_native=axis_angle(vp, Z), assoc_native=axis_angle(va, Y),
                       proj_affine=axis_angle(Raff @ vp, Z),
                       assoc_affine=axis_angle(Raff @ va, Y))
            good = True
            for tag, code, v, ax in (("proj", 1, vp, Z), ("assoc", 2, va, Y)):
                m = (mask == code) & sel
                if m.sum() == 0:
                    good = False
                    break
                Rl, det = jacobian_rotation(field, np.argwhere(m), zooms)
                if Rl is None:
                    good = False
                    break
                rec[f"{tag}_nonlinear"] = axis_angle(Rl @ v, ax)
                rec[f"{tag}_jac_det"] = det
                store_vec(rec, tag, "native", v)
                store_vec(rec, tag, "affine", Raff @ v)
                store_vec(rec, tag, "nonlinear", Rl @ v)
                # how far the local warp rotation departs from the global affine
                rec[f"{tag}_local_vs_affine"] = np.degrees(np.arccos(np.clip(
                    (np.trace(Rl @ Raff.T) - 1) / 2, -1, 1)))
            if good:
                rows.append(rec)
        if n % 50 == 0:
            print(f"  {n}/{len(todo)} rows={len(rows)} skipped={skipped}", flush=True)
    print(f"done: {len(rows)} rows, {skipped} sessions skipped")
    return pd.DataFrame(rows)


def report(d: pd.DataFrame) -> None:
    stages = [c for c in ("native", "affine", "nonlinear")
              if f"proj_{c}" in d.columns]
    print(f"\n{len(d)} region-hemispheres, "
          f"{d.Subject_ID.nunique()} participants\n")
    print(f"{'tract':8s} {'stage':10s} {'median':>8s} {'IQR':>16s} "
          f"{'5-95 pct':>16s} {'SD':>7s}")
    for tag in ("proj", "assoc"):
        for st in stages:
            v = d[f"{tag}_{st}"].dropna()
            print(f"{tag:8s} {st:10s} {v.median():8.2f} "
                  f"[{np.percentile(v,25):6.2f},{np.percentile(v,75):6.2f}] "
                  f"[{np.percentile(v,5):6.2f},{np.percentile(v,95):6.2f}] "
                  f"{v.std():7.2f}")
        print()
    print("spread about the cohort mean axis, by hemisphere "
          "(the between-participant variability itself)")
    print(f"{'tract':8s} {'hemi':5s} {'stage':10s} {'median':>8s} {'90th':>8s} {'SD':>7s}")
    for tag in ("proj", "assoc"):
        for hemi in ("L", "R"):
            sub = d[d.hemi == hemi]
            for st in stages:
                a = dispersion(sub, tag, st)
                if a is None:
                    continue
                print(f"{tag:8s} {hemi:5s} {st:10s} {np.median(a):8.2f} "
                      f"{np.percentile(a,90):8.2f} {a.std():7.2f}")
        print()

    for tag in ("proj", "assoc"):
        c = f"{tag}_local_vs_affine"
        if c in d.columns:
            v = d[c].dropna()
            print(f"{tag}: local warp rotation departs from the global affine by "
                  f"a median {v.median():.2f} deg (IQR "
                  f"[{np.percentile(v,25):.2f}, {np.percentile(v,75):.2f}])")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--affine", action="store_true")
    ap.add_argument("--nonlinear", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    if a.nonlinear:
        d = run_nonlinear(a.limit)
        d.to_csv(HERE / "registration_aligns_tracts.csv", index=False)
    else:
        d = run_affine()
        d.to_csv(HERE / "registration_aligns_tracts_affine.csv", index=False)
    report(d)
