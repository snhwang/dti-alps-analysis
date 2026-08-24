"""
Take the perivascular axis from the data instead of from a cross product.

The refined index builds its perivascular axis as v_proj x v_assoc. That is a
construction, not a measurement, and pvs_axis_exists.py showed it is the worse
of the two available descriptions of what is actually there: the measured
second eigenvector sits about 11 degrees from the cross product but only about
8 from the left-right axis.

This replaces the construction with an estimate. Tract directions are still
taken from the tract band as before. The perivascular axis is instead the
principal eigenvector of the planarity-weighted dyadic sum of v2 over the
region, v2 being per voxel the direction of greatest diffusivity in the plane
perpendicular to the local fibre. Dyadic averaging is used because eigenvectors
carry a sign ambiguity, and the weight is the Westin planar coefficient
CP = (l2 - l3) / l1, which is how well defined v2 is in that voxel, exactly
analogous to weighting v1 by CL.

This is rotation-invariant for the same reason the refined index is: v2 rotates
with the tensor, so the averaged axis rotates with it and the ratio does not
change. That distinguishes it from ALPS-PAS, which also uses the lambda2
direction but selects it by alignment with scanner x and is therefore invariant
about x alone.

Variants compared:
  classic     fixed scanner axes
  cross       v_proj x v_assoc, the current refined index
  v2_sphere   measured axis, v2 averaged over the measurement spheres
  v2_slab     measured axis, v2 averaged over the tract bands

Also reported: how far each estimated axis sits from scanner x and from the
cross product, which says whether the measured axis is really a different
answer or the same one by another route.

Usage:
    ALPS_TENSOR_SUFFIX=_b1500 python measured_pvs_axis.py --cohort hcpa --limit 150
"""

from __future__ import annotations

import argparse
import os
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
from registration_aligns_tracts import polar_rotation

HERE = Path(__file__).resolve().parent
DIFF = HERE.parent.parent / "diffusion"
OUT = winpath("Q:/dti_output")
SLAB_MM = 8.0
# The FA floor excludes voxels whose tensor direction is unreliable and whose
# diffusivities are contaminated by CSF, which matters because these regions sit
# near the ventricles. It is also, strictly, a selection on the measured signal,
# which is the objection this paper raises against positioning a region by the
# appearance of the diffusion map. Both choices carry an age dependence and in
# opposite directions: keeping the floor removes more voxels in older brains
# because FA falls, and dropping it admits more ventricular partial volume in
# older brains because ventricles enlarge. It is configurable so the two can be
# compared rather than assumed.
FA_MIN = float(os.environ.get("ALPS_FA_MIN", "0.2"))
# Radius in millimetres of the native-space sphere drawn at each warped
# region's centre. This is the primary analysis, because warping the template
# mask into native space leaves region size varying almost eightfold across
# HCP-A, and a size that varies has to be adjusted for. Drawing the sphere
# fresh at the warped centre holds size fixed instead, so the adjustment
# becomes unnecessary rather than arguable.
#
# Zero restores the warped mask, the behaviour of the first submission. It
# writes to a _warpedmask name so the two can be compared and so neither run
# can silently overwrite the other.
SPHERE_MM = float(os.environ.get("ALPS_SPHERE_MM", "5"))
VARIANTS = ["classic", "cross", "v2_sphere", "v2_slab", "pv_perp", "anat_x"]
SHELL = os.environ.get("ALPS_TENSOR_SUFFIX", "")


def acute(u, v):
    return float(np.degrees(np.arccos(np.clip(abs(float(np.dot(u, v))), 0, 1))))


def main() -> None:
    import nibabel as nib

    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=["hcpa", "dlbs"], default="hcpa")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--all-sessions", action="store_true",
                    help="keep single-visit participants, for phenotype overlap")
    args = ap.parse_args()

    if args.cohort == "hcpa":
        src = pd.read_csv(DIFF / "HCP" / "hcpa_alps_spheres_5mm.csv")
        mot = pd.read_csv(DIFF / "HCP" / "hcpa_motion.csv")
        src = src.merge(mot[["Subject_ID", "Visit", "Eddy_Mean_RMS"]],
                        on=["Subject_ID", "Visit"], how="left")
        rms = pd.to_numeric(src.Eddy_Mean_RMS, errors="coerce")
        thr = float(np.nanpercentile(rms.dropna(), 76.4))
    else:
        src = pd.read_csv(DIFF / "DLBS" / "dlbs_alps_spheres_5mm.csv")
        mot = pd.read_csv(DIFF / "DLBS" / "dlbs_motion.csv")
        src = src.merge(mot[["DTI_Session_ID", "Eddy_Mean_RMS"]],
                        on="DTI_Session_ID", how="left")
        rms = pd.to_numeric(src.Eddy_Mean_RMS, errors="coerce")
        thr = 0.5
        src["Visit"] = src["Session"]

    src = src[(src.status == "ok") & (rms <= thr)].copy()
    src["Age"] = parse_age(src["Age"])
    src = src.dropna(subset=["Age"])
    if not args.all_sessions:
        counts = src.Subject_ID.value_counts()
        src = src[src.Subject_ID.isin(counts[counts >= 2].index)]
    if args.limit:
        rng = np.random.default_rng(20260810)
        keep = rng.choice(sorted(src.Subject_ID.unique()),
                          size=min(args.limit, src.Subject_ID.nunique()), replace=False)
        src = src[src.Subject_ID.isin(keep)]
    src = src.sort_values(["Subject_ID", "Visit"])
    print(f"cohort {args.cohort}: {len(src)} sessions, {src.Subject_ID.nunique()} participants\n")

    # Resolved before the loop so the partial results can be flushed as they
    # accumulate. A run of this length should not have to start over because
    # the machine was restarted near the end of it.
    suffix = "_all" if args.all_sessions else ""
    # The primary 5 mm sphere takes the plain name, since it is what the
    # manuscript reports and what every downstream script should read without
    # having to know about this option at all.
    if SPHERE_MM == 5:
        _fx = ""
    elif SPHERE_MM:
        _fx = f"_sph{SPHERE_MM:g}"
    else:
        _fx = "_warpedmask"
    if FA_MIN != 0.2:
        _fx += f"_fa{FA_MIN:g}"
    outp = HERE / f"measured_pvs_axis_{args.cohort}{SHELL}{suffix}{_fx}.csv"
    print(f"writing to {outp.name}, flushed every 50 sessions\n", flush=True)

    rows = []
    for i, r in enumerate(src.itertuples(), 1):
        sd = OUT / r.DTI_Session_ID / "processed"
        lab_p = sd / "atlas" / "jhu_labels_registered.nii.gz"
        sph_p = sd / "atlas" / "sphere_roi" / "sphere_roi_combined.nii.gz"
        if not (lab_p.exists() and sph_p.exists()):
            continue
        try:
            limg = nib.load(str(lab_p)); lab = limg.get_fdata().astype(int)
            sph = nib.load(str(sph_p)).get_fdata().astype(int)
            ev = nib.load(str(sd / f"tensor_eigenvalues{SHELL}.nii.gz")).get_fdata()
            vc = nib.load(str(sd / f"tensor_eigenvectors{SHELL}.nii.gz")).get_fdata()
        except Exception:
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

        ii, jj, kk = np.indices(lab.shape)
        Af = limg.affine
        xw = Af[0, 0] * ii + Af[0, 1] * jj + Af[0, 2] * kk + Af[0, 3]
        yw = Af[1, 0] * ii + Af[1, 1] * jj + Af[1, 2] * kk + Af[1, 3]
        zw = Af[2, 0] * ii + Af[2, 1] * jj + Af[2, 2] * kk + Af[2, 3]

        def resphere(m, radius):
            """Replace a warped region with a true sphere at its own centre.

            The template spheres are a fixed 515 mm^3, but warping the mask
            into native space distorts both its size and its shape: HCP-A
            regions range over thirteenfold. Eroding the warped mask would fix
            the size and keep the distorted shape.

            Warping only the centre and drawing the sphere fresh in native
            space fixes both. Registration then decides where the region sits,
            which is what it is good at, and not what shape it is. This is also
            what the original method does, which places spheres in native space
            rather than warping masks into it.

            The centre is the centroid of the warped mask in millimetres, so
            anisotropic voxels do not squash it. Selection is geometric and
            looks at no measured quantity, so unlike an FA-ranked rule it
            cannot introduce a dependence on age.
            """
            if radius <= 0 or not m.any():
                return m
            cx, cy, cz = xw[m].mean(), yw[m].mean(), zw[m].mean()
            d2 = (xw - cx) ** 2 + (yw - cy) ** 2 + (zw - cz) ** 2
            return d2 <= radius ** 2

        def evec(m, which):
            V = vc[m]; o = srt[m]
            v = np.take_along_axis(V, o[:, None, which:which + 1], 2)[:, :, 0]
            return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)

        def pack(m):
            return {"v1": evec(m, 0), "fa": fa[m], "evals": ev[m], "evecs": vc[m]}

        acc = {k: [] for k in VARIANTS}
        ang = {k: [] for k in ("v2_to_x", "v2_to_cross", "cross_to_x",
                               "v2_proj_to_assoc", "v2_proj_to_cross",
                               "v2_assoc_to_cross")}
        # regional eigenvalue means, so any normalization of the transverse
        # anisotropy can be formed downstream without returning to the images
        eig = {f"{e}_{r}": [] for e in ("l1", "l2", "l3")
               for r in ("proj", "assoc")}
        # Region sizes, so the effect of re-sphering on size variation can be
        # read off the output instead of assumed. The geometric count is what
        # the placement rule alone produces and is fixed by construction once
        # the radius is fixed. The plain count is what survives the FA mask,
        # which the sphere does not control, so the two differ and both are
        # worth carrying. Counts are per hemisphere, like every other quantity
        # here, since the record averages over the two.
        cnt = {k: [] for k in ("n_proj", "n_assoc",
                               "n_proj_geom", "n_assoc_geom")}
        for hemi, side, scr, slf in (("L", xw < 0, 26, 42), ("R", xw > 0, 25, 41)):
            if SPHERE_MM:
                mp_g = resphere((sph == 1) & side, SPHERE_MM)
                ma_g = resphere((sph == 2) & side, SPHERE_MM)
            else:
                mp_g = (sph == 1) & side
                ma_g = (sph == 2) & side
            mp_s = mp_g & (fa >= FA_MIN)
            ma_s = ma_g & (fa >= FA_MIN)
            if mp_s.sum() < 4 or ma_s.sum() < 4:
                continue
            z0 = float(np.median(zw[sph > 0])) if (sph > 0).any() else 0.0
            band = np.abs(zw - z0) <= SLAB_MM
            mp_l = (lab == scr) & (fa >= FA_MIN) & band
            ma_l = (lab == slf) & (fa >= FA_MIN) & band
            if mp_l.sum() < 10 or ma_l.sum() < 10:
                continue

            P, A = pack(mp_s), pack(ma_s)
            vp = align(principal(pack(mp_l)["v1"], weights_for("cl", pack(mp_l))), Z)
            va = align(principal(pack(ma_l)["v1"], weights_for("cl", pack(ma_l))), Y)
            p_cross = np.cross(vp, va); p_cross /= max(np.linalg.norm(p_cross), 1e-12)

            def v2_axis(masks):
                v2 = np.vstack([evec(m, 1) for m in masks])
                w = np.concatenate([CP[m] for m in masks])
                ok = w > 0
                if ok.sum() < 6:
                    return None
                a = principal(v2[ok], w[ok])
                return a / max(np.linalg.norm(a), 1e-12)

            p_sph = v2_axis([mp_s, ma_s])
            p_slab = v2_axis([mp_l, ma_l])
            if p_sph is None or p_slab is None:
                continue
            p_sph = align(p_sph, X); p_slab = align(p_slab, X)

            def alps(p):
                op = np.cross(p, vp); op /= max(np.linalg.norm(op), 1e-12)
                oa = np.cross(p, va); oa /= max(np.linalg.norm(oa), 1e-12)
                return ((dd(P["evals"], P["evecs"], p) + dd(A["evals"], A["evecs"], p))
                        / (dd(P["evals"], P["evecs"], op) + dd(A["evals"], A["evecs"], oa)))

            acc["classic"].append((dd(P["evals"], P["evecs"], X) + dd(A["evals"], A["evecs"], X))
                                  / (dd(P["evals"], P["evecs"], Y) + dd(A["evals"], A["evecs"], Z)))

            # Per-voxel greatest perpendicular direction. Using each voxel's own
            # largest perpendicular diffusivity makes the numerator lambda2 and the
            # denominator lambda3, so no axis is estimated and invariance is exact:
            # eigenvalues are unchanged by rotation.
            acc["pv_perp"].append(float((l2[mp_s].mean() + l2[ma_s].mean())
                                        / (l3[mp_s].mean() + l3[ma_s].mean())))
            for _e, _v in (("l1", l1), ("l2", l2), ("l3", l3)):
                eig[f"{_e}_proj"].append(float(_v[mp_s].mean()))
                eig[f"{_e}_assoc"].append(float(_v[ma_s].mean()))
            cnt["n_proj"].append(float(mp_s.sum()))
            cnt["n_assoc"].append(float(ma_s.sum()))
            cnt["n_proj_geom"].append(float(mp_g.sum()))
            cnt["n_assoc_geom"].append(float(ma_g.sum()))
            acc["cross"].append(alps(p_cross))
            # Anatomical left-right, the same axis in both hemispheres, pulled
            # back from the template by the rotation of the subject-to-template
            # affine. Rotation-invariant because R'x is fixed in the anatomy
            # whatever the head was doing. A session without an affine simply
            # contributes no anat_x, keeping every other variant intact.
            _aff = sd / "atlas" / "subject_to_mni_affine.mat"
            if _aff.exists():
                try:
                    _M = np.loadtxt(_aff)
                    if _M.shape == (4, 4):
                        _p = polar_rotation(_M[:3, :3]).T @ X
                        _n = np.linalg.norm(_p)
                        if _n > 1e-10:
                            acc["anat_x"].append(alps(_p / _n))
                except Exception:
                    pass
            acc["v2_sphere"].append(alps(p_sph))
            acc["v2_slab"].append(alps(p_slab))
            ang["v2_to_x"].append(acute(p_slab, X))
            ang["v2_to_cross"].append(acute(p_slab, p_cross))
            ang["cross_to_x"].append(acute(p_cross, X))

            # One axis has to serve both regions, and it attains the bound in
            # both only if v2 is the same direction in each and equals their
            # common perpendicular. Both halves of that are testable, so the
            # two regional axes are also estimated apart.
            p_proj = v2_axis([mp_l])
            p_assoc = v2_axis([ma_l])
            if p_proj is not None and p_assoc is not None:
                p_proj, p_assoc = align(p_proj, X), align(p_assoc, X)
                ang["v2_proj_to_assoc"].append(acute(p_proj, p_assoc))
                ang["v2_proj_to_cross"].append(acute(p_proj, p_cross))
                ang["v2_assoc_to_cross"].append(acute(p_assoc, p_cross))

        if not acc["classic"]:
            continue
        rec = {"Subject_ID": r.Subject_ID, "Visit": r.Visit, "Age": r.Age}
        for k, v in {**acc, **ang, **eig, **cnt}.items():
            if v:
                rec[k] = float(np.mean(v))
        rows.append(rec)
        if i % 50 == 0:
            print(f"  {i}/{len(src)}", flush=True)
            pd.DataFrame(rows).to_csv(outp, index=False)

    d = pd.DataFrame(rows)
    d.to_csv(outp, index=False)
    lon = d[d.Subject_ID.isin(d.Subject_ID.value_counts()[lambda s: s >= 2].index)]
    print(f"\n{len(d)} sessions, {len(lon)} longitudinal, {lon.Subject_ID.nunique()} participants\n")

    print("axis geometry (degrees, mean)")
    for k in ("v2_to_x", "v2_to_cross", "cross_to_x",
              "v2_proj_to_assoc", "v2_proj_to_cross", "v2_assoc_to_cross"):
        if k in d:
            print(f"  {k:<18s} {d[k].mean():5.1f}   median {d[k].median():5.1f}")
    if "v2_proj_to_assoc" in d:
        print("\n  A single axis attains the bound in both regions only if the two")
        print("  regional v2 directions coincide and equal the common perpendicular.")
        print(f"  They differ by a median {d.v2_proj_to_assoc.median():.1f} degrees, so")
        print("  no one axis can be the aligned axis for both.")

    base = variance_components(lon.dropna(subset=["classic"]), "classic")
    print(f"\n{'variant':<12s} {'ICC':>7s} {'var/classic':>12s} {'r age':>8s} {'disatt':>8s}")
    for k in VARIANTS:
        s = lon.dropna(subset=[k])
        if len(s) < 20:
            continue
        vc = variance_components(s, k)
        ds = d.dropna(subset=[k])
        r = stats.pearsonr(ds.Age, ds[k])[0]
        print(f"{k:<12s} {vc['icc']:7.3f} {vc['var_within']/base['var_within']:12.2f} "
              f"{r:8.3f} {r/np.sqrt(max(vc['icc'],1e-9)):8.3f}")


if __name__ == "__main__":
    main()
