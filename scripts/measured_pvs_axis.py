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
CP = (l2 - l3) / l1.

CP is not the exact analogue of weighting v1 by CL. Eigenvector stability goes
with the gap to the NEAREST eigenvalue. v1 has one relevant neighbour, l2, so
CL is exactly right for it. v2 has two, l1 and l3, so the analogous weight is
min(CL, CP): CP alone is blind to a closing l1-l2 gap, which frees v2 inside
that eigenplane just as it frees v1. See v2_weight_gap.py, which measures how
far apart the two weights place the pooled axis. The bias from using CP is to
admit voxels with an ill-determined v2, which inflates the within-region
dispersion this axis is used to estimate rather than shrinking it.

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
# region's centre. Zero, the default, keeps the warped mask itself, which is
# the conventional region and the one the manuscript reports.
#
# Redrawing the sphere was the primary analysis for part of this revision,
# because warping leaves region size varying eightfold and that variation had
# to be adjusted for. Testing it showed the adjustment was the error rather
# than the variation: holding size fixed changes the index by r=0.97 to 0.99
# and the age associations by at most 0.006, while the volume-age correlation
# reverses sign between the two placements, +0.29 warped against -0.34 fixed.
# The warped mask grows with the atrophy it is warped into, so adjusting for
# its volume removes age variance through a registration pathway that never
# reaches the measurement. With the adjustment dropped there is no reason to
# leave the conventional region, and every reason to stay with it in a paper
# asking what the published index measures.
SPHERE_MM = float(os.environ.get("ALPS_SPHERE_MM", "0"))
# "sphere" draws a fresh sphere at the warped centre, "erode" takes the same
# volume from inside the warped mask. Both write suffixed filenames.
PLACEMENT = os.environ.get("ALPS_PLACEMENT", "sphere")
# Absolute voxel target for the erode placement, per hemisphere-region.
ERODE_N = int(os.environ.get("ALPS_ERODE_N", "0"))
VARIANTS = ["classic", "cross", "v2_sphere", "v2_slab", "pv_perp", "anat_x"]
SHELL = os.environ.get("ALPS_TENSOR_SUFFIX", "")
# Weight for pooling v2 into a regional axis. "cp" is the Westin planar
# coefficient and what the manuscript reports. "gap" is min(CL, CP), which is
# what the reliability rule in Methods actually gives, since v2 lies between
# two neighbours and its stability is set by the smaller of the two gaps. The
# switch exists so the difference can be measured on the reported quantities
# rather than argued about: only v2_sphere and v2_slab depend on it, because
# classic, cross and anat_x use no v2 axis and pv_perp uses no axis at all.
# "gap" writes its own filenames so neither run overwrites the other.
V2_WEIGHT = os.environ.get("ALPS_V2_WEIGHT", "cp")
assert V2_WEIGHT in ("cp", "gap"), f"ALPS_V2_WEIGHT must be cp or gap, got {V2_WEIGHT}"


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
    if V2_WEIGHT != "cp":
        suffix += f"_v2{V2_WEIGHT}"
    # A subset run must never land on the production filename. Without this a
    # --limit run silently replaces the full cohort's results with a fraction
    # of them, and nothing downstream notices until a number moves.
    if args.limit:
        suffix += f"_limit{args.limit}"
    # The primary 5 mm sphere takes the plain name, since it is what the
    # manuscript reports and what every downstream script should read without
    # having to know about this option at all.
    if not SPHERE_MM:
        _fx = ""
    elif PLACEMENT != "sphere":
        _fx = f"_{PLACEMENT}{ERODE_N}" if ERODE_N else f"_{PLACEMENT}"
    else:
        _fx = f"_sphere{SPHERE_MM:g}"
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
            CL = np.where(l1 > 0, (l1 - l2) / l1, 0.0)
        V2W = CP if V2_WEIGHT == "cp" else np.minimum(CL, CP)
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

            ALPS_PLACEMENT=erode keeps the same fixed size but takes it from
            inside the warped mask, as the voxels nearest its centroid, rather
            than drawing a fresh sphere. That was dismissed above for keeping
            the distorted shape, which turns out to be the property that keeps
            the region inside the tract: measured against the JHU labels, a
            true 5 mm sphere puts 7 to 9 per cent of the association region
            outside its label while the warped mask stays in, because the
            inverse warp erodes it to about 60 per cent of nominal volume.
            """
            if radius <= 0 or not m.any():
                return m
            cx, cy, cz = xw[m].mean(), yw[m].mean(), zw[m].mean()
            d2 = (xw - cx) ** 2 + (yw - cy) ** 2 + (zw - cz) ** 2
            if PLACEMENT != "erode":
                return d2 <= radius ** 2
            # Same target volume, taken from within the warped mask. If the
            # mask is smaller than the target it is used whole, which is the
            # honest behaviour: the region cannot be grown without leaving the
            # tissue the mask identifies.
            target = ERODE_N or int(np.count_nonzero(d2 <= radius ** 2))
            idx = np.flatnonzero(m)
            if idx.size <= target:
                return m
            order = np.argsort(d2.ravel()[idx])[:target]
            out = np.zeros(m.shape, bool)
            out.ravel()[idx[order]] = True
            return out

        def evec(m, which):
            V = vc[m]; o = srt[m]
            v = np.take_along_axis(V, o[:, None, which:which + 1], 2)[:, :, 0]
            return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)

        def pack(m):
            return {"v1": evec(m, 0), "fa": fa[m], "evals": ev[m], "evecs": vc[m]}

        acc = {k: [] for k in VARIANTS}
        # v2_to_* are measured against the slab axis, which is what the
        # shortfall decomposition uses as its dispersion reference. The sphere
        # axis is the one whose own alpha is actually zero by construction,
        # since it is pooled over exactly the voxels the diffusivities come
        # from, so the same angles are recorded against it and the two axes are
        # compared directly. Without v2sph_to_*, the reference cannot be
        # changed without rerunning this.
        ang = {k: [] for k in ("v2_to_x", "v2_to_cross", "cross_to_x",
                               "v2_proj_to_assoc", "v2_proj_to_cross",
                               "v2_assoc_to_cross",
                               "v2sph_to_x", "v2sph_to_cross", "sph_to_slab")}
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
        # n_*_slab are the direction-estimation regions. Methods justifies using
        # a larger region than the measurement one by directional error falling
        # as n^-1/2, and without these the manuscript says "larger" three times
        # without ever saying how much larger.
        cnt = {k: [] for k in ("n_proj", "n_assoc",
                               "n_proj_geom", "n_assoc_geom",
                               "n_proj_slab", "n_assoc_slab")}
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
                w = np.concatenate([V2W[m] for m in masks])
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
            cnt["n_proj_slab"].append(float(mp_l.sum()))
            cnt["n_assoc_slab"].append(float(ma_l.sum()))
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
            ang["v2sph_to_x"].append(acute(p_sph, X))
            ang["v2sph_to_cross"].append(acute(p_sph, p_cross))
            # The offset between the two candidate reference axes. This is the
            # term the slab reference carries and does not estimate, since its
            # axis is pooled over the band while the diffusivities come from
            # the sphere.
            ang["sph_to_slab"].append(acute(p_sph, p_slab))

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
