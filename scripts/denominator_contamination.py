"""What measuring the tract directions buys, without reference to any outcome.

DTI-ALPS is defined as a comparison of diffusivities in the plane perpendicular
to the local fiber. The denominators are meant to carry tissue diffusivity
across the tract, not along it. Classic evaluates them along scanner y and z, so
whenever the tract departs from those axes the denominator admits some of the
fiber's own lambda1, which is two to three times the perpendicular eigenvalues.
The quantity computed is then not the quantity defined, whatever it correlates
with.

This measures that contamination directly. For each region and each variant, the
denominator direction u gives a diffusivity sum_i lambda_i (u . v_i)^2, and the
share contributed by lambda1 is

    lambda1 (u . v1)^2 / sum_i lambda_i (u . v_i)^2

averaged over the voxels of the region. For the corrected variants u is built as
a cross product with the measured tract direction, so it stays perpendicular to
the fiber wherever the perivascular axis points and the share collapses to
whatever per-voxel scatter remains about the regional mean.

No age, no reliability, no group contrast. Just how much of each denominator is
the fiber leaking in. It is the construct-validity claim stated as a number, and
it holds whether or not any correlation improves.

    python denominator_contamination.py --cohort dlbs

Writes denominator_contamination_<cohort>.csv.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import atomic_io  # noqa: F401  writes become atomic on import
from data_paths import winpath

sys.path.insert(0, str(Path(__file__).resolve().parent))
from anatomical_x_variant import HEMIS, unit  # noqa: E402
from direction_estimators import X, Y, Z, align, principal, weights_for  # noqa: E402
from registration_aligns_tracts import polar_rotation  # noqa: E402

HERE = Path(__file__).resolve().parent
DIFF = HERE.parent.parent / "diffusion"
OUT = winpath("Q:/dti_output")
FA_MIN = 0.2
SLAB_MM = 8.0
# Radius of the native-space sphere drawn at each warped region centre,
# matching the default in measured_pvs_axis. Zero restores the warped mask.
SPHERE_MM = float(os.environ.get("ALPS_SPHERE_MM", "0"))


def lambda1_share(evals, evecs, u):
    """Fraction of the diffusivity along u that comes from the fiber's own lambda1.

    Zero means u is perpendicular to every voxel's principal direction, so the
    denominator carries only cross-fiber diffusion, which is what the index is
    defined to compare.
    """
    u = np.asarray(u, float)
    # evecs[v, i, j] is component i of eigenvector j, the same convention the
    # rest of the pipeline uses when it takes v1 as evecs[..., 0]. So the dot
    # product contracts the component axis i, not the eigenvector axis j.
    # Contracting the wrong one silently returns a plausible number.
    proj = np.einsum("vij,i->vj", evecs, u) ** 2      # (voxels, 3)
    total = (evals * proj).sum(1)
    lam1 = evals[:, 0] * proj[:, 0]
    good = total > 0
    if not good.any():
        return np.nan
    return float(np.mean(lam1[good] / total[good]))


def age_dependence(d, cohort: str) -> None:
    """Does the contamination vary with age, or only shift the level?

    This separates the two error sources the paper is about. Posture is
    age-graded, so contamination it causes is age-graded too and biases a slope.
    Anatomy is not age-graded, as the tract-direction angles show directly, so
    contamination it causes shifts the level and leaves associations alone.

    That is why template reorientation and the corrected variants reach the same
    age correlation while carrying denominators that differ by a factor of two:
    what reorientation leaves behind is the anatomical part, and that part does
    not covary with age.
    """
    from scipy import stats

    src = ("measured_pvs_axis_dlbs.csv" if cohort == "dlbs"
           else "measured_pvs_axis_hcpa_b1500_all.csv")
    try:
        age = pd.read_csv(HERE / src)[["Subject_ID", "Age"]].drop_duplicates("Subject_ID")
    except Exception:
        return
    x = d.copy()
    for f in (x, age):
        f["Subject_ID"] = f.Subject_ID.astype(str)
    m = x.groupby("Subject_ID").mean(numeric_only=True).reset_index().merge(age, on="Subject_ID")
    if len(m) < 20:
        return
    print()
    print(f"  does the contamination itself track age?  ({len(m)} participants)")
    for tag in ("classic", "vecreg", "refined", "anat_x"):
        cells = []
        for reg in ("proj", "assoc"):
            col = f"{tag}_{reg}"
            if col not in m:
                continue
            r, pv = stats.pearsonr(m[col], m.Age)
            cells.append(f"{reg} r={r:+.3f}{'*' if pv < 0.05 else ' '}")
        if cells:
            print(f"    {tag:10s} " + "   ".join(cells))
    print("    A starred value is an age-graded denominator, which biases a slope.")
    print("    An unstarred one shifts the level and leaves associations alone.")


def main() -> None:
    import nibabel as nib

    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=["dlbs", "hcpa"], default="dlbs")
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args()
    shell = "_b1500" if args.cohort == "hcpa" else ""

    if args.cohort == "dlbs":
        src = pd.read_csv(DIFF / "DLBS" / "dlbs_alps_spheres_5mm.csv")
    else:
        src = pd.read_csv(DIFF / "HCP" / "hcpa_alps_spheres_5mm.csv")
    src = src[src.status == "ok"].head(args.limit)

    rows = []
    for r in src.itertuples():
        sd = OUT / r.DTI_Session_ID / "processed"
        try:
            limg = nib.load(str(sd / "atlas" / "jhu_labels_registered.nii.gz"))
            lab = limg.get_fdata().astype(int)
            sph = nib.load(str(sd / "atlas" / "sphere_roi"
                               / "sphere_roi_combined.nii.gz")).get_fdata().astype(int)
            evals = nib.load(str(sd / f"tensor_eigenvalues{shell}.nii.gz")).get_fdata()
            evecs = nib.load(str(sd / f"tensor_eigenvectors{shell}.nii.gz")).get_fdata()
            M = np.loadtxt(sd / "atlas" / "subject_to_mni_affine.mat")
        except Exception:
            continue
        p_anat = unit(polar_rotation(M[:3, :3]).T @ X, X)

        md = evals.mean(-1)
        nu = np.sqrt(((evals - md[..., None]) ** 2).sum(-1))
        de = np.sqrt((evals ** 2).sum(-1))
        fa = np.clip(np.sqrt(1.5) * np.divide(nu, de, out=np.zeros_like(nu), where=de != 0), 0, 1)
        ii, jj, kk = np.indices(lab.shape)
        A = limg.affine
        zc = A[2, 0] * ii + A[2, 1] * jj + A[2, 2] * kk + A[2, 3]
        xw = A[0, 0] * ii + A[0, 1] * jj + A[0, 2] * kk + A[0, 3]
        yw = A[1, 0] * ii + A[1, 1] * jj + A[1, 2] * kk + A[1, 3]
        z0 = float(np.median(zc[sph > 0])) if (sph > 0).any() else 0.0
        band = np.abs(zc - z0) <= SLAB_MM

        def resphere(m, radius):
            """The placement rule measured_pvs_axis uses, applied here too.

            Contamination is a property of the voxels the index is measured in,
            so measuring it inside the warped mask while the index is measured
            inside a redrawn sphere would describe a region the manuscript does
            not report anywhere else.
            """
            if radius <= 0 or not m.any():
                return m
            cx, cy, cz = xw[m].mean(), yw[m].mean(), zc[m].mean()
            d2 = (xw - cx) ** 2 + (yw - cy) ** 2 + (zc - cz) ** 2
            return d2 <= radius ** 2

        def pack(m):
            v1 = evecs[m][:, :, 0]
            n = np.linalg.norm(v1, axis=1, keepdims=True)
            n[n == 0] = 1
            return {"v1": v1 / n, "fa": fa[m], "evals": evals[m], "evecs": evecs[m]}

        for hemi, scr, slf in HEMIS:
            side = xw < 0 if hemi == "L" else xw > 0
            mp_s = resphere((sph == 1) & side, SPHERE_MM) & (fa >= FA_MIN)
            ma_s = resphere((sph == 2) & side, SPHERE_MM) & (fa >= FA_MIN)
            mp_l = (lab == scr) & (fa >= FA_MIN) & band
            ma_l = (lab == slf) & (fa >= FA_MIN) & band
            if mp_s.sum() < 4 or ma_s.sum() < 4 or mp_l.sum() < 10 or ma_l.sum() < 10:
                continue
            proj, assoc = pack(mp_s), pack(ma_s)
            vp = align(principal(pack(mp_l)["v1"], weights_for("cl", pack(mp_l))), Z)
            va = align(principal(pack(ma_l)["v1"], weights_for("cl", pack(ma_l))), Y)
            p_cross = unit(np.cross(vp, va), X)

            rec = dict(Subject_ID=r.Subject_ID, hemi=hemi)
            # classic: fixed scanner axes for both denominators
            rec["classic_proj"] = lambda1_share(proj["evals"], proj["evecs"], Y)
            rec["classic_assoc"] = lambda1_share(assoc["evals"], assoc["evecs"], Z)
            # template reorientation: vecreg warps the tensors into template
            # space and evaluates fixed template axes there. That is the same
            # quantity as evaluating R'y and R'z in native space, since
            # u'(RDR')u = (R'u)'D(R'u), so it needs no reoriented volumes to
            # measure. Using the affine alone is the favourable case for it: the
            # nonlinear warp's local rotation departs from the affine by 5 to 6
            # degrees and removes less of the direction spread than the affine.
            R = polar_rotation(M[:3, :3])
            rec["vecreg_proj"] = lambda1_share(proj["evals"], proj["evecs"], unit(R.T @ Y, Y))
            rec["vecreg_assoc"] = lambda1_share(assoc["evals"], assoc["evecs"], unit(R.T @ Z, Z))
            # corrected: perpendicular to the measured tract direction by construction
            for tag, p in (("refined", p_cross), ("anat_x", p_anat)):
                rec[f"{tag}_proj"] = lambda1_share(
                    proj["evals"], proj["evecs"], unit(np.cross(p, vp), Y))
                rec[f"{tag}_assoc"] = lambda1_share(
                    assoc["evals"], assoc["evecs"], unit(np.cross(p, va), Y))
            rows.append(rec)

    d = pd.DataFrame(rows)
    d.to_csv(HERE / f"denominator_contamination_{args.cohort}.csv", index=False)
    age_dependence(d, args.cohort)

    print(f"{args.cohort.upper()}: {d.Subject_ID.nunique()} participants, "
          f"{len(d)} region-hemispheres\n")
    print("  share of each denominator contributed by the fiber's own lambda1")
    print(f"  {'variant':10s} {'projection':>12s} {'association':>13s}")
    for tag in ("classic", "vecreg", "refined", "anat_x"):
        print(f"  {tag:10s} {100 * d[f'{tag}_proj'].median():11.2f}% "
              f"{100 * d[f'{tag}_assoc'].median():12.2f}%")
    print()
    print(f"  classic admits {d.classic_proj.median() / max(d.refined_proj.median(), 1e-9):.1f}x "
          f"as much in the projection region and "
          f"{d.classic_assoc.median() / max(d.refined_assoc.median(), 1e-9):.1f}x in the association region.")
    print("  The corrected denominators are perpendicular to the measured fiber by")
    print("  construction, so what remains is per-voxel scatter about the regional mean.")


if __name__ == "__main__":
    main()
