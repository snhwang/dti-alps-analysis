"""A middle variant: register for the perivascular axis, measure the tract axes.

The refined index takes the perivascular axis as the cross product of the two
measured tract directions. That construction has a property worth questioning.
Because the tract directions differ between hemispheres, so does the cross
product, and the index therefore uses a different perivascular axis on each
side of the same brain. The anatomy does not work that way. Perivascular spaces
around the deep medullary veins run medial to lateral on both sides, so as an
unsigned axis they should be the same direction bilaterally.

This variant keeps everything else and changes only that. A registration
supplies the anatomical left-right direction, which is the same in both
hemispheres by construction, and the projection and association directions are
still measured in each participant. Concretely, with R the rotation of the
subject-to-template affine recovered by polar decomposition, the perivascular
axis is R' x rather than v_proj x v_assoc.

    classic     fixed scanner x, y, z
    refined     p = v_proj x v_assoc, per hemisphere
    anat_x      p = R' x, the same axis in both hemispheres

The denominators are built the same way in all three, orthogonal to p and to
the tract direction of their own region, so the comparison isolates the choice
of perivascular axis and nothing else.

It is rotation-invariant, because R' x is a fixed anatomical direction whatever
the head was doing. Plain scanner x would not be, which is the whole subject of
this paper.

It costs one thing the refined index does not pay. The cross product is exactly
perpendicular to both tract directions, so no fiber lambda1 enters the
numerator. R' x is not, so some does, in proportion to the square of its
component along the fiber. That leakage is reported here rather than assumed
small, since lambda1 is two to three times the perpendicular eigenvalues and the
numerator is what the index is most sensitive to.

    python anatomical_x_variant.py --cohort dlbs

Writes anatomical_x_variant_<cohort>.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from data_paths import winpath

import atomic_io  # noqa: F401  writes become atomic on import

sys.path.insert(0, str(Path(__file__).resolve().parent))
from direction_estimators import X, Y, Z, align, principal, weights_for  # noqa: E402
from estimator_variants import directional_diffusivity  # noqa: E402
from registration_aligns_tracts import polar_rotation  # noqa: E402

HERE = Path(__file__).resolve().parent
DIFF = HERE.parent.parent / "diffusion"
OUT = winpath("Q:/dti_output")
FA_MIN = 0.2
SLAB_MM = 8.0
HEMIS = (("L", 26, 42), ("R", 25, 41))


def unit(v, fallback):
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else fallback


def index_for(proj, assoc, vp, va, p):
    """The ALPS ratio along a given perivascular axis p.

    Denominators are orthogonal to p and to their own region's tract direction,
    exactly as the refined index builds them, so only p varies between variants.
    """
    op = unit(np.cross(p, vp), Y)
    oa = unit(np.cross(p, va), Y)
    num = (directional_diffusivity(proj["evals"], proj["evecs"], p)
           + directional_diffusivity(assoc["evals"], assoc["evecs"], p))
    den = (directional_diffusivity(proj["evals"], proj["evecs"], op)
           + directional_diffusivity(assoc["evals"], assoc["evecs"], oa))
    return num / den if den else np.nan


def classic_index(proj, assoc):
    num = (directional_diffusivity(proj["evals"], proj["evecs"], X)
           + directional_diffusivity(assoc["evals"], assoc["evecs"], X))
    den = (directional_diffusivity(proj["evals"], proj["evecs"], Y)
           + directional_diffusivity(assoc["evals"], assoc["evecs"], Z))
    return num / den if den else np.nan


def session(sd: Path, shell: str):
    """Per-hemisphere indices and geometry for one session."""
    import nibabel as nib

    lab_p = sd / "atlas" / "jhu_labels_registered.nii.gz"
    sph_p = sd / "atlas" / "sphere_roi" / "sphere_roi_combined.nii.gz"
    aff_p = sd / "atlas" / "subject_to_mni_affine.mat"
    if not (lab_p.exists() and sph_p.exists() and aff_p.exists()):
        return None
    try:
        limg = nib.load(str(lab_p))
        lab = limg.get_fdata().astype(int)
        sph = nib.load(str(sph_p)).get_fdata().astype(int)
        evals = nib.load(str(sd / f"tensor_eigenvalues{shell}.nii.gz")).get_fdata()
        evecs = nib.load(str(sd / f"tensor_eigenvectors{shell}.nii.gz")).get_fdata()
        M = np.loadtxt(aff_p)
        if M.shape != (4, 4):
            return None
    except Exception:
        return None

    # Template x pulled back into native space. R maps native to template, the
    # same convention registration_aligns_tracts uses for its affine stage.
    R = polar_rotation(M[:3, :3])
    p_anat = unit(R.T @ X, X)

    md = evals.mean(axis=-1)
    nu = np.sqrt(((evals - md[..., None]) ** 2).sum(axis=-1))
    de = np.sqrt((evals ** 2).sum(axis=-1))
    fa = np.clip(np.sqrt(1.5) * np.divide(nu, de, out=np.zeros_like(nu), where=de != 0), 0, 1)

    ii, jj, kk = np.indices(lab.shape)
    A = limg.affine
    zc = A[2, 0] * ii + A[2, 1] * jj + A[2, 2] * kk + A[2, 3]
    xw = A[0, 0] * ii + A[0, 1] * jj + A[0, 2] * kk + A[0, 3]

    def pack(mask):
        v1 = evecs[mask][:, :, 0]
        n = np.linalg.norm(v1, axis=1, keepdims=True)
        n[n == 0] = 1
        return {"v1": v1 / n, "fa": fa[mask], "evals": evals[mask], "evecs": evecs[mask]}

    z0 = float(np.median(zc[sph > 0])) if (sph > 0).any() else 0.0
    band = np.abs(zc - z0) <= SLAB_MM

    rows = []
    for hemi, scr, slf in HEMIS:
        side = xw < 0 if hemi == "L" else xw > 0
        mp_s, ma_s = (sph == 1) & side & (fa >= FA_MIN), (sph == 2) & side & (fa >= FA_MIN)
        mp_l = (lab == scr) & (fa >= FA_MIN) & band
        ma_l = (lab == slf) & (fa >= FA_MIN) & band
        if mp_s.sum() < 4 or ma_s.sum() < 4 or mp_l.sum() < 10 or ma_l.sum() < 10:
            continue
        proj, assoc = pack(mp_s), pack(ma_s)
        vp = align(principal(pack(mp_l)["v1"], weights_for("cl", pack(mp_l))), Z)
        va = align(principal(pack(ma_l)["v1"], weights_for("cl", pack(ma_l))), Y)
        p_cross = unit(np.cross(vp, va), X)

        rows.append(dict(
            hemi=hemi,
            classic=classic_index(proj, assoc),
            refined=index_for(proj, assoc, vp, va, p_cross),
            anat_x=index_for(proj, assoc, vp, va, p_anat),
            scanner_x=index_for(proj, assoc, vp, va, X),
            # geometry. leak_* is the fraction of lambda1 the numerator admits,
            # which is zero for the cross product by construction.
            leak_proj=float((p_anat @ vp) ** 2),
            leak_assoc=float((p_anat @ va) ** 2),
            anat_to_cross=float(np.degrees(np.arccos(
                np.clip(abs(p_anat @ p_cross), 0, 1)))),
            anat_to_x=float(np.degrees(np.arccos(np.clip(abs(p_anat @ X), 0, 1)))),
            cross_to_x=float(np.degrees(np.arccos(np.clip(abs(p_cross @ X), 0, 1)))),
        ))
    return rows or None


def icc11(d, col):
    """ICC(1,1) from an unbalanced one-way random-effects ANOVA."""
    d = d.dropna(subset=[col])
    ni = d.groupby("Subject_ID")[col].size()
    ni = ni[ni >= 1]
    if len(ni) < 3:
        return np.nan
    mi = d.groupby("Subject_ID")[col].mean()
    a, N, grand = len(ni), int(ni.sum()), d[col].mean()
    msb = float((ni * (mi - grand) ** 2).sum() / (a - 1))
    msw = float(sum(((d[d.Subject_ID == s][col] - mi[s]) ** 2).sum()
                    for s in ni.index) / max(N - a, 1))
    n0 = (N - (ni ** 2).sum() / N) / (a - 1)
    den = msb + (n0 - 1) * msw
    return (msb - msw) / den if den else np.nan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=["dlbs", "hcpa"], default="dlbs")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    shell = "_b1500" if args.cohort == "hcpa" else ""

    if args.cohort == "dlbs":
        src = pd.read_csv(DIFF / "DLBS" / "dlbs_alps_spheres_5mm.csv")
        src = src[src.status == "ok"].copy()
        src["Visit"] = src["Session"]
    else:
        src = pd.read_csv(DIFF / "HCP" / "hcpa_alps_spheres_5mm.csv")
        src = src[src.status == "ok"].copy()
    keep = pd.read_csv(HERE / (f"roi_placement_quality_{args.cohort}"
                               f"{'_b1500' if args.cohort == 'hcpa' else '_all'}.csv"))
    src = src.drop(columns=[c for c in ("Age",) if c in src.columns])
    src = src.merge(keep[["Subject_ID", "Visit", "Age"]], on=["Subject_ID", "Visit"])
    src = src.sort_values(["Subject_ID", "Visit"])
    if args.limit:
        src = src.head(args.limit)
    print(f"{args.cohort}: {len(src)} sessions, {src.Subject_ID.nunique()} participants\n",
          flush=True)

    rows = []
    for i, r in enumerate(src.itertuples(), 1):
        got = session(OUT / r.DTI_Session_ID / "processed", shell)
        if not got:
            continue
        for g in got:
            rows.append(dict(Subject_ID=r.Subject_ID, Visit=r.Visit, Age=r.Age, **g))
        if i % 50 == 0:
            print(f"  {i}/{len(src)}", flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(HERE / f"anatomical_x_variant_{args.cohort}.csv", index=False)
    report(d, args.cohort)


def report(d: pd.DataFrame, cohort: str) -> None:
    from scipy import stats

    m = d.groupby(["Subject_ID", "Visit"]).agg(
        Age=("Age", "first"), classic=("classic", "mean"),
        refined=("refined", "mean"), anat_x=("anat_x", "mean"),
        scanner_x=("scanner_x", "mean"),
        leak_proj=("leak_proj", "mean"), leak_assoc=("leak_assoc", "mean"),
        anat_to_cross=("anat_to_cross", "mean"),
        anat_to_x=("anat_to_x", "mean"),
        cross_to_x=("cross_to_x", "mean")).reset_index()

    print(f"\n{len(m)} sessions, {m.Subject_ID.nunique()} participants\n")
    print("geometry")
    print(f"  registered anatomical x from scanner x : median "
          f"{m.anat_to_x.median():5.2f} deg")
    print(f"  cross product from scanner x           : median "
          f"{m.cross_to_x.median():5.2f} deg")
    print(f"  the two candidate axes from each other : median "
          f"{m.anat_to_cross.median():5.2f} deg")
    print(f"  lambda1 admitted by anat_x, projection : median "
          f"{100 * m.leak_proj.median():5.2f}%  (cross product admits 0 by construction)")
    print(f"  lambda1 admitted by anat_x, association: median "
          f"{100 * m.leak_assoc.median():5.2f}%")

    one = m.sort_values(["Subject_ID", "Visit"]).groupby("Subject_ID").first().reset_index()
    print(f"\nage association, one session per participant (n = {len(one)})")
    for c in ("classic", "refined", "anat_x", "scanner_x"):
        s = one[[c, "Age"]].dropna()
        r, p = stats.pearsonr(s[c], s.Age)
        print(f"  {c:10s} r = {r:+.4f}   p = {p:.3g}")

    rep = m[m.Subject_ID.isin(m.Subject_ID.value_counts()[lambda s: s >= 2].index)]
    if len(rep) > 10:
        print(f"\nreproducibility, ICC(1,1) on {rep.Subject_ID.nunique()} "
              f"participants with repeat visits")
        for c in ("classic", "refined", "anat_x", "scanner_x"):
            print(f"  {c:10s} {icc11(rep, c):.4f}")

    print(f"\nagreement between the two corrected variants: "
          f"r = {m[['refined', 'anat_x']].corr().iloc[0, 1]:.4f}")


if __name__ == "__main__":
    main()
