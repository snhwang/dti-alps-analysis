"""Does the measured axis describe individual anatomy better than scanner x?

The claim this tests is not that the corrected index correlates better with age.
It is the prior one, that an axis measured in each participant is a better
description of that participant's anatomy than a fixed scanner axis. Age is the
wrong yardstick for it. A higher age correlation is not evidence of validity,
and this paper's own argument turns on the corrected index having a weaker one.

The reference comes from structural MRI, through pvs_direction, so it is
independent of the diffusion tensor. That independence is the point. The second
eigenvector cannot serve, because in the ALPS regions it is contaminated by
callosal fibers running left to right, so it favours scanner x for a reason
unrelated to perivascular spaces.

Two comparisons, and the second is the one that matters.

  within-session   is the structural perivascular axis closer to the
                   tract-derived axis or to scanner x? A fixed axis can win this
                   on bias alone, since it carries no estimation variance, so a
                   loss here is not decisive.

  between-subject  does the structural axis covary with the tract-derived axis
                   across participants? Scanner x is identical in everyone and
                   so predicts exactly zero of the between-participant variance,
                   by construction and at any bias level. If per-participant
                   perivascular orientation tracks the per-participant measured
                   axis, the measured axis is carrying real individual anatomy.
                   This is the test the claim actually makes, and a fixed axis
                   cannot win it.

No registration is involved anywhere. HCP delivers diffusion under <ID>/T1w/
Diffusion in ACPC space and the structural volumes share that frame, so masks
move between grids by world coordinates alone.

HCP-A is the hardest cohort to win in and is used deliberately. Its diffusion is
anatomically aligned in preprocessing, so scanner x is already close to the
anatomical left-right axis, and it is the cohort where this paper finds no
head-position confound at all. Nothing here is helped by the effect the rest of
the paper is about.

    python pvs_validation.py --limit 20            prototype
    python pvs_validation.py                       full cohort

Writes pvs_validation.csv. Under the AABC Data Use Terms no HCP-A identifier or
derived value may be redistributed, so this output stays local.
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from data_paths import winpath

import atomic_io  # noqa: F401  writes become atomic on import

sys.path.insert(0, str(Path(__file__).resolve().parent))
from direction_estimators import Y, Z, align, principal, weights_for  # noqa: E402
from pvs_direction import angle_between, pvs_axis, resample_mask  # noqa: E402

HERE = Path(__file__).resolve().parent
DIFF = HERE.parent.parent / "diffusion"
OUT = winpath("Q:/dti_output")
STRUCT_DIRS = (winpath("F:/"), winpath("P:/"), winpath("M:/"))
SHELL = "_b1500"
FA_MIN = 0.2
SLAB_MM = 8.0
X = np.array([1.0, 0.0, 0.0])

# JHU ICBM-DTI-81 labels, superior corona radiata and superior longitudinal
# fasciculus, left and right.
HEMIS = (("L", 26, 42), ("R", 25, 41))


def index_structural():
    """Map (Subject_ID, Visit) to its StructuralRecommended zip.

    They are spread across drives with no single root, and H: holds none, so
    the index is built once rather than guessed per session.
    """
    idx = {}
    for root in STRUCT_DIRS:
        if not root.exists():
            continue
        for p in root.rglob("*StructuralRecommended.zip"):
            try:
                if p.stat().st_size < 100 * 1024 * 1024:
                    continue          # stub packages, same trap as the diffusion zips
            except OSError:
                continue
            parts = p.name.split("_")
            if len(parts) >= 2:
                idx.setdefault((parts[0], parts[1]), p)
    return idx


def extract(zip_path: Path, suffix: str, dest: Path):
    """Pull one volume out of a structural package without unpacking it all."""
    with zipfile.ZipFile(zip_path) as z:
        hits = [n for n in z.namelist() if n.endswith(suffix)]
        if not hits:
            return None
        dest.parent.mkdir(parents=True, exist_ok=True)
        with z.open(hits[0]) as src, open(dest, "wb") as out:
            while True:
                chunk = src.read(1 << 22)
                if not chunk:
                    break
                out.write(chunk)
    return dest


def tract_axes(sd: Path):
    """Per-hemisphere projection, association and cross-product axes.

    Built exactly as the paper builds them, from the tract-label slab rather
    than from the measurement spheres, since estimating them from the spheres
    is what produced the retracted inter-fiber angle.
    """
    lab_p = sd / "atlas" / "jhu_labels_registered.nii.gz"
    sph_p = sd / "atlas" / "sphere_roi" / "sphere_roi_combined.nii.gz"
    if not (lab_p.exists() and sph_p.exists()):
        return None
    limg = nib.load(str(lab_p))
    lab = limg.get_fdata().astype(int)
    sph = nib.load(str(sph_p)).get_fdata().astype(int)
    try:
        evals = nib.load(str(sd / f"tensor_eigenvalues{SHELL}.nii.gz")).get_fdata()
        evecs = nib.load(str(sd / f"tensor_eigenvectors{SHELL}.nii.gz")).get_fdata()
    except Exception:
        return None

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

    out = {}
    for hemi, scr, slf in HEMIS:
        side = xw < 0 if hemi == "L" else xw > 0
        mp = (lab == scr) & (fa >= FA_MIN) & band
        ma = (lab == slf) & (fa >= FA_MIN) & band
        if mp.sum() < 10 or ma.sum() < 10:
            continue
        vp = align(principal(pack(mp)["v1"], weights_for("cl", pack(mp))), Z)
        va = align(principal(pack(ma)["v1"], weights_for("cl", pack(ma))), Y)
        p = np.cross(vp, va)
        n = np.linalg.norm(p)
        if n < 1e-9:
            continue
        out[hemi] = dict(vp=vp, va=va, pvs=p / n,
                         sphere_side=(sph > 0) & side, n_lab=int(mp.sum() + ma.sum()))
    return out if out else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="number of participants to run, one session each")
    ap.add_argument("--modality", choices=["T2w", "T1w"], default="T2w",
                    help="T2w shows perivascular spaces bright, T1w dark")
    ap.add_argument("--response-pct", type=float, default=80.0)
    ap.add_argument("--out", default="pvs_validation.csv")
    args = ap.parse_args()
    polarity = "bright" if args.modality == "T2w" else "dark"
    suffix = f"T1w/{args.modality}_acpc_dc_restore.nii.gz"

    src = pd.read_csv(DIFF / "HCP" / "hcpa_alps_spheres_5mm.csv")
    keep = pd.read_csv(HERE / "roi_placement_quality_hcpa_b1500.csv")[
        ["Subject_ID", "Visit", "Age"]]
    src = src.drop(columns=[c for c in ("Age",) if c in src.columns])
    src = src.merge(keep, on=["Subject_ID", "Visit"])
    src = src.sort_values(["Subject_ID", "Visit"]).groupby("Subject_ID").first().reset_index()
    if args.limit:
        src = src.head(args.limit)

    print(f"indexing structural packages ...", flush=True)
    sidx = index_structural()
    print(f"  {len(sidx)} sessions available\n", flush=True)

    scratch = winpath("C:/tmp/pvs_validation")
    scratch.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, r in enumerate(src.itertuples(), 1):
        zp = sidx.get((r.Subject_ID, r.Visit))
        if zp is None:
            continue
        ax = tract_axes(OUT / r.DTI_Session_ID / "processed")
        if ax is None:
            continue
        vol = scratch / f"{r.Subject_ID}_{r.Visit}_{args.modality}.nii.gz"
        try:
            if not vol.exists() and extract(zp, suffix, vol) is None:
                continue
            img = nib.load(str(vol))
            sph_img = nib.load(str(OUT / r.DTI_Session_ID / "processed" / "atlas"
                                    / "sphere_roi" / "sphere_roi_combined.nii.gz"))
            for hemi, d in ax.items():
                roi_img = nib.Nifti1Image(d["sphere_side"].astype(np.uint8),
                                          sph_img.affine)
                roi = resample_mask(roi_img, img)
                paxis, info = pvs_axis(img, roi, polarity=polarity,
                                       response_pct=args.response_pct)
                if paxis is None:
                    continue
                rows.append(dict(
                    Subject_ID=r.Subject_ID, Visit=r.Visit, Age=r.Age, hemi=hemi,
                    pvs_x=float(paxis[0]), pvs_y=float(paxis[1]), pvs_z=float(paxis[2]),
                    tract_x=float(d["pvs"][0]), tract_y=float(d["pvs"][1]),
                    tract_z=float(d["pvs"][2]),
                    to_x=angle_between(paxis, X),
                    to_tract=angle_between(paxis, d["pvs"]),
                    tract_to_x=angle_between(d["pvs"], X),
                    coherence=info["coherence"], n_used=info["n_used"],
                    n_roi=info["n_roi"]))
        finally:
            if vol.exists():
                vol.unlink()          # 46 MB each, do not accumulate
        if rows:
            print(f"  [{i}/{len(src)}] {r.Subject_ID} {r.Visit}  "
                  f"to_x {rows[-1]['to_x']:5.2f}  to_tract {rows[-1]['to_tract']:5.2f}  "
                  f"coh {rows[-1]['coherence']:.3f}", flush=True)

    if not rows:
        print("no sessions produced an axis")
        return
    d = pd.DataFrame(rows)
    d.to_csv(HERE / args.out, index=False)
    report(d)


def report(d: pd.DataFrame) -> None:
    from scipy import stats

    print(f"\n{len(d)} region-hemispheres, {d.Subject_ID.nunique()} participants")
    print(f"  vesselness coherence: median {d.coherence.median():.3f}")
    print(f"\nwithin-session, structural perivascular axis versus each candidate")
    print(f"  to scanner x        median {d.to_x.median():6.2f} deg")
    print(f"  to tract-derived    median {d.to_tract.median():6.2f} deg")
    w = stats.wilcoxon(d.to_x, d.to_tract)
    print(f"  paired difference   {(d.to_x - d.to_tract).median():+6.2f} deg, "
          f"p = {w.pvalue:.3g}, favors tract in "
          f"{100 * (d.to_x > d.to_tract).mean():.1f}% of rows")

    print(f"\nbetween-subject, which candidate tracks individual anatomy")
    print(f"  scanner x is constant, so it explains 0.0% by construction")
    for h in sorted(d.hemi.unique()):
        s = d[d.hemi == h]
        if len(s) < 8:
            continue
        # Do participants whose tract axis leans further from x also show a
        # structural perivascular axis leaning further from x, and in the same
        # direction? Component-wise is the sharper form, since the angle alone
        # discards which way the lean goes.
        r_ang, p_ang = stats.pearsonr(s.tract_to_x, s.to_x)
        print(f"  {h}: angle from x, tract vs structural   r = {r_ang:+.3f}  p = {p_ang:.3g}  n = {len(s)}")
        for c in ("y", "z"):
            r_c, p_c = stats.pearsonr(s[f"tract_{c}"], s[f"pvs_{c}"])
            print(f"     {c} component                          r = {r_c:+.3f}  p = {p_c:.3g}")


if __name__ == "__main__":
    main()
