"""
SUPERSEDED. Do not use the sphere columns from this script.

classic_sphere and refined_sphere are copied from Traditional_Avg and
Refined_Avg in the source table, which come from an earlier pipeline, not
recomputed here. On the 23 sessions this file shares with
roi_variants_hcpa_b1500.csv they disagree by up to 0.295, and the intraclass
correlation they imply for the sphere (0.871) is contradicted by every other
analysis, which gives 0.957 on these same participants.

The whole-tract and slab columns are computed here and are internally
comparable with each other, and they take hemisphere from the JHU label rather
than from geometry, so they are unaffected by the hemisphere-pairing fault. The
region-size comparison the manuscript reports comes from roi_variants.py, which
computes every region the same way on 358 sessions.

Whole anatomical tract labels instead of small spheres.

The conventional ALPS ROIs are small spheres, about 106 native voxels each at a
5 mm radius, and they cannot be enlarged isotropically because the SCR and SLF
centres are only 12 mm apart, so spheres merge above a 6 mm radius. Subsampling
shows the refined index's direction estimate is still improving at 106 voxels,
which raises the question of whether the whole JHU tract labels would serve it
better: SCR is about 7500 template voxels and SLF about 6600, and being
separate labels they never merge.

The trade is anatomical. A sphere samples one locality where the ALPS geometry
is assumed to hold; a whole label integrates along a tract that curves, so the
mean direction is a mixture over its extent. SCR spans z +19 to +44 and SLF z
+2 to +41 in template space, which is a lot of curvature to average over.

Both are computed here on the same sessions so the comparison is within-session.
A slab-restricted variant is also included, the whole label intersected with an
axial band at the conventional ALPS level, which keeps the in-plane extent of
the anatomical label while avoiding integration along the curving tract.

Labels: 25 SCR_R, 26 SCR_L, 41 SLF_R, 42 SLF_L.

Usage:
    python whole_tract_roi.py --limit 140
"""

from __future__ import annotations

import argparse
import glob
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from data_paths import winpath

import atomic_io  # noqa: F401  writes become atomic on import

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from estimator_variants import directional_diffusivity, variance_components
from direction_estimators import weights_for, principal, align, X, Y, Z
from alps_common import parse_age

HERE = Path(__file__).resolve().parent
DIFF = HERE.parent.parent / "diffusion"
OUT = winpath("Q:/dti_output")
LABELS = {"SCR_R": 25, "SCR_L": 26, "SLF_R": 41, "SLF_L": 42}
SLAB_MM = 12.0          # half-height of the axial band, about the ALPS level
FA_MIN = 0.2


def alps(proj, assoc):
    """Classic and refined for one hemisphere, plus the inter-fibre angle."""
    cl = ((directional_diffusivity(proj["evals"], proj["evecs"], X)
           + directional_diffusivity(assoc["evals"], assoc["evecs"], X))
          / (directional_diffusivity(proj["evals"], proj["evecs"], Y)
             + directional_diffusivity(assoc["evals"], assoc["evecs"], Z)))
    vp = align(principal(proj["v1"], weights_for("cl", proj)), Z)
    va = align(principal(assoc["v1"], weights_for("cl", assoc)), Y)
    p = np.cross(vp, va); p /= max(np.linalg.norm(p), 1e-12)
    op = np.cross(p, vp); op /= max(np.linalg.norm(op), 1e-12)
    oa = np.cross(p, va); oa /= max(np.linalg.norm(oa), 1e-12)
    rf = ((directional_diffusivity(proj["evals"], proj["evecs"], p)
           + directional_diffusivity(assoc["evals"], assoc["evecs"], p))
          / (directional_diffusivity(proj["evals"], proj["evecs"], op)
             + directional_diffusivity(assoc["evals"], assoc["evecs"], oa)))
    inter = np.degrees(np.arccos(np.clip(abs(vp @ va), 0, 1)))
    return cl, rf, inter, len(proj["v1"]), len(assoc["v1"])


def main() -> None:
    import nibabel as nib

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=140)
    args = ap.parse_args()

    src = pd.read_csv(DIFF / "HCP" / "hcpa_alps_spheres_5mm.csv")
    src = src[src.status == "ok"].copy()
    src["Age"] = parse_age(src["Age"])
    counts = src.Subject_ID.value_counts()
    src = src[src.Subject_ID.isin(counts[counts >= 2].index)]
    src = src.sort_values(["Subject_ID", "Visit"]).head(args.limit)
    print(f"{len(src)} sessions, {src.Subject_ID.nunique()} participants\n")

    rows = []
    for i, r in enumerate(src.itertuples(), 1):
        sd = OUT / r.DTI_Session_ID / "processed"
        lab_p = sd / "atlas" / "jhu_labels_registered.nii.gz"
        if not lab_p.exists():
            continue
        try:
            lab = nib.load(str(lab_p)).get_fdata().astype(int)
            evals = nib.load(str(sd / "tensor_eigenvalues.nii.gz")).get_fdata()
            evecs = nib.load(str(sd / "tensor_eigenvectors.nii.gz")).get_fdata()
            aff = nib.load(str(lab_p)).affine
        except Exception:
            continue

        md = evals.mean(axis=-1)
        num = np.sqrt(((evals - md[..., None]) ** 2).sum(axis=-1))
        den = np.sqrt((evals**2).sum(axis=-1))
        fa = np.clip(np.sqrt(1.5) * np.divide(num, den, out=np.zeros_like(num),
                                              where=den != 0), 0, 1)

        # world z of every voxel, for the slab variant
        ii, jj, kk = np.indices(lab.shape)
        zc = (aff[2, 0] * ii + aff[2, 1] * jj + aff[2, 2] * kk + aff[2, 3])
        scr_any = np.isin(lab, [25, 26])
        z_centre = float(np.median(zc[scr_any])) if scr_any.any() else 0.0

        def grab(codes, slab):
            m = np.isin(lab, codes) & (fa >= FA_MIN)
            if slab:
                m &= np.abs(zc - z_centre) <= SLAB_MM
            v1 = evecs[m][:, :, 0]
            n = np.linalg.norm(v1, axis=1, keepdims=True); n[n == 0] = 1
            return {"v1": v1 / n, "fa": fa[m], "evals": evals[m], "evecs": evecs[m]}

        rec = {"Subject_ID": r.Subject_ID, "Visit": r.Visit, "Age": r.Age,
               "classic_sphere": r.Traditional_Avg, "refined_sphere": r.Refined_Avg}
        for tag, slab in (("whole", False), ("slab", True)):
            cl, rf, ang, np_, na_ = [], [], [], [], []
            for hemi, sc, sl in (("L", [26], [42]), ("R", [25], [41])):
                proj, assoc = grab(sc, slab), grab(sl, slab)
                if len(proj["v1"]) < 10 or len(assoc["v1"]) < 10:
                    continue
                c, f, a, n1, n2 = alps(proj, assoc)
                cl.append(c); rf.append(f); ang.append(a); np_.append(n1); na_.append(n2)
            if cl:
                rec[f"classic_{tag}"] = float(np.mean(cl))
                rec[f"refined_{tag}"] = float(np.mean(rf))
                rec[f"inter_{tag}"] = float(np.mean(ang))
                rec[f"nvox_{tag}"] = float(np.mean(np_ + na_))
        rows.append(rec)
        if i % 20 == 0:
            print(f"  {i}/{len(src)}", flush=True)

    d = pd.DataFrame(rows).dropna(
        subset=["classic_whole", "refined_whole", "classic_slab", "refined_slab",
                "classic_sphere", "refined_sphere"])
    counts = d.Subject_ID.value_counts()
    lon = d[d.Subject_ID.isin(counts[counts >= 2].index)]
    print(f"\nusable {len(d)} sessions, longitudinal {len(lon)}, "
          f"{lon.Subject_ID.nunique()} participants")
    for tag in ("whole", "slab"):
        print(f"  {tag:6s}: {d['nvox_'+tag].median():.0f} voxels per ROI, "
              f"inter-fibre angle {d['inter_'+tag].median():.1f} deg")
    print(f"  sphere: 106 voxels per ROI (from the 5 mm run)")

    from scipy import stats
    print(f"\n{'ROI':<8s} {'method':<8s} {'ICC':>7s} {'var_within':>12s} "
          f"{'penalty':>8s} {'r age':>8s} {'disatt':>8s}")
    for tag in ("sphere", "slab", "whole"):
        cv = variance_components(lon, f"classic_{tag}")
        rv = variance_components(lon, f"refined_{tag}")
        for nm, vc in (("classic", cv), ("refined", rv)):
            col = f"{nm}_{tag}"
            r = stats.linregress(d["Age"], d[col])[2]
            dis = r / np.sqrt(vc["icc"]) if vc["icc"] > 0 else np.nan
            pen = "" if nm == "classic" else f"{rv['var_within']/cv['var_within']:.2f}x"
            print(f"{tag:<8s} {nm:<8s} {vc['icc']:7.3f} {vc['var_within']:12.6f} "
                  f"{pen:>8s} {r:8.3f} {dis:8.3f}")

    d.to_csv(HERE / "whole_tract_roi.csv", index=False)
    print(f"\nWrote {HERE/'whole_tract_roi.csv'}")


if __name__ == "__main__":
    main()
