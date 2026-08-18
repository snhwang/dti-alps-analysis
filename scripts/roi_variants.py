"""
Better measurement regions for DTI-ALPS, not restricted to spheres.

why_offtract.py showed the conventional 5 mm sphere spills past the tract: the
voxels whose direction does not match the tract are markedly more isotropic
(CS 1.43x, FA 0.66x, CL 0.64x) and sit about half as far from the label
boundary, while planarity is unchanged, so it is partial volume at the tract
edge rather than fibre crossing, and it is the same in both cohorts despite very
different acquisitions.

That points at the region, not the data. This sweeps alternatives.

  sphere       conventional 5 mm sphere. Baseline.
  sph_lab      sphere intersected with the JHU tract label.
  sph_e1       sphere and label, dropping voxels within 1 mm of the label edge.
  sph_e2       same at 2 mm.
  core_n       the N most interior voxels of the label at the ALPS level, N set
               to the sphere's own voxel count, so size is matched and only
               shape and position differ.
  band         the whole label restricted to the 8 mm axial band. Large.
  fa_core      sphere and label with FA >= 0.35 rather than 0.2.

Every criterion is geometric or scalar. None selects voxels by their direction
relative to the scanner, which would make the region definition depend on head
position and reintroduce the artefact the paper is about.

Direction estimation is held fixed at the 8 mm tract band for every variant, so
differences are attributable to the measurement region alone.

Endpoints: test-retest ICC, within-participant variance, age association, and
the off-tract fraction, which says whether the spill was actually fixed.

Usage:
    ALPS_TENSOR_SUFFIX=_b1500 python roi_variants.py --cohort hcpa --limit 150
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

HERE = Path(__file__).resolve().parent
DIFF = HERE.parent.parent / "diffusion"
OUT = winpath("Q:/dti_output")
SLAB_MM, FA_MIN, THRESH = 8.0, 0.2, 45.0
SHELL = os.environ.get("ALPS_TENSOR_SUFFIX", "")
VARIANTS = ["sphere", "sph_lab", "sph_e2", "sph_e3", "core_n", "core_half", "band", "fa_core"]


def main() -> None:
    import nibabel as nib
    from scipy import ndimage

    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=["hcpa", "dlbs"], default="hcpa")
    ap.add_argument("--limit", type=int, default=None)
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
    counts = src.Subject_ID.value_counts()
    src = src[src.Subject_ID.isin(counts[counts >= 2].index)]
    if args.limit:
        rng = np.random.default_rng(20260810)
        keep = rng.choice(sorted(src.Subject_ID.unique()),
                          size=min(args.limit, src.Subject_ID.nunique()), replace=False)
        src = src[src.Subject_ID.isin(keep)]
    src = src.sort_values(["Subject_ID", "Visit"])
    print(f"cohort {args.cohort}: {len(src)} sessions, {src.Subject_ID.nunique()} participants\n")

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

        md = ev.mean(-1)
        nu = np.sqrt(((ev - md[..., None]) ** 2).sum(-1))
        de = np.sqrt((ev ** 2).sum(-1))
        fa = np.clip(np.sqrt(1.5) * np.divide(nu, de, out=np.zeros_like(nu), where=de != 0), 0, 1)

        ii, jj, kk = np.indices(lab.shape)
        Af = limg.affine
        vox = np.abs(np.diag(Af)[:3])
        xw = Af[0, 0] * ii + Af[0, 1] * jj + Af[0, 2] * kk + Af[0, 3]
        zw = Af[2, 0] * ii + Af[2, 1] * jj + Af[2, 2] * kk + Af[2, 3]

        rec = {"Subject_ID": r.Subject_ID, "Visit": r.Visit, "Age": r.Age}
        acc = {}
        for hemi, side, scr, slf in (("L", xw < 0, 26, 42), ("R", xw > 0, 25, 41)):
            base_p = (sph == 1) & side & (fa >= FA_MIN)
            base_a = (sph == 2) & side & (fa >= FA_MIN)
            if base_p.sum() < 4 or base_a.sum() < 4:
                continue
            z0 = float(np.median(zw[sph > 0])) if (sph > 0).any() else 0.0
            band = np.abs(zw - z0) <= SLAB_MM
            lp, la = (lab == scr), (lab == slf)
            bp = lp & (fa >= FA_MIN) & band
            ba = la & (fa >= FA_MIN) & band
            if bp.sum() < 10 or ba.sum() < 10:
                continue

            def pk(m):
                v1 = vc[m][:, :, 0]
                n = np.linalg.norm(v1, axis=1, keepdims=True); n[n == 0] = 1
                return {"v1": v1 / n, "fa": fa[m], "evals": ev[m], "evecs": vc[m]}

            vp = align(principal(pk(bp)["v1"], weights_for("cl", pk(bp))), Z)
            va = align(principal(pk(ba)["v1"], weights_for("cl", pk(ba))), Y)
            p = np.cross(vp, va); p /= max(np.linalg.norm(p), 1e-12)
            op = np.cross(p, vp); op /= max(np.linalg.norm(op), 1e-12)
            oa = np.cross(p, va); oa /= max(np.linalg.norm(oa), 1e-12)

            dp_ = ndimage.distance_transform_edt(lp, sampling=vox)
            da_ = ndimage.distance_transform_edt(la, sampling=vox)

            def core_n(base, lab_m, dist, n_target):
                cand = lab_m & (fa >= FA_MIN) & band
                if cand.sum() == 0:
                    return None
                idx = np.argwhere(cand)
                dvals = dist[cand]
                # nearest to the conventional centre, then most interior
                ctr = np.array(np.nonzero(base)).mean(axis=1)
                near = np.linalg.norm(idx - ctr, axis=1)
                order = np.lexsort((near, -dvals))
                sel = idx[order[:max(n_target, 4)]]
                m = np.zeros_like(cand)
                m[tuple(sel.T)] = True
                return m

            defs = {
                "sphere":  (base_p, base_a),
                "sph_lab": (base_p & lp, base_a & la),
                "sph_e2":  (base_p & lp & (dp_ >= 2.0), base_a & la & (da_ >= 2.0)),
                "sph_e3":  (base_p & lp & (dp_ >= 3.0), base_a & la & (da_ >= 3.0)),
                "core_n":  (core_n(base_p, lp, dp_, int(base_p.sum())),
                            core_n(base_a, la, da_, int(base_a.sum()))),
                "core_half": (core_n(base_p, lp, dp_, int(base_p.sum()) // 2),
                              core_n(base_a, la, da_, int(base_a.sum()) // 2)),
                "band":    (bp, ba),
                "fa_core": (base_p & lp & (fa >= 0.35), base_a & la & (fa >= 0.35)),
            }
            for nm, (mp, ma) in defs.items():
                if mp is None or ma is None or mp.sum() < 4 or ma.sum() < 4:
                    continue
                P, A = pk(mp), pk(ma)
                acc.setdefault(f"classic_{nm}", []).append(
                    (dd(P["evals"], P["evecs"], X) + dd(A["evals"], A["evecs"], X))
                    / (dd(P["evals"], P["evecs"], Y) + dd(A["evals"], A["evecs"], Z)))
                acc.setdefault(f"refined_{nm}", []).append(
                    (dd(P["evals"], P["evecs"], p) + dd(A["evals"], A["evecs"], p))
                    / (dd(P["evals"], P["evecs"], op) + dd(A["evals"], A["evecs"], oa)))
                ang = np.degrees(np.arccos(np.clip(np.abs(A["v1"] @ va), 0, 1)))
                acc.setdefault(f"off_{nm}", []).append(float((ang > THRESH).mean()))
                acc.setdefault(f"nvox_{nm}", []).append(float(ma.sum()))

        if not acc:
            continue
        for k, v in acc.items():
            rec[k] = float(np.mean(v))
        rows.append(rec)
        if i % 50 == 0:
            print(f"  {i}/{len(src)}", flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(HERE / f"roi_variants_{args.cohort}{SHELL}.csv", index=False)
    lon = d[d.Subject_ID.isin(d.Subject_ID.value_counts()[lambda s: s >= 2].index)]
    print(f"\n{len(d)} sessions, {len(lon)} longitudinal, {lon.Subject_ID.nunique()} participants\n")

    base_v = variance_components(lon.dropna(subset=["classic_sphere"]), "classic_sphere")["var_within"]
    print(f"{'region':<10s} {'nvox':>5s} {'off%':>6s} | {'classic ICC':>11s} {'r age':>7s} "
          f"| {'refined ICC':>11s} {'r age':>7s} {'var/base':>9s}")
    for nm in VARIANTS:
        cc, rc = f"classic_{nm}", f"refined_{nm}"
        if cc not in d:
            continue
        s = lon.dropna(subset=[cc, rc])
        if len(s) < 20:
            continue
        vc_c = variance_components(s, cc); vc_r = variance_components(s, rc)
        dc = d.dropna(subset=[cc]); dr = d.dropna(subset=[rc])
        r_c = stats.pearsonr(dc.Age, dc[cc])[0]
        r_r = stats.pearsonr(dr.Age, dr[rc])[0]
        print(f"{nm:<10s} {d[f'nvox_{nm}'].mean():5.0f} {100*d[f'off_{nm}'].mean():5.1f}% "
              f"| {vc_c['icc']:11.3f} {r_c:7.3f} | {vc_r['icc']:11.3f} {r_r:7.3f} "
              f"{vc_r['var_within']/base_v:9.2f}")


if __name__ == "__main__":
    main()
