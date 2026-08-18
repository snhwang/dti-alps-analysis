"""
Is there a preferred perpendicular direction for the ALPS numerator to point at?

The reference-frame question has no answer from reproducibility alone. Fixed
scanner axes and tract-derived axes are two definitions, not two estimates of a
common quantity, so neither can be scored against the other. What can be asked
is whether the anatomy the index presumes actually exists.

DTI-ALPS assumes perivascular spaces run perpendicular to both fiber families,
and measures diffusivity along that perpendicular. If that assumption holds,
then within a region the second eigenvector v2, which lies in the plane
perpendicular to the local fiber direction v1, should:

  1. be well defined, which requires lambda2 > lambda3, i.e. non-trivial
     planarity CP = (l2 - l3) / l1. If lambda2 ~ lambda3 the perpendicular plane
     is isotropic and v2 is an arbitrary direction in it.
  2. be coherently oriented across voxels rather than uniformly distributed in
     that plane.
  3. point somewhere near the assumed perivascular axis.

Coherence is measured with axial statistics. Eigenvectors are antipodally
symmetric, so the angle of v2 within the perpendicular plane is doubled before
taking a resultant length R2. R2 near 0 means uniform, near 1 means a single
preferred axis. A uniform null gives R2 ~ sqrt(pi)/(2*sqrt(n)) for n voxels, so
the observed value is reported against that.

Direction is measured two ways for each voxel, both as acute angles:
  to_pvs      angle between v2 and the tract cross-product axis projected into
              the plane perpendicular to that voxel's own v1
  to_x        angle between v2 and scanner x projected into the same plane
Clustering near zero for either says the presumed axis is the real one. Both
near 45 degrees, with low R2, would say the plane has no preferred direction and
the index's numerator is not pointing at anything in particular.

Usage:
    ALPS_TENSOR_SUFFIX=_b1500 python pvs_axis_exists.py --cohort hcpa --limit 80
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

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from direction_estimators import weights_for, principal, align, X, Y, Z
from alps_common import parse_age

HERE = Path(__file__).resolve().parent
DIFF = HERE.parent.parent / "diffusion"
OUT = winpath("Q:/dti_output")
SLAB_MM, FA_MIN = 8.0, 0.2
SHELL = os.environ.get("ALPS_TENSOR_SUFFIX", "")


def axial_R(angles):
    """Resultant length for axial (mod pi) data: double the angle, then resultant."""
    if len(angles) < 3:
        return np.nan
    a = 2 * np.asarray(angles)
    return float(np.hypot(np.cos(a).mean(), np.sin(a).mean()))


def main() -> None:
    import nibabel as nib

    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=["hcpa", "dlbs"], default="hcpa")
    ap.add_argument("--limit", type=int, default=80)
    ap.add_argument("--cp-min", type=float, default=0.15,
                    help="planarity below which v2 is treated as undefined")
    args = ap.parse_args()

    if args.cohort == "hcpa":
        src = pd.read_csv(DIFF / "HCP" / "hcpa_alps_spheres_5mm.csv")
    else:
        src = pd.read_csv(DIFF / "DLBS" / "dlbs_alps_spheres_5mm.csv")
        src["Visit"] = src["Session"]
    src = src[src.status == "ok"].copy()
    src["Age"] = parse_age(src["Age"])
    src = src.dropna(subset=["Age"]).head(args.limit)

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
        zw = Af[2, 0] * ii + Af[2, 1] * jj + Af[2, 2] * kk + Af[2, 3]

        for hemi, side, scr, slf in (("L", xw < 0, 26, 42), ("R", xw > 0, 25, 41)):
            m_p = (sph == 1) & side & (fa >= FA_MIN)
            m_a = (sph == 2) & side & (fa >= FA_MIN)
            if m_p.sum() < 6 or m_a.sum() < 6:
                continue
            z0 = float(np.median(zw[sph > 0])) if (sph > 0).any() else 0.0
            band = np.abs(zw - z0) <= SLAB_MM
            bp = (lab == scr) & (fa >= FA_MIN) & band
            ba = (lab == slf) & (fa >= FA_MIN) & band
            if bp.sum() < 10 or ba.sum() < 10:
                continue

            def unit1(m):
                v = vc[m][:, :, 0]
                n = np.linalg.norm(v, axis=1, keepdims=True); n[n == 0] = 1
                return v / n

            vp = align(principal(unit1(bp), weights_for("cl", {"fa": fa[bp], "evals": ev[bp]})), Z)
            va = align(principal(unit1(ba), weights_for("cl", {"fa": fa[ba], "evals": ev[ba]})), Y)
            pax = np.cross(vp, va); pax /= max(np.linalg.norm(pax), 1e-12)

            for nm, m in (("SCR", m_p), ("SLF", m_a)):
                order = srt[m]
                V = vc[m]
                v1 = np.take_along_axis(V, order[:, None, 0:1], 2)[:, :, 0]
                v2 = np.take_along_axis(V, order[:, None, 1:2], 2)[:, :, 0]
                for a in (v1, v2):
                    a /= np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
                cp = CP[m]
                keep = cp >= args.cp_min
                if keep.sum() < 6:
                    continue
                v1k, v2k = v1[keep], v2[keep]

                def in_plane(ref):
                    """Project a fixed reference into each voxel's perpendicular plane."""
                    r = np.broadcast_to(ref, v1k.shape)
                    q = r - (v1k * r).sum(1, keepdims=True) * v1k
                    n = np.linalg.norm(q, axis=1, keepdims=True)
                    ok = n[:, 0] > 1e-8
                    return q / np.maximum(n, 1e-12), ok

                out = {"Subject_ID": r.Subject_ID, "Visit": r.Visit, "hemi": hemi,
                       "roi": nm, "n": int(keep.sum()),
                       "frac_defined": float(keep.mean()), "CP_mean": float(cp[keep].mean())}
                for tag, ref in (("pvs", pax), ("x", X)):
                    q, ok = in_plane(ref)
                    ang = np.degrees(np.arccos(np.clip(np.abs((v2k * q).sum(1)), 0, 1)))
                    out[f"to_{tag}_median"] = float(np.median(ang[ok]))
                    out[f"to_{tag}_frac_lt30"] = float((ang[ok] < 30).mean())
                # coherence of v2 within the plane, referenced to the projected PVS axis
                q, ok = in_plane(pax)
                w = np.cross(v1k, q)
                th = np.arctan2((v2k * w).sum(1)[ok], (v2k * q).sum(1)[ok])
                out["R2_axial"] = axial_R(th)
                out["R2_null"] = float(np.sqrt(np.pi) / (2 * np.sqrt(max(ok.sum(), 1))))
                rows.append(out)
        if i % 20 == 0:
            print(f"  {i}/{len(src)}", flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(HERE / f"pvs_axis_exists_{args.cohort}{SHELL}.csv", index=False)
    print(f"\n{args.cohort}: {len(d)} region-hemisphere-sessions\n")
    print(f"{'ROI':<5s} {'n vox':>6s} {'CP>=thr':>8s} {'CP':>6s} {'R2':>6s} {'null':>6s} "
          f"{'ang to PVS':>11s} {'<30deg':>7s} {'ang to x':>9s} {'<30deg':>7s}")
    for nm, g in d.groupby("roi"):
        print(f"{nm:<5s} {g.n.mean():6.0f} {100*g.frac_defined.mean():7.1f}% "
              f"{g.CP_mean.mean():6.3f} {g.R2_axial.mean():6.3f} {g.R2_null.mean():6.3f} "
              f"{g.to_pvs_median.mean():10.1f}d {100*g.to_pvs_frac_lt30.mean():6.1f}% "
              f"{g.to_x_median.mean():8.1f}d {100*g.to_x_frac_lt30.mean():6.1f}%")
    print("\n  45 deg and R2 at the null level would mean the perpendicular plane")
    print("  has no preferred direction, so no reference frame can be correct.")


if __name__ == "__main__":
    main()
