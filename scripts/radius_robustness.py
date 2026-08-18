"""
Does anything change if the measurement spheres are Taoka's size?

The regions used throughout are the 5 mm radius spheres distributed with the
Barisano reference implementation. That implementation has since made 2.5 mm the
default, closer to the 5 mm diameter circle of the original description, and
either is a defensible choice: the larger sphere averages more voxels, the
smaller one fits the narrow tracts with less spill into neighbouring tissue.

Since the conclusions should not depend on which was picked, this rebuilds the
spheres at 2.5 mm about the same centres, warps them into each native space with
the cached atlas transform, and recomputes the endpoints the paper rests on:
reliability, the age association, region composition, and in DLBS the share of
the age coefficient carried by head position.

    python radius_robustness.py --cohort dlbs
    python radius_robustness.py --cohort hcpa --limit 300
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from data_paths import winpath, refined_rois

import atomic_io  # noqa: F401  writes become atomic on import

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from estimator_variants import directional_diffusivity as dd, variance_components
from direction_estimators import principal, align, X, Y, Z

DIFF = HERE.parent.parent / "diffusion"
OUT = winpath("Q:/dti_output")
TEMPLATE_ROIS = refined_rois()
SMALL = HERE / "rois_2p5mm"
CENTRES = {"L_SCR": (-26, -16, 27), "R_SCR": (26, -16, 27),
           "L_SLF": (-38, -16, 27), "R_SLF": (38, -16, 27)}
LABELS = {"L_SCR": 1, "R_SCR": 2, "L_SLF": 3, "R_SLF": 4}
FA_MIN, SLAB_MM, RADIUS = 0.2, 8.0, 2.5


def to_fsl(p) -> str:
    p = str(p).replace("\\", "/")
    return f"/mnt/{p[0].lower()}{p[2:]}" if len(p) > 1 and p[1] == ":" else p


def run_fsl(cmd: str) -> None:
    r = subprocess.run(f'wsl -e bash -lc "{cmd}"', shell=True,
                       capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip()[:300])


def build_small_spheres() -> Path:
    """One label image at 2.5 mm radius, on the same grid as the shipped ROIs."""
    import nibabel as nib
    SMALL.mkdir(exist_ok=True)
    out = SMALL / "rois_2p5mm.nii.gz"
    if out.exists():
        return out
    ref = nib.load(str(TEMPLATE_ROIS / "L_SCR.nii.gz"))
    A = ref.affine
    ii, jj, kk = np.indices(ref.shape)
    w = np.stack([A[a, 0] * ii + A[a, 1] * jj + A[a, 2] * kk + A[a, 3] for a in range(3)])
    lab = np.zeros(ref.shape, np.uint8)
    for name, c in CENTRES.items():
        d2 = sum((w[a] - c[a]) ** 2 for a in range(3))
        lab[d2 <= RADIUS ** 2] = LABELS[name]
    nib.save(nib.Nifti1Image(lab, A), str(out))
    for name, v in LABELS.items():
        print(f"  {name}: {int((lab == v).sum())} voxels at {RADIUS} mm radius")
    return out


def sessions(cohort: str, limit: int | None):
    if cohort == "hcpa":
        src = pd.read_csv(DIFF / "HCP" / "hcpa_alps_spheres_5mm.csv")
        shell = "_b1500"
    else:
        src = pd.read_csv(DIFF / "DLBS" / "dlbs_alps_spheres_5mm.csv")
        shell = ""
    src = src[src.status == "ok"] if "status" in src else src
    if "Visit" not in src.columns and "Session" in src.columns:
        src = src.rename(columns={"Session": "Visit"})     # DLBS names it Session
    if cohort == "hcpa":
        counts = src.Subject_ID.value_counts()
        src = src[src.Subject_ID.isin(counts[counts >= 2].index)]
    if limit:
        keep = sorted(src.Subject_ID.unique())[:limit]
        src = src[src.Subject_ID.isin(keep)]
    return src.sort_values(["Subject_ID", "Visit"]), shell


def main() -> None:
    import nibabel as nib
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=["hcpa", "dlbs"], default="dlbs")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    print(f"building {RADIUS} mm spheres")
    small = build_small_spheres()
    src, shell = sessions(args.cohort, args.limit)
    print(f"\n{args.cohort}: {len(src)} sessions, {src.Subject_ID.nunique()} participants\n")

    rows = []
    for i, r in enumerate(src.itertuples(), 1):
        sd = OUT / r.DTI_Session_ID / "processed"
        warp = sd / "atlas" / "atlas_to_subject_warp.nii.gz"
        lab_p = sd / "atlas" / "jhu_labels_registered.nii.gz"
        fa_p, ev_p = sd / "fa.nii.gz", sd / f"tensor_eigenvalues{shell}.nii.gz"
        vc_p = sd / f"tensor_eigenvectors{shell}.nii.gz"
        if not all(p.exists() for p in (warp, lab_p, fa_p, ev_p, vc_p)):
            continue
        native = sd / "atlas" / "rois_2p5mm_native.nii.gz"
        if not native.exists():
            try:
                run_fsl(f"applywarp --in={to_fsl(small)} --ref={to_fsl(fa_p)} "
                        f"--warp={to_fsl(warp)} --out={to_fsl(native)} --interp=nn")
            except Exception:
                continue
        try:
            limg = nib.load(str(lab_p)); lab = np.rint(limg.get_fdata()).astype(int)
            sph = np.rint(nib.load(str(native)).get_fdata()).astype(int)
            ev = nib.load(str(ev_p)).get_fdata()
            vc = nib.load(str(vc_p)).get_fdata()
        except Exception:
            continue

        srt = np.argsort(ev, axis=-1)[..., ::-1]
        l1 = np.take_along_axis(ev, srt[..., 0:1], -1)[..., 0]
        l2 = np.take_along_axis(ev, srt[..., 1:2], -1)[..., 0]
        md = ev.mean(-1)
        nu = np.sqrt(((ev - md[..., None]) ** 2).sum(-1))
        de = np.sqrt((ev ** 2).sum(-1))
        fa = np.clip(np.sqrt(1.5) * np.divide(nu, de, out=np.zeros_like(nu), where=de != 0), 0, 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            CL = np.where(l1 > 0, (l1 - l2) / l1, 0.0)

        ii, jj, kk = np.indices(lab.shape); Af = limg.affine
        zw = Af[2, 0] * ii + Af[2, 1] * jj + Af[2, 2] * kk + Af[2, 3]
        z0 = float(np.median(zw[sph > 0])) if (sph > 0).any() else 0.0
        band = np.abs(zw - z0) <= SLAB_MM

        def evec(m, which):
            V = vc[m]; o = srt[m]
            v = np.take_along_axis(V, o[:, None, which:which + 1], 2)[:, :, 0]
            return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)

        acc, offs = {"classic": [], "refined": []}, []
        for scr_l, slf_l, jscr, jslf in ((1, 3, 26, 42), (2, 4, 25, 41)):
            mp = (sph == scr_l) & (fa >= FA_MIN)
            ma = (sph == slf_l) & (fa >= FA_MIN)
            if mp.sum() < 4 or ma.sum() < 4:
                continue
            dp = (lab == jscr) & (fa >= FA_MIN) & band
            da = (lab == jslf) & (fa >= FA_MIN) & band
            if dp.sum() < 10 or da.sum() < 10:
                continue
            vp = align(principal(evec(dp, 0), CL[dp]), Z)
            va = align(principal(evec(da, 0), CL[da]), Y)
            p = np.cross(vp, va); p /= max(np.linalg.norm(p), 1e-12)
            op = np.cross(p, vp); op /= max(np.linalg.norm(op), 1e-12)
            oa = np.cross(p, va); oa /= max(np.linalg.norm(oa), 1e-12)
            Pe, Pv, Ae, Av = ev[mp], vc[mp], ev[ma], vc[ma]
            acc["classic"].append((dd(Pe, Pv, X) + dd(Ae, Av, X))
                                  / (dd(Pe, Pv, Y) + dd(Ae, Av, Z)))
            acc["refined"].append((dd(Pe, Pv, p) + dd(Ae, Av, p))
                                  / (dd(Pe, Pv, op) + dd(Ae, Av, oa)))
            # association-region tissue pointing away from its own tract
            v1a = evec(ma, 0)
            offs.append(float((np.abs(v1a @ va) < np.cos(np.radians(45))).mean()))
        if not acc["classic"]:
            continue
        rows.append({"Subject_ID": r.Subject_ID, "Visit": r.Visit, "Age": r.Age,
                     "classic": float(np.mean(acc["classic"])),
                     "refined": float(np.mean(acc["refined"])),
                     "slf_off_tract": float(np.mean(offs)),
                     "n_scr": int((sph == 1).sum() + (sph == 2).sum()),
                     "n_slf": int((sph == 3).sum() + (sph == 4).sum())})
        if i % 25 == 0:
            print(f"  {i}/{len(src)}", flush=True)

    d = pd.DataFrame(rows)
    d["Age"] = pd.to_numeric(d.Age, errors="coerce")
    out = HERE / f"radius_robustness_{args.cohort}.csv"
    d.to_csv(out, index=False)
    print(f"\n{len(d)} sessions computed at {RADIUS} mm radius")
    if len(d) > 20:
        from scipy import stats
        lon = d[d.Subject_ID.isin(d.Subject_ID.value_counts()[lambda s: s >= 2].index)]
        for c in ("classic", "refined"):
            icc = variance_components(lon.dropna(subset=[c]), c)["icc"] if len(lon) > 20 else np.nan
            s = d.dropna(subset=[c, "Age"])
            print(f"  {c:<9s} ICC {icc:.3f}   age r {stats.pearsonr(s.Age, s[c])[0]:+.3f}   "
                  f"mean {d[c].mean():.3f}")
        print(f"  association off-tract fraction, median {d.slf_off_tract.median()*100:.1f}%")
        print(f"  region size, median {d.n_scr.median():.0f} and {d.n_slf.median():.0f} voxels")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
