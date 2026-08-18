"""
Two checks on the measured-perpendicular index before it is believed.

1. Rotation invariance. The claim is that averaging v2 keeps invariance because
   v2 rotates with the tensor. That is an argument, not a result. Here the
   tensors are rotated by a known R and the index recomputed. It should be
   unchanged to numerical precision. If it is not, the argument is wrong.

2. What it is actually measuring. If the estimated axis p is close to v2, then
   D(p) is close to lambda2, and the orthogonal denominator axes are close to
   v3, so the index approaches (l2_proj + l2_assoc) / (l3_proj + l3_assoc).
   That is a tensor shape ratio, closely related to planarity, and it would have
   an age association for ordinary microstructural reasons that have nothing to
   do with perivascular fluid. This correlates the index against that ratio to
   see how much of it is simply lambda2 over lambda3.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from data_paths import winpath
from scipy import stats

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from estimator_variants import directional_diffusivity as dd
from direction_estimators import weights_for, principal, align, X, Y, Z
from rotation_study_slab import euler_rotation

HERE = Path(__file__).resolve().parent
DIFF = HERE.parent.parent / "diffusion"
OUT = winpath("Q:/dti_output")
SLAB_MM, FA_MIN = 8.0, 0.2
SHELL = os.environ.get("ALPS_TENSOR_SUFFIX", "_b1500")


def session_values(sd, R=None):
    import nibabel as nib
    limg = nib.load(str(sd / "atlas" / "jhu_labels_registered.nii.gz"))
    lab = limg.get_fdata().astype(int)
    sph = nib.load(str(sd / "atlas" / "sphere_roi" / "sphere_roi_combined.nii.gz")).get_fdata().astype(int)
    ev = nib.load(str(sd / f"tensor_eigenvalues{SHELL}.nii.gz")).get_fdata()
    vc = nib.load(str(sd / f"tensor_eigenvectors{SHELL}.nii.gz")).get_fdata()
    if R is not None:
        vc = np.einsum("ij,...jk->...ik", R, vc)

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

    def evec(m, w):
        V = vc[m]; o = srt[m]
        v = np.take_along_axis(V, o[:, None, w:w + 1], 2)[:, :, 0]
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)

    out = {"v2": [], "shape": []}
    for side, scr, slf in ((xw < 0, 26, 42), (xw > 0, 25, 41)):
        mp = (sph == 1) & side & (fa >= FA_MIN)
        ma = (sph == 2) & side & (fa >= FA_MIN)
        if mp.sum() < 4 or ma.sum() < 4:
            continue
        z0 = float(np.median(zw[sph > 0])) if (sph > 0).any() else 0.0
        band = np.abs(zw - z0) <= SLAB_MM
        bp = (lab == scr) & (fa >= FA_MIN) & band
        ba = (lab == slf) & (fa >= FA_MIN) & band
        if bp.sum() < 10 or ba.sum() < 10:
            continue
        vp = align(principal(evec(bp, 0), weights_for("cl", {"fa": fa[bp], "evals": ev[bp]})), Z)
        va = align(principal(evec(ba, 0), weights_for("cl", {"fa": fa[ba], "evals": ev[ba]})), Y)
        v2 = np.vstack([evec(bp, 1), evec(ba, 1)])
        w = np.concatenate([CP[bp], CP[ba]])
        ok = w > 0
        p = principal(v2[ok], w[ok]); p /= max(np.linalg.norm(p), 1e-12)
        op = np.cross(p, vp); op /= max(np.linalg.norm(op), 1e-12)
        oa = np.cross(p, va); oa /= max(np.linalg.norm(oa), 1e-12)
        num = dd(ev[mp], vc[mp], p) + dd(ev[ma], vc[ma], p)
        den = dd(ev[mp], vc[mp], op) + dd(ev[ma], vc[ma], oa)
        out["v2"].append(num / den)
        out["shape"].append((l2[mp].mean() + l2[ma].mean()) / (l3[mp].mean() + l3[ma].mean()))
    return ({k: float(np.mean(v)) for k, v in out.items() if v}) or None


def _ready(sd):
    return ((sd / "atlas" / "jhu_labels_registered.nii.gz").exists()
            and (sd / "atlas" / "sphere_roi" / "sphere_roi_combined.nii.gz").exists()
            and (sd / f"tensor_eigenvalues{SHELL}.nii.gz").exists()
            and (sd / f"tensor_eigenvectors{SHELL}.nii.gz").exists())


def main() -> None:
    src = pd.read_csv(DIFF / "HCP" / "hcpa_alps_spheres_5mm.csv")
    src = src[src.status == "ok"].head(120)

    print("1. ROTATION INVARIANCE  (index recomputed on rotated tensors)")
    rng = np.random.default_rng(20260810)
    rel = []
    for r in src.head(6).itertuples():
        sd = OUT / r.DTI_Session_ID / "processed"
        if not _ready(sd):
            continue
        a = session_values(sd)
        R = euler_rotation(*(rng.normal(0, 1, 3) * 15))
        b = session_values(sd, R)
        if a and b:
            d = 100 * abs(b["v2"] - a["v2"]) / a["v2"]
            rel.append(d)
            print(f"   {a['v2']:.5f} -> {b['v2']:.5f}   change {d:.3e}%")
    if rel:
        print(f"   max |change| {max(rel):.3e}%  "
              f"({'INVARIANT' if max(rel) < 1e-6 else 'NOT INVARIANT'})")

    print("\n2. IS IT JUST lambda2 / lambda3?")
    rows = []
    for r in src.itertuples():
        sd = OUT / r.DTI_Session_ID / "processed"
        if not _ready(sd):
            continue
        v = session_values(sd)
        if v:
            rows.append(v)
    d = pd.DataFrame(rows)
    if len(d) > 5:
        rho = stats.pearsonr(d["v2"], d["shape"])[0]
        print(f"   n={len(d)}   r(index, (l2_p+l2_a)/(l3_p+l3_a)) = {rho:+.3f}")
        print("   near 1 would mean the index is a tensor shape ratio by another name.")


if __name__ == "__main__":
    main()
