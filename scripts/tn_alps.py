"""
ALPS variants in trigeminal neuralgia (OpenNeuro ds005713), a patient cohort.

Everything to this point has been healthy-aging data, where no ALPS variant
showed a robust association with anything except age. The published case for the
index rests on patient cohorts, so this is the first test on one: 120 patients
with chronic trigeminal neuralgia against 53 controls, 3T Philips Ingenia,
single shell b=1500 at 2.0 mm isotropic, which matches the HCP-Aging shell.

Two questions, in order of importance:

  1. Do any variants separate patients from controls? If none do, the cohort
     cannot adjudicate between them either.
  2. If they separate, does the orientation-corrected index separate them
     better? That is the claim the whole project needs and has not been able to
     test, because group discrimination is exactly the endpoint the simulated
     positioning artefact was shown to corrupt.

Clinical gradients are also tested within patients: Sindou grade (nerve
compression severity at surgery), pain severity, and disease duration.

The ALPS spheres were never generated for this dataset, so they are warped in
from template space with each session's cached atlas-to-subject warp, the same
operation the DLBS and HCP-A pipelines performed. This writes a new
processed/atlas/sphere_roi/ folder per session and modifies nothing existing.

Follow-up sessions (subject IDs ending 'fu') are excluded: they are post-
surgical, and the baseline contrast is the one the clinical table supports.

Usage:
    python tn_alps.py --warp-only     # generate spheres, no analysis
    python tn_alps.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from data_paths import winpath

import atomic_io  # noqa: F401  writes become atomic on import

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
DIFFREPO = HERE.parent.parent / "diffusion"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(DIFFREPO))
os.environ.setdefault("DTI_OUTPUT_DIR", str(DIFFREPO / "dti_output"))

from estimator_variants import directional_diffusivity as dd
from registration_aligns_tracts import polar_rotation
from direction_estimators import weights_for, principal, align, X, Y, Z

ROOT = Path(winpath("M:/ds005713-derivatives/dti_output/ds005713_preproc"))
PARTICIPANTS = Path(winpath("M:/ds005713-download/participants_v2.0.1.tsv"))
ROI_SRC = winpath("C:/tmp/alps_roi/old_ROIs_JHU_ALPS_5mm_radius")
SLAB_MM, FA_MIN = 8.0, 0.2
# Radius of the native-space sphere drawn at each warped region's centre, in
# millimetres, matching the default in measured_pvs_axis. Zero restores the
# warped mask of the first submission. Both files read the same variable, so
# the placement rule cannot drift between the variants and the comparators.
SPHERE_MM = float(os.environ.get("ALPS_SPHERE_MM", "0"))
BANDS = (8.0, 12.0, 16.0)   # 8 mm was tuned on 1.5 mm data; 2 mm needs more
VARIANTS = (["classic", "cross", "v2_sphere", "ALPS-PAS", "per-voxel", "pv_perp", "anat_x"]
            + [f"{k}_b{int(b)}" for b in BANDS for k in ("cross", "v2_slab")]
            + ["n_slab"])


def warp_spheres(sdir: Path, force: bool = False):
    """Warp the four template spheres into native space; 1 = SCR, 2 = SLF."""
    import nibabel as nib
    import fastapi_diffusion_processor as fdp

    atlas = sdir / "processed" / "atlas"
    warp = atlas / "atlas_to_subject_warp.nii.gz"
    ref = sdir / "processed" / "fa.nii.gz"
    if not (warp.exists() and ref.exists()):
        return None
    out = atlas / "sphere_roi"
    combined = out / "sphere_roi_combined.nii.gz"
    if combined.exists() and not force:
        return combined
    out.mkdir(parents=True, exist_ok=True)
    got = {}
    for nm in ("L_SCR", "R_SCR", "L_SLF", "R_SLF"):
        dst = out / f"{nm}_native.nii.gz"
        if not dst.exists() or force:
            fdp._run_fsl(
                f"applywarp --in={fdp._to_fsl_path(ROI_SRC / (nm + '.nii.gz'))} "
                f"--ref={fdp._to_fsl_path(ref)} --warp={fdp._to_fsl_path(warp)} "
                f"--out={fdp._to_fsl_path(dst)} --interp=nn")
        if not dst.exists():
            return None
        got[nm] = nib.load(str(dst)).get_fdata() > 0.5
    ri = nib.load(str(ref))
    lab = np.zeros(ri.shape[:3], dtype=np.uint8)
    lab[got["L_SCR"] | got["R_SCR"]] = 1
    lab[got["L_SLF"] | got["R_SLF"]] = 2
    nib.save(nib.Nifti1Image(lab, ri.affine), str(combined))
    return combined


def alps_variants(sdir: Path):
    import nibabel as nib

    proc = sdir / "processed"
    sph_p = proc / "atlas" / "sphere_roi" / "sphere_roi_combined.nii.gz"
    lab_p = proc / "atlas" / "jhu_labels_registered.nii.gz"
    if not (sph_p.exists() and lab_p.exists()):
        return None
    try:
        limg = nib.load(str(lab_p)); lab = limg.get_fdata().astype(int)
        sph = nib.load(str(sph_p)).get_fdata().astype(int)
        # HCP-A carries two shells and the paper fits b=1500 only, so its
        # tensors are written with a suffix. measured_pvs_axis reads the same
        # environment variable; honoring it here lets this function serve the
        # aging cohorts as well as the single-shell one it was written for.
        # Empty by default, which is correct for any single-shell cohort.
        shell = os.environ.get("ALPS_TENSOR_SUFFIX", "")
        ev = nib.load(str(proc / f"tensor_eigenvalues{shell}.nii.gz")).get_fdata()
        vc = nib.load(str(proc / f"tensor_eigenvectors{shell}.nii.gz")).get_fdata()
    except Exception:
        return None
    if lab.shape != sph.shape or lab.shape != ev.shape[:3]:
        return None

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
        """Redraw a warped region as a true sphere at its own centre.

        The same rule measured_pvs_axis applies to the variants, repeated here
        because ALPS-PAS and the per-voxel index appear beside them in the same
        tables. If the comparators kept the warped masks while the variants did
        not, those tables would be comparing region placement rather than the
        methods they are meant to compare.
        """
        if radius <= 0 or not m.any():
            return m
        cx, cy, cz = xw[m].mean(), yw[m].mean(), zw[m].mean()
        d2 = (xw - cx) ** 2 + (yw - cy) ** 2 + (zw - cz) ** 2
        return d2 <= radius ** 2

    def evec(m, w):
        V = vc[m]; o = srt[m]
        v = np.take_along_axis(V, o[:, None, w:w + 1], 2)[:, :, 0]
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)

    acc = {k: {} for k in VARIANTS}
    for hemi, side, scr, slf in (("L", xw < 0, 26, 42), ("R", xw > 0, 25, 41)):
        mp = resphere((sph == 1) & side, SPHERE_MM) & (fa >= FA_MIN)
        ma = resphere((sph == 2) & side, SPHERE_MM) & (fa >= FA_MIN)
        if mp.sum() < 4 or ma.sum() < 4:
            continue
        z0 = float(np.median(zw[sph > 0])) if (sph > 0).any() else 0.0
        band = np.abs(zw - z0) <= SLAB_MM
        bp = (lab == scr) & (fa >= FA_MIN) & band
        ba = (lab == slf) & (fa >= FA_MIN) & band
        if bp.sum() < 10 or ba.sum() < 10:
            continue
        acc["n_slab"][hemi] = float(bp.sum() + ba.sum())
        vp = align(principal(evec(bp, 0), weights_for("cl", {"fa": fa[bp], "evals": ev[bp]})), Z)
        va = align(principal(evec(ba, 0), weights_for("cl", {"fa": fa[ba], "evals": ev[ba]})), Y)
        p_cross = np.cross(vp, va); p_cross /= max(np.linalg.norm(p_cross), 1e-12)

        def v2_axis(masks):
            v2 = np.vstack([evec(m, 1) for m in masks])
            w = np.concatenate([CP[m] for m in masks])
            ok = w > 0
            if ok.sum() < 6:
                return None
            a = principal(v2[ok], w[ok])
            return align(a / max(np.linalg.norm(a), 1e-12), X)

        p_sph, p_slab = v2_axis([mp, ma]), v2_axis([bp, ba])
        if p_sph is None or p_slab is None:
            continue

        def alps(p):
            op = np.cross(p, vp); op /= max(np.linalg.norm(op), 1e-12)
            oa = np.cross(p, va); oa /= max(np.linalg.norm(oa), 1e-12)
            return ((dd(ev[mp], vc[mp], p) + dd(ev[ma], vc[ma], p))
                    / (dd(ev[mp], vc[mp], op) + dd(ev[ma], vc[ma], oa)))

        acc["classic"].__setitem__(hemi, (dd(ev[mp], vc[mp], X) + dd(ev[ma], vc[ma], X))
                              / (dd(ev[mp], vc[mp], Y) + dd(ev[ma], vc[ma], Z)))
        acc["cross"].__setitem__(hemi, alps(p_cross))
        # Anatomical left-right from the subject-to-template affine, the same
        # axis in both hemispheres. Absent affine leaves the other variants
        # untouched rather than dropping the session.
        _aff = proc / "atlas" / "subject_to_mni_affine.mat"
        if _aff.exists():
            try:
                _M = np.loadtxt(_aff)
                if _M.shape == (4, 4):
                    _p = polar_rotation(_M[:3, :3]).T @ X
                    _n = np.linalg.norm(_p)
                    if _n > 1e-10:
                        acc["anat_x"].__setitem__(hemi, alps(_p / _n))
            except Exception:
                pass
        acc["v2_sphere"].__setitem__(hemi, alps(p_sph))
        for bw in BANDS:
            bnd = np.abs(zw - z0) <= bw
            bpw = (lab == scr) & (fa >= FA_MIN) & bnd
            baw = (lab == slf) & (fa >= FA_MIN) & bnd
            if bpw.sum() < 10 or baw.sum() < 10:
                continue
            vpw = align(principal(evec(bpw, 0), weights_for("cl", {"fa": fa[bpw], "evals": ev[bpw]})), Z)
            vaw = align(principal(evec(baw, 0), weights_for("cl", {"fa": fa[baw], "evals": ev[baw]})), Y)
            pc = np.cross(vpw, vaw); pc /= max(np.linalg.norm(pc), 1e-12)
            v2w = np.vstack([evec(bpw, 1), evec(baw, 1)])
            ww = np.concatenate([CP[bpw], CP[baw]])
            okw = ww > 0
            if okw.sum() < 6:
                continue
            pw = align(principal(v2w[okw], ww[okw]), X)
            pw /= max(np.linalg.norm(pw), 1e-12)
            for tag, pax, vpu, vau in (("cross", pc, vpw, vaw), ("v2_slab", pw, vpw, vaw)):
                op = np.cross(pax, vpu); op /= max(np.linalg.norm(op), 1e-12)
                oa = np.cross(pax, vau); oa /= max(np.linalg.norm(oa), 1e-12)
                acc[f"{tag}_b{int(bw)}"][hemi] = float(
                    (dd(ev[mp], vc[mp], pax) + dd(ev[ma], vc[ma], pax))
                    / (dd(ev[mp], vc[mp], op) + dd(ev[ma], vc[ma], oa)))

        # ALPS-PAS (Ajouz): lambda2/lambda3 assigned by which eigenvector is
        # more x-aligned. Uses no estimated axes, so it is invariant about x only.
        num, den = [], []
        for m_ in (mp, ma):
            o = srt[m_]
            e2 = np.take_along_axis(ev[m_], o[:, 1:2], 1)[:, 0]
            e3 = np.take_along_axis(ev[m_], o[:, 2:3], 1)[:, 0]
            V = vc[m_]
            idx = np.arange(len(o))
            v2x = np.abs(V[idx, 0, o[:, 1]])
            v3x = np.abs(V[idx, 0, o[:, 2]])
            pick = v2x > v3x
            num.append(np.where(pick, e2, e3).mean())
            den.append(np.where(pick, e3, e2).mean())
        acc["ALPS-PAS"].__setitem__(hemi, float((num[0] + num[1]) / (den[0] + den[1])))

        # Per-voxel greatest perpendicular direction. Taking each voxel's own
        # largest perpendicular diffusivity makes the numerator lambda2 and the
        # denominator lambda3 by definition, so no axis is estimated at all and
        # invariance is trivial: eigenvalues do not change under rotation. This is
        # ALPS-PAS without the scanner-x sorting that costs it invariance.
        acc["pv_perp"].__setitem__(hemi, float(
            (l2[mp].mean() + l2[ma].mean()) / (l3[mp].mean() + l3[ma].mean())))

        # Per-voxel variant in the spirit of LD-ALPS: each voxel's own principal
        # direction crossed with the opposite tract's mean direction.
        pnum, pden = [], []
        for m_, other in ((mp, va), (ma, vp)):
            v1v = evec(m_, 0)
            pv = np.cross(v1v, other)
            nn = np.linalg.norm(pv, axis=1, keepdims=True)
            good = nn[:, 0] > 1e-8
            if not good.any():
                continue
            pv = pv[good] / nn[good]
            ov = np.cross(pv, v1v[good])
            ov /= np.maximum(np.linalg.norm(ov, axis=1, keepdims=True), 1e-12)
            evk, vck = ev[m_][good], vc[m_][good]
            dp_ = np.einsum("nkj,nj->nk", np.transpose(vck, (0, 2, 1)), pv)
            do_ = np.einsum("nkj,nj->nk", np.transpose(vck, (0, 2, 1)), ov)
            pnum.append((evk * dp_ ** 2).sum(axis=1).mean())
            pden.append((evk * do_ ** 2).sum(axis=1).mean())
        if pnum:
            acc["per-voxel"].__setitem__(hemi, float(np.sum(pnum) / np.sum(pden)))
    if not acc["classic"]:
        return None
    out = {}
    for k, hv in acc.items():
        if not hv:
            continue
        for h, v in hv.items():
            out[f"{k}_{h}"] = float(v)
        out[k] = float(np.mean(list(hv.values())))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warp-only", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--followup", action="store_true",
                    help="process the post-surgical sessions instead of "
                         "the baselines, writing tn_alps_followup.csv")
    args = ap.parse_args()

    sessions = sorted(d for d in ROOT.iterdir() if d.is_dir())
    if args.limit:
        sessions = sessions[:args.limit]
    print(f"{len(sessions)} sessions under {ROOT}\n")

    rows, warped, failed = [], 0, 0
    for i, sd in enumerate(sessions, 1):
        meta = sd / "metadata.json"
        if not meta.exists():
            continue
        name = json.load(open(meta)).get("name", "")
        if not name:
            continue
        # Baselines are sub-###, post-surgical follow-ups are sub-###fu.
        if name.endswith("fu") != args.followup:
            continue
        if warp_spheres(sd) is None:
            failed += 1
            continue
        warped += 1
        if args.warp_only:
            if i % 25 == 0:
                print(f"  warped {warped}/{i}", flush=True)
            continue
        v = alps_variants(sd)
        if v:
            rows.append({"BIDS_ID": name, **v})
        if i % 25 == 0:
            print(f"  {i}/{len(sessions)}  usable {len(rows)}", flush=True)

    print(f"\nspheres present for {warped} sessions, {failed} without a usable warp")
    if args.warp_only:
        return

    d = pd.DataFrame(rows).drop_duplicates(subset="BIDS_ID")
    d.to_csv(HERE / ("tn_alps_followup.csv" if args.followup
                    else "tn_alps.csv"), index=False)
    _out = "tn_alps_followup.csv" if args.followup else "tn_alps.csv"
    print(f"{len(d)} sessions with ALPS values -> {_out}")


if __name__ == "__main__":
    main()
