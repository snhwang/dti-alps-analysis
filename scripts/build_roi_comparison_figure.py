"""
Hand-drawn against atlas-placed regions, same session, same slice.

The figure this replaces showed regions drawn by hand, which is not how any
analysis in this paper was done, and its own caption said so. Since the
manual-versus-automated comparison is retained as a result, the honest figure
shows both on the same brain.

The session is chosen to be representative rather than flattering: among those
carrying both region sets, the one whose manual and automated index values sit
closest to their respective cohort medians. A session picked for agreement would
misrepresent how well the two correspond.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from data_paths import winpath

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
DIFF = HERE.parent.parent / "diffusion"
OUT = winpath("Q:/dti_output")
EXCLUDE = {"session_20260122_160723"}      # damaged manual mask
SHELL = ""                                  # DLBS is fitted to b=1000 alone
PAD = 20                                    # crop margin around the regions, voxels
FA_MIN = 0.2                                # the threshold every analysis applies

C_PROJ = "#4da3ff"      # projection regions
C_ASSOC = "#ffd24d"     # association regions


def dec_rgb(fa, evecs, affine):
    """Directionally encoded colour: |v1| in world coordinates, scaled by FA."""
    v1 = evecs[..., :, 0]
    R = affine[:3, :3]
    R = R / np.maximum(np.linalg.norm(R, axis=0, keepdims=True), 1e-12)
    v1w = np.einsum("ij,...j->...i", R, v1)
    return np.clip(np.abs(v1w) * fa[..., None], 0, 1)


def pick_session():
    """DLBS sessions carrying both region sets.

    HCP-A gives sharper images but its hand-drawn regions cannot be trusted:
    in 21 of 36 sessions the file recorded as manual holds a programmatic
    placement instead. DLBS has 78 sessions, none affected, and it is the cohort
    whose manual-versus-automated comparison the paper reports.
    """
    src = pd.read_csv(DIFF / "HCP" / "lifespan_alps_results.csv")
    src = src.dropna(subset=["DTI_Session_ID"])
    ok = []
    for r in src.itertuples():
        sid = str(r.DTI_Session_ID)
        if sid in EXCLUDE:
            continue
        sd = OUT / sid / "processed"
        need = ("alps_rois_manual.nii.gz", "fa.nii.gz",
                f"tensor_eigenvectors{SHELL}.nii.gz",
                f"tensor_eigenvalues{SHELL}.nii.gz")
        if all((sd / q).exists() for q in need) \
           and (sd / "atlas" / "sphere_roi" / "sphere_roi_combined.nii.gz").exists() \
           and is_hand_drawn(sd / "alps_rois_manual.nii.gz") \
           and has_four(sd / "alps_rois_manual.nii.gz"):
            ok.append(sid)
    if not ok:
        raise RuntimeError("no session carries both region sets")
    # Selected for placement quality, not for agreement between the two region
    # sets. The figure shows where the regions belong, and hand-drawn placement
    # varies enough that a session taken at random illustrates the variability
    # instead. Off-tract fraction is judged by diffusion direction rather than by
    # the atlas, so the criterion does not favour the atlas-placed regions. The
    # caption states that this is a well-placed example, and the variability is
    # reported numerically in the text.
    # off-tract quality, from manual_roi_offtract.py. The previous file,
    # manual_roi_offtract_dlbs.csv, had no generator anywhere in the tree.
    quality = pd.read_csv(HERE / "manual_roi_offtract.csv").set_index("sid")
    best, score = None, np.inf
    for sid in ok:
        if sid not in quality.index:
            continue
        # placement quality, plus a penalty for showing the spheres off-equator
        d = (float(quality.loc[sid, "worst"])
             + 0.05 * centre_offset(OUT / sid / "processed"))
        if d < score:
            best, score = sid, d
    return best or ok[0], len(ok)



def has_four(path: Path) -> bool:
    import nibabel as nib
    try:
        im = nib.load(str(path))
        return len(four_region_slices(np.rint(im.get_fdata()).astype(int), im.affine)) > 0
    except Exception:
        return False


def four_region_slices(man, affine):
    """Slices carrying all four hand-drawn regions: two tracts, two hemispheres.

    Raters drew each hemisphere on whatever slice suited it, so the sides do not
    always coincide. Where they do not, one slice cannot show the placement.
    """
    ii, jj, kk = np.indices(man.shape)
    xw = (affine[0, 0] * ii + affine[0, 1] * jj
          + affine[0, 2] * kk + affine[0, 3])
    out = []
    for k in range(man.shape[2]):
        s = man[:, :, k]
        if not (s > 0).any():
            continue
        got = sum(1 for v in (1, 2) for side in (xw[:, :, k] < 0, xw[:, :, k] > 0)
                  if ((s == v) & side).sum() >= 3)
        if got == 4:
            out.append(k)
    return out



def centre_offset(sd: Path) -> int:
    """Slices between the hand-drawn regions and the atlas spheres' equator.

    A sphere sampled away from its central slice yields a small ragged
    cross-section, so a figure drawn there misrepresents the placement.
    """
    import nibabel as nib
    try:
        mi = nib.load(str(sd / "alps_rois_manual.nii.gz"))
        man = np.rint(mi.get_fdata()).astype(int)
        sph = np.rint(nib.load(str(sd / "atlas" / "sphere_roi"
                                    / "sphere_roi_combined.nii.gz")).get_fdata()).astype(int)
    except Exception:
        return 99
    good = four_region_slices(man, mi.affine)
    if not good:
        return 99
    k_sph = int(np.argmax([(sph[:, :, k] > 0).sum() for k in range(sph.shape[2])]))
    return int(min(abs(k - k_sph) for k in good))


def is_hand_drawn(path: Path) -> bool:
    """False when every region is a perfectly filled cube.

    A rater cannot draw a filled 3x3x3 cube four times; a placement routine can
    do nothing else. This distinguishes genuine hand-drawn regions from a
    programmatic placement that was snapshotted under the manual filename.
    """
    import nibabel as nib
    try:
        m = np.rint(nib.load(str(path)).get_fdata()).astype(int)
    except Exception:
        return False
    blobs, cubes = 0, 0
    for val in (1, 2):
        v = m == val
        if not v.any():
            continue
        xs = np.where(v.any(axis=(1, 2)))[0]
        mid = (xs.min() + xs.max()) // 2
        for sel in (slice(None, mid + 1), slice(mid + 1, None)):
            s = np.zeros_like(v)
            s[sel] = v[sel]
            if not s.any():
                continue
            idx = np.argwhere(s)
            bb = idx.max(0) - idx.min(0) + 1
            blobs += 1
            if s.sum() == np.prod(bb) and len(set(bb)) == 1:
                cubes += 1
    return blobs > 0 and cubes == 0


def ensure_small(sd: Path):
    """The 2.5 mm regions in this session's space, warped if not already there."""
    import subprocess
    out = sd / "atlas" / "rois_2p5mm_native.nii.gz"
    if out.exists():
        return out
    src = HERE / "rois_2p5mm" / "rois_2p5mm.nii.gz"
    warp = sd / "atlas" / "atlas_to_subject_warp.nii.gz"
    if not (src.exists() and warp.exists()):
        return None

    def f(p):
        p = str(p).replace("\\", "/")
        return f"/mnt/{p[0].lower()}{p[2:]}" if len(p) > 1 and p[1] == ":" else p
    cmd = (f"applywarp --in={f(src)} --ref={f(sd / 'fa.nii.gz')} "
           f"--warp={f(warp)} --out={f(out)} --interp=nn")
    r = subprocess.run(f'wsl -e bash -lc "{cmd}"', shell=True,
                       capture_output=True, text=True, timeout=600)
    return out if (r.returncode == 0 and out.exists()) else None


def outline(ax, mask2d, colour, lw=1.4, dashed=False):
    """Draw the boundary of a binary mask without filling it."""
    if not mask2d.any():
        return
    ax.contour(mask2d.T, levels=[0.5], colors=[colour], linewidths=lw,
               linestyles=[(0, (4, 2)) if dashed else "solid"])


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import nibabel as nib

    sid, n_avail = pick_session()
    sd = OUT / sid / "processed"
    # The same single shell the analyses use. Each session directory also
    # holds an unsuffixed tensor fitted across b=1500 and b=3000 together,
    # which the paper does not use because it underestimates diffusivity and
    # inflates within-participant variance, so the figure must not use it
    # either. No FA map is stored per shell, so FA is recomputed from the
    # b=1500 eigenvalues rather than read from the multi-shell fa.nii.gz.
    fa_img = nib.load(str(sd / "fa.nii.gz"))          # geometry reference only
    evals = nib.load(str(sd / f"tensor_eigenvalues{SHELL}.nii.gz")).get_fdata()
    md = evals.mean(-1)
    num = np.sqrt(((evals - md[..., None]) ** 2).sum(-1))
    den = np.sqrt((evals ** 2).sum(-1))
    fa = np.clip(np.sqrt(1.5) * np.divide(num, den, out=np.zeros_like(num),
                                          where=den != 0), 0, 1)
    evecs = nib.load(str(sd / f"tensor_eigenvectors{SHELL}.nii.gz")).get_fdata()
    man = np.rint(nib.load(str(sd / "alps_rois_manual.nii.gz")).get_fdata()).astype(int)
    sph = np.rint(nib.load(str(sd / "atlas" / "sphere_roi" / "sphere_roi_combined.nii.gz"))
                  .get_fdata()).astype(int)
    small = ensure_small(sd)
    sm = np.rint(nib.load(str(small)).get_fdata()).astype(int) if small else None

    # One slice for both panels, so the anatomy is identical and the two
    # placements can be compared directly. Drawing each at its own best slice
    # would make the panels non-comparable. The slice maximises the smaller of
    # the two voxel counts, which keeps both sets well represented instead of
    # favouring whichever is more compact: hand-drawn regions occupy one to three
    # slices, while the warped spheres span about eight and still carry most of
    # their voxels at that level.
    candidates = four_region_slices(man, fa_img.affine) or list(range(man.shape[2]))
    # among slices carrying all four hand-drawn regions, take the one closest to
    # the spheres' central slice, so neither set is shown near its edge
    k_sph_peak = int(np.argmax([(sph[:, :, k] > 0).sum() for k in range(sph.shape[2])]))
    k_common = min(candidates, key=lambda k: (abs(k - k_sph_peak), -int((man[:, :, k] > 0).sum())))
    k_man = k_sph = k_common

    # a common crop, so the two panels are directly comparable in scale
    xs, ys = np.where((man > 0).any(-1) | (sph > 0).any(-1))
    x0, x1 = max(xs.min() - PAD, 0), min(xs.max() + PAD + 1, fa.shape[0])
    y0, y1 = max(ys.min() - PAD, 0), min(ys.max() + PAD + 1, fa.shape[1])

    # size the figure to the crop, so the panels carry no dead space
    aspect = (y1 - y0) / (x1 - x0)
    width = 7.0
    # Two panels. A 2.5 mm panel was tried and dropped: at that radius the
    # projection and association regions sometimes occupy no common axial slice,
    # so it cannot be relied on to show both. The radius comparison is reported
    # numerically in the robustness section instead.
    panels = [(man, k_man, "(a) hand-drawn", (1, 2)),
              (sph, k_sph, r"(b) atlas-placed", (1, 2))]
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(width, width / n * aspect + 0.85))
    for ax, (lab, k, title, (pv, av)) in zip(np.atleast_1d(axes), panels):
        rgb = dec_rgb(fa[:, :, k], evecs[:, :, k], fa_img.affine)
        ax.imshow(np.transpose(rgb[x0:x1, y0:y1], (1, 0, 2)), origin="lower",
                  interpolation="nearest")
        # White against every DEC hue; the two regions differ by line style
        # rather than colour, since red, green, blue and their mixtures are all
        # already in use by the direction encoding.
        # Outline the voxels the analyses actually use. Every region in this
        # paper is restricted to FA >= 0.2, so drawing the raw mask would show
        # more tissue than is ever measured.
        sub = np.where(fa[x0:x1, y0:y1, k] >= FA_MIN, lab[x0:x1, y0:y1, k], 0)
        # the 2.5 mm image labels hemispheres separately (1,2 projection;
        # 3,4 association), the others use 1 projection and 2 association
        proj = np.isin(sub, [1, 2]) if av == 3 else (sub == 1)
        assoc = np.isin(sub, [3, 4]) if av == 3 else (sub == 2)
        outline(ax, proj, "white", lw=1.5)
        outline(ax, assoc, "white", lw=1.5, dashed=True)
        ax.set_title(title, loc="left", color="#222222", fontsize=10, pad=4)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    handles = [plt.Line2D([], [], color="0.25", lw=1.6, label="projection (SCR)"),
               plt.Line2D([], [], color="0.25", lw=1.6, ls=(0, (4, 2)),
                          label="association (SLF)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 0.005))
    fig.subplots_adjust(wspace=0.03, bottom=0.11, top=0.92, left=0.01, right=0.99)

    out = HERE.parent / "fig_roi_manual_vs_auto.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"session {sid}, chosen from {n_avail} carrying both")
    print(f"  common slice {k_common}")
    print(f"    hand-drawn: proj {int((man[:,:,k_common]==1).sum())}, "
          f"assoc {int((man[:,:,k_common]==2).sum())} voxels")
    print(f"    atlas:      proj {int((sph[:,:,k_common]==1).sum())}, "
          f"assoc {int((sph[:,:,k_common]==2).sum())} voxels "
          f"(peak {max(int((sph[:,:,k]==1).sum()) for k in range(sph.shape[2]))}, "
          f"{max(int((sph[:,:,k]==2).sum()) for k in range(sph.shape[2]))})")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
