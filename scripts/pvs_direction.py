"""Perivascular space orientation from structural MRI, independent of diffusion.

Why this exists
---------------
DTI-ALPS assumes perivascular spaces run along scanner x. This paper replaces
that assumption with an axis measured from the diffusion tensor, and the
question that follows is whether the measured axis is a better description of
the anatomy than the fixed one. Nothing in the diffusion data can settle it. The
second eigenvector is the obvious candidate for a reference, but in the ALPS
regions it is contaminated by the callosal fibers, which also run left to right,
so it points near scanner x for a reason that has nothing to do with
perivascular spaces. Any reference drawn from the tensor inherits the confound
the test is supposed to adjudicate.

This module supplies a reference that does not. It estimates perivascular
orientation from structural contrast alone, so it is independent of the tensor,
of the ALPS index, and of every quantity either candidate axis is built from.

How it works
------------
Perivascular spaces are fluid-filled tubes, bright on T2w and dark on T1w. A
tube is exactly what Hessian eigenanalysis detects: at the right scale, the two
eigenvalues across the tube are large and of one sign while the eigenvalue along
it is near zero. That gives two things at every voxel, a vesselness response
after Frangi, and the tube axis itself as the eigenvector of the
smallest-magnitude eigenvalue. The direction is what we are after, and it comes
free with the detection rather than needing a separate segmentation step.

Directions are pooled with the same estimator the paper uses for tract
directions, the principal eigenvector of a weighted sum of outer products
sum_i w_i u_i u_i^T. Tube axes are sign-ambiguous like eigenvectors, so a vector
mean is not defined for them, whereas the dyadic sum discards sign by
construction. Weights are the vesselness responses.

Frames
------
For HCP Lifespan data no registration is required. Diffusion is delivered under
<ID>/T1w/Diffusion, already in ACPC space, and the structural volumes share that
frame with the same origin and axis directions at a finer voxel size. Masks are
therefore carried between grids by world coordinates alone. resample_mask does
that and refuses to guess if the two affines are not consistent.

Status
------
The estimator is validated and works. On synthetic tubes of known orientation it
recovers the axis to 0.4-2.7 degrees with a coherence of 0.93, and it shows no
pull toward any particular axis, including none toward x.

It is not sufficient for the ALPS regions. Run on HCP-A structural volumes in
the measurement spheres, it returns axes 50 to 90 degrees from both candidate
directions with a coherence of only 0.5 to 0.7, and the same region measured on
T1w and on T2w disagrees by a median of 22 degrees. Random axis pairs would
disagree by 57, so there is real structure being detected, but the quantity to
be resolved here is a few degrees and 22 degrees of measurement noise cannot
resolve it. Raising the response threshold lifts coherence to 0.70 without
moving the axis.

The likely reason is that Frangi vesselness responds to any tubular intensity
pattern, and in homogeneous deep white matter at 0.8 mm most of what clears the
threshold is not a perivascular space. Settling the question needs a segmenter
trained to identify perivascular spaces specifically, so that orientation is
measured only inside structures already established to be perivascular, rather
than a generic filter asked to find them and orient them in one step.

Kept because the direction machinery is the reusable part and is independent of
how the spaces are found. Given a PVS mask from any source, dyadic_axis and
resample_mask give the axis directly.

Usage
-----
    from pvs_direction import pvs_axis, resample_mask

    axis, info = pvs_axis(t2_img, roi_mask_in_t2_grid, polarity="bright")

or from the command line, on any pair of NIfTI files:

    python pvs_direction.py --image T2w.nii.gz --mask roi.nii.gz --polarity bright

The estimator is generic. It takes an image and a mask and returns an axis, so
it is reusable for any question about perivascular orientation, not only this
one.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy import ndimage

# Perivascular spaces in the centrum semiovale are on the order of a millimetre
# across. At the 0.8 mm structural resolution these scales bracket that without
# reaching the width of a vessel or of a white matter bundle.
DEFAULT_SCALES_MM = (0.6, 0.9, 1.3, 1.8)

# Frangi's constants. alpha and beta separate tubes from sheets and from blobs
# and are the values from the original paper. c scales the structureness term
# and is set per image from the response, since it depends on intensity units.
ALPHA = 0.5
BETA = 0.5


def _hessian(vol: np.ndarray, sigma_vox: np.ndarray) -> np.ndarray:
    """Gamma-normalized Hessian at one scale, as a (...,3,3) array.

    The sigma^2 factor makes responses comparable across scales, which is what
    lets the multi-scale maximum pick the scale that fits each structure.
    """
    sm = ndimage.gaussian_filter(vol, sigma_vox, mode="nearest")
    scale = float(np.mean(sigma_vox) ** 2)
    H = np.empty(vol.shape + (3, 3), dtype=np.float32)
    for i in range(3):
        gi = np.gradient(sm, axis=i)
        for j in range(i, 3):
            d = np.gradient(gi, axis=j) * scale
            H[..., i, j] = d
            H[..., j, i] = d
    return H


def _frangi_at_scale(vol, sigma_vox, mask, polarity):
    """Vesselness and tube direction at a single scale, evaluated inside mask.

    Returns (response, directions) with response zero outside mask. Eigenvalues
    are ordered by absolute value, so lam[0] is the along-tube one and its
    eigenvector is the tube axis.
    """
    H = _hessian(vol, sigma_vox)[mask]
    lam, vec = np.linalg.eigh(H)          # ascending by signed value
    order = np.argsort(np.abs(lam), axis=1)
    lam = np.take_along_axis(lam, order, axis=1)
    vec = np.take_along_axis(vec, order[:, None, :], axis=2)

    l1, l2, l3 = lam[:, 0], lam[:, 1], lam[:, 2]
    eps = 1e-12
    # A tube of the wanted polarity has both cross-tube eigenvalues of one sign.
    # Everything else is rejected outright rather than down-weighted.
    if polarity == "bright":
        ok = (l2 < 0) & (l3 < 0)
    elif polarity == "dark":
        ok = (l2 > 0) & (l3 > 0)
    else:
        raise ValueError("polarity must be 'bright' or 'dark'")

    Ra = np.abs(l2) / (np.abs(l3) + eps)                    # plate versus tube
    Rb = np.abs(l1) / (np.sqrt(np.abs(l2 * l3)) + eps)      # blob versus tube
    S = np.sqrt(l1 ** 2 + l2 ** 2 + l3 ** 2)                # structureness
    c = 0.5 * float(S.max()) if S.size and S.max() > 0 else 1.0

    v = ((1.0 - np.exp(-(Ra ** 2) / (2 * ALPHA ** 2)))
         * np.exp(-(Rb ** 2) / (2 * BETA ** 2))
         * (1.0 - np.exp(-(S ** 2) / (2 * c ** 2))))
    v = np.where(ok, v, 0.0)
    return v.astype(np.float32), vec[:, :, 0].astype(np.float32)


def vesselness(vol, zooms, mask, polarity="bright", scales_mm=DEFAULT_SCALES_MM):
    """Multi-scale vesselness and tube direction inside mask.

    At each voxel the scale giving the largest response wins, and that scale's
    eigenvector is kept as the tube axis. Returns (response, directions), both
    indexed by the voxels of mask in C order.
    """
    zooms = np.asarray(zooms, float)[:3]
    vol = np.asarray(vol, np.float32)
    best_v = None
    best_u = None
    for s in scales_mm:
        v, u = _frangi_at_scale(vol, s / zooms, mask, polarity)
        if best_v is None:
            best_v, best_u = v, u
        else:
            take = v > best_v
            best_v = np.where(take, v, best_v)
            best_u = np.where(take[:, None], u, best_u)
    return best_v, best_u


def dyadic_axis(directions, weights):
    """Principal axis of a weighted set of sign-ambiguous directions.

    The same estimator the paper uses for tract directions. Outer products
    discard sign, so no orientation convention has to be imposed first and the
    result cannot depend on the order the voxels arrive in.
    """
    u = np.asarray(directions, float)
    w = np.asarray(weights, float)
    n = np.linalg.norm(u, axis=1)
    good = (n > 0) & np.isfinite(n) & np.isfinite(w) & (w > 0)
    if good.sum() < 3:
        return None, 0.0, int(good.sum())
    u = u[good] / n[good, None]
    w = w[good]
    T = (w[:, None, None] * u[:, :, None] * u[:, None, :]).sum(0) / w.sum()
    lam, vec = np.linalg.eigh(T)
    axis = vec[:, -1]
    # Coherence: how concentrated the directions are about that axis. 1 means a
    # single direction, 1/3 means uniform on the sphere.
    coherence = float(lam[-1] / lam.sum())
    return axis / np.linalg.norm(axis), coherence, int(good.sum())


def resample_mask(mask_img, target_img):
    """Carry a mask onto another grid by world coordinates, nearest neighbour.

    HCP structural and diffusion volumes share the ACPC frame, so this is exact
    up to the grid change and involves no registration. The function checks that
    the two affines are consistent rather than assuming it.
    """
    A_t = np.asarray(target_img.affine, float)
    A_m = np.asarray(mask_img.affine, float)
    R_t = A_t[:3, :3] / np.linalg.norm(A_t[:3, :3], axis=0)
    R_m = A_m[:3, :3] / np.linalg.norm(A_m[:3, :3], axis=0)
    if not np.allclose(R_t, R_m, atol=1e-3):
        raise ValueError("grids are not in a common frame; register first")

    shape = target_img.shape[:3]
    idx = np.indices(shape).reshape(3, -1)
    world = A_t[:3, :3] @ idx + A_t[:3, 3:4]
    vox = np.linalg.solve(A_m[:3, :3], world - A_m[:3, 3:4])
    vox = np.rint(vox).astype(int)
    m = np.asarray(mask_img.dataobj)
    inside = np.all((vox >= 0) & (vox < np.array(m.shape[:3])[:, None]), axis=0)
    out = np.zeros(idx.shape[1], bool)
    flat = np.ravel_multi_index(vox[:, inside], m.shape[:3])
    out[inside] = np.asarray(m).ravel()[flat] > 0
    return out.reshape(shape)


def pvs_axis(image_img, roi_mask, polarity="bright", scales_mm=DEFAULT_SCALES_MM,
             response_pct=80.0):
    """Estimate the perivascular axis inside roi_mask, in world coordinates.

    Only the strongest responses are pooled. Vesselness is defined everywhere,
    including in tissue holding no vessel at all, so keeping every voxel would
    average real tube directions together with noise directions from flat
    regions. response_pct sets the percentile kept.
    """
    vol = np.asanyarray(image_img.dataobj, dtype=np.float32)
    zooms = image_img.header.get_zooms()[:3]
    roi = np.asarray(roi_mask, bool)
    if roi.sum() < 10:
        return None, {"n_roi": int(roi.sum()), "reason": "roi too small"}

    v, u = vesselness(vol, zooms, roi, polarity=polarity, scales_mm=scales_mm)
    if v.size == 0 or not np.any(v > 0):
        return None, {"n_roi": int(roi.sum()), "reason": "no vessel response"}

    thr = np.percentile(v[v > 0], response_pct) if (v > 0).sum() > 10 else 0.0
    keep = v >= max(thr, 1e-9)
    axis_vox, coh, n = dyadic_axis(u[keep], v[keep])
    if axis_vox is None:
        return None, {"n_roi": int(roi.sum()), "reason": "too few responses"}

    # Voxel axes to world. Directions transform by the linear part, and the
    # result has to be renormalized because that part is not orthonormal when
    # the voxels are not isotropic.
    A = np.asarray(image_img.affine, float)[:3, :3]
    axis = A @ axis_vox
    axis = axis / np.linalg.norm(axis)
    if axis[0] < 0:
        axis = -axis          # a sign convention, the axis itself is unsigned
    return axis, {"n_roi": int(roi.sum()), "n_used": n, "coherence": coh,
                  "median_response": float(np.median(v[keep]))}


def angle_between(a, b):
    """Acute angle in degrees between two unsigned axes."""
    a = np.asarray(a, float) / np.linalg.norm(a)
    b = np.asarray(b, float) / np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(abs(float(a @ b)), 0.0, 1.0))))


def main() -> None:
    import nibabel as nib

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--image", required=True, help="structural volume (T2w or T1w)")
    ap.add_argument("--mask", required=True, help="ROI mask, any grid in the same frame")
    ap.add_argument("--polarity", choices=["bright", "dark"], default="bright",
                    help="bright for T2w, dark for T1w")
    ap.add_argument("--response-pct", type=float, default=80.0)
    args = ap.parse_args()

    img = nib.load(args.image)
    roi = resample_mask(nib.load(args.mask), img)
    axis, info = pvs_axis(img, roi, polarity=args.polarity,
                          response_pct=args.response_pct)
    if axis is None:
        print(f"no axis: {info}")
        return
    print(f"PVS axis (world RAS): [{axis[0]:+.4f}, {axis[1]:+.4f}, {axis[2]:+.4f}]")
    print(f"  angle to scanner x: {angle_between(axis, [1, 0, 0]):.2f} deg")
    print(f"  voxels in roi {info['n_roi']}, pooled {info['n_used']}, "
          f"coherence {info['coherence']:.3f}")


if __name__ == "__main__":
    main()
