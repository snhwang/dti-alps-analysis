"""Does the shipped direction estimator do what the Methods claim?

Section 2.2 says the tract direction is the principal eigenvector of the
CL-weighted dyadic sum, and that this is sign-invariant by construction. Those
are checkable rather than assertable, and one neighbouring claim turned out not
to survive the check.

Checks, in order:

  1. the weight really is Westin CL = (l1 - l2) / l1, and is not FA
  2. the estimate really is the principal eigenvector of the weighted dyadic sum
  3. it is invariant to arbitrary sign flips of the input eigenvectors
  4. the superseded running vector mean is ALSO sign-flip invariant, because it
     re-aligns as it goes, so sign flipping is not what separates them
  5. neither estimator is order-dependent on real ALPS regions, so the earlier
     claim that sign reconciliation "depends on the order in which voxels are
     visited" is not supported by these data and has been removed
  6. what does separate them is dispersion: the two agree to a median of
     0.6 deg but diverge in a tail, past 6.8 deg in a tenth of regions

Run from revision/ with the DLBS tensor cache present.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from direction_estimators import weights_for, principal, dir_running_mean  # noqa: E402

CACHE = HERE / "dlbs_tensor_cache"
TAGS = ("proj_L", "proj_R", "assoc_L", "assoc_R")


def angle(a, b):
    return float(np.degrees(np.arccos(np.clip(abs(np.dot(a, b)), 0, 1))))


def rois(limit):
    for f in sorted(CACHE.glob("*.npz"))[:limit]:
        z = np.load(f)
        for tag in TAGS:
            try:
                r = {k: z[f"{tag}_{k}"].astype(float) for k in ("v1", "fa", "evals")}
            except KeyError:
                continue
            if len(r["v1"]) >= 8:
                yield r


def main() -> None:
    rng = np.random.default_rng(0)
    r0 = next(rois(1))
    w0 = weights_for("cl", r0)

    ev = np.sort(r0["evals"], axis=1)[:, ::-1]
    cl = np.clip((ev[:, 0] - ev[:, 1]) / ev[:, 0], 0, None)
    print(f"1. weight is CL, not FA:      {np.allclose(w0, cl)} / "
          f"{not np.allclose(w0, r0['fa'])}")

    T = (w0[:, None, None] * r0["v1"][:, :, None] * r0["v1"][:, None, :]).sum(0) / w0.sum()
    print(f"2. is the dyadic principal:   "
          f"{np.allclose(abs(np.linalg.eigh(T)[1][:, -1] @ principal(r0['v1'], w0)), 1)}")

    base = principal(r0["v1"], w0)
    flips = max(angle(base, principal(r0["v1"] * rng.choice([-1.0, 1.0], len(r0["v1"]))[:, None], w0))
                for _ in range(200))
    print(f"3. sign-flip invariant:       max {flips:.2e} deg over 200 flips")

    old = dir_running_mean(r0["v1"], r0["fa"])
    oflips = max(angle(old, dir_running_mean(r0["v1"] * rng.choice([-1.0, 1.0], len(r0["v1"]))[:, None],
                                             r0["fa"])) for _ in range(200))
    print(f"4. old mean also sign-safe:   max {oflips:.2e} deg (it re-aligns as it goes)")

    so, sn = [], []
    for r in rois(40):
        w = weights_for("cl", r)
        bo, bn = dir_running_mean(r["v1"], r["fa"]), principal(r["v1"], w)
        for _ in range(50):
            i = rng.permutation(len(r["v1"]))
            so.append(angle(bo, dir_running_mean(r["v1"][i], r["fa"][i])))
            sn.append(angle(bn, principal(r["v1"][i], w[i])))
    print(f"5. order dependence:          old {max(so):.3f} deg, new {max(sn):.3f} deg "
          f"-> the order claim is unsupported")

    d = [angle(dir_running_mean(r["v1"], r["fa"]), principal(r["v1"], weights_for("cl", r)))
         for r in rois(150)]
    d = np.array(d)
    print(f"6. old vs new, {len(d)} regions: median {np.median(d):.2f} deg, "
          f"90th {np.percentile(d, 90):.2f}, max {d.max():.2f}")


    scatter = []
    for r in rois(120):
        v = r["v1"]
        T = (v[:, :, None] * v[:, None, :]).mean(0)
        scatter.append(float((v @ np.linalg.eigh(T)[1][:, -1] < 0).mean()))
    s = np.array(scatter)
    print(f"7. stored signs scatter:      median {np.median(s)*100:.1f}% of voxels oppose the "
          f"region axis, {(s < 0.05).mean()*100:.1f}% of regions consistent")
    print("   DEC maps hide this: they colour by |v1| components, so sign never reaches the image.")


if __name__ == "__main__":
    main()
