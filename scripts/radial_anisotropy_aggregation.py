"""Is pv_perp the radial anisotropy, or one particular summary of it?

pv_perp is reported as the voxelwise measured-axis variant, and it equals
lambda2/lambda3 in the sense that no axis is estimated. But the number it
reports is

    (mean lambda2 over SCR + mean lambda2 over SLF)
    -----------------------------------------------
    (mean lambda3 over SCR + mean lambda3 over SLF)

which is a ratio of means. The radial anisotropy of a voxel is lambda2/lambda3,
and the natural summary of that over a region is the mean of the voxelwise
ratios. Those two are not equal.

The sign of the difference is not guaranteed, which is worth stating carefully
because it is easy to assert otherwise. Jensen gives E[1/l3] >= 1/E[l3], but the
quantity here is E[l2/l3] against E[l2]/E[l3], a ratio of two correlated
variables. The difference depends on the covariance between l2 and 1/l3 as well
as on the curvature, so it can fall either way. In these data it is positive on
average and negative in some sessions.

So the question is not which is correct, both are defensible summaries, but how
far apart they are. If the gap is negligible the two names describe one number.
If it is not, then "pv_perp is the radial anisotropy" holds only for the pooled
summary, and a paper that computed radial anisotropy the other way would get a
different value.

Three summaries are compared per session:

    pooled      ratio of means, what pv_perp reports
    voxelwise   mean of the per-voxel lambda2/lambda3
    median      median per-voxel ratio, as a robustness check on the tail

Uses the ds001907 derivative trees, which already carry tensor eigenvalues and
the warped ALPS spheres.

    python radial_anisotropy_aggregation.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
WORK = Path(r"M:\ds001907-derivatives")
FA_MIN = 0.2


def session_values(d: Path):
    proc = d / "processed"
    ev_p = proc / "tensor_eigenvalues.nii.gz"
    sph_p = proc / "atlas" / "sphere_roi" / "sphere_roi_combined.nii.gz"
    if not (ev_p.exists() and sph_p.exists()):
        return None
    ev = nib.load(str(ev_p)).get_fdata()
    sph = nib.load(str(sph_p)).get_fdata().astype(int)
    l1, l2, l3 = ev[..., 0], ev[..., 1], ev[..., 2]
    md = ev.mean(-1)
    nu = np.sqrt(((ev - md[..., None]) ** 2).sum(-1))
    de = np.sqrt((ev ** 2).sum(-1))
    fa = np.clip(np.sqrt(1.5) * np.divide(nu, de, out=np.zeros_like(nu),
                                          where=de != 0), 0, 1)
    ok = (fa >= FA_MIN) & (l3 > 0)
    mp, ma = (sph == 1) & ok, (sph == 2) & ok       # SCR projection, SLF assoc
    if mp.sum() < 20 or ma.sum() < 20:
        return None
    pooled = ((l2[mp].mean() + l2[ma].mean()) / (l3[mp].mean() + l3[ma].mean()))
    vox = np.concatenate([l2[mp] / l3[mp], l2[ma] / l3[ma]])
    return {"pooled": float(pooled), "voxelwise": float(vox.mean()),
            "median": float(np.median(vox)), "n_vox": int(vox.size),
            "cv_l3": float(np.concatenate([l3[mp], l3[ma]]).std()
                           / np.concatenate([l3[mp], l3[ma]]).mean())}


def main() -> None:
    argparse.ArgumentParser().parse_args()
    rows = []
    for sub in sorted(p for p in WORK.iterdir() if p.is_dir()):
        for ses in sorted(p for p in sub.iterdir() if p.is_dir()):
            v = session_values(ses)
            if v:
                rows.append({"subject": sub.name, "session": ses.name, **v})
    d = pd.DataFrame(rows)
    if d.empty:
        print("no sessions with eigenvalues and spheres")
        return
    d["gap"] = d.voxelwise - d.pooled
    d["gap_pct"] = 100 * d.gap / d.pooled
    d.to_csv(HERE / "radial_anisotropy_aggregation.csv", index=False)

    print(f"{len(d)} sessions\n")
    for c in ("pooled", "voxelwise", "median"):
        print(f"   {c:<10s} mean {d[c].mean():.3f}   SD {d[c].std():.3f}   "
              f"range {d[c].min():.3f} to {d[c].max():.3f}")
    print(f"\n   Jensen gap, voxelwise mean minus pooled ratio:")
    print(f"      {d.gap.mean():+.3f} absolute, {d.gap_pct.mean():+.1f} percent "
          f"of the pooled value")
    print(f"      always positive: {bool((d.gap > 0).all())}")
    print(f"      lambda3 coefficient of variation {d.cv_l3.mean():.2f}, "
          f"which is what drives the gap")

    r, p = stats.pearsonr(d.pooled, d.voxelwise)
    print(f"\n   correlation between the two summaries across sessions "
          f"r={r:.4f} (p={p:.1e})")
    print("\n" + "=" * 66)
    if r > 0.95 and abs(d.gap_pct.mean()) < 5:
        print("Same number to within a few percent, and they rank sessions")
        print("identically. The two names describe one quantity.")
    elif r > 0.95:
        print("They rank sessions almost identically, so any association with a")
        print("phenotype will be nearly the same, but the absolute values differ")
        print(f"by {d.gap_pct.mean():.0f} percent. A reported radial anisotropy is")
        print("only comparable to pv_perp if it was pooled the same way.")
    else:
        print("The two summaries are not interchangeable, in level or in rank.")
    print(f"\n   wrote {HERE / 'radial_anisotropy_aggregation.csv'}")


if __name__ == "__main__":
    main()
