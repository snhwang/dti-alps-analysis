"""What does the two-region design buy once the index reduces to the ratio?

The classic index needs two regions for a geometric reason. It measures along
fixed scanner axes, so it needs two tracts running in different directions,
projection near z and association near y, both perpendicular to x. That is what
makes x a candidate perivascular direction in both places, and it is the whole
justification for pairing them.

The voxelwise measured axis has no axes. It is

    (<l2>_proj + <l2>_assoc) / (<l3>_proj + <l3>_assoc)

which is a l3-weighted average of the two regions' radial anisotropies. The
geometric reason for pairing them is gone, so the pairing has to earn its place
some other way, or not. Three things are asked here.

  Does the composite beat either region alone against age? If one region tracks
  age better than the pair, averaging is costing signal rather than adding it.

  Are the two regions redundant? Two measures of the same tissue property in
  different white matter should correlate. How much they do bounds how much the
  second region can add.

  What is the weight? If w sits near a half the composite is a plain average,
  and if it does not the pairing silently favours one region.

Reliability matters here because a noisier measure correlates worse for reasons
that are not about signal, so the age correlations are also reported
disattenuated by each measure's own ICC across repeat visits.

    python two_roi_value.py --cohort hcpa
    python two_roi_value.py --cohort dlbs
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent

SOURCES = {
    "hcpa": "measured_pvs_axis_hcpa_b1500_all.csv",
    "dlbs": "measured_pvs_axis_dlbs.csv",
}


def icc11(d: pd.DataFrame, col: str) -> float:
    """ICC(1,1) across repeat visits, one-way random effects."""
    g = d.dropna(subset=[col]).groupby("Subject_ID")[col]
    sizes = g.size()
    keep = sizes[sizes > 1].index
    if len(keep) < 10:
        return float("nan")
    x = d[d.Subject_ID.isin(keep)].dropna(subset=[col])
    grand = x[col].mean()
    means = x.groupby("Subject_ID")[col].mean()
    n = x.groupby("Subject_ID").size()
    k = n.mean()
    msb = (n * (means - grand) ** 2).sum() / (len(means) - 1)
    msw = ((x[col] - x.Subject_ID.map(means)) ** 2).sum() / (len(x) - len(means))
    return float((msb - msw) / (msb + (k - 1) * msw))


def r_age(d: pd.DataFrame, col: str) -> tuple[float, float, int]:
    x = d.dropna(subset=[col, "Age"])
    r, p = stats.pearsonr(x[col], x.Age)
    return float(r), float(p), len(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="hcpa", choices=sorted(SOURCES))
    args = ap.parse_args()

    src = HERE / SOURCES[args.cohort]
    d = pd.read_csv(src)
    need = ["l2_proj", "l3_proj", "l2_assoc", "l3_assoc", "Age"]
    d = d.dropna(subset=need).copy()

    # the two regional radial anisotropies, and the composite that pairs them
    d["rho_proj"] = d.l2_proj / d.l3_proj
    d["rho_assoc"] = d.l2_assoc / d.l3_assoc
    d["alps_vox"] = (d.l2_proj + d.l2_assoc) / (d.l3_proj + d.l3_assoc)
    # the weight the pairing applies, which is not a half unless l3 matches
    d["w"] = d.l3_proj / (d.l3_proj + d.l3_assoc)
    # a plain unweighted average, to separate the pairing from the weighting
    d["rho_mean"] = (d.rho_proj + d.rho_assoc) / 2

    # one session per participant for the cross-sectional endpoint, matching
    # the convention the manuscript uses for age associations
    first = d.sort_values(["Subject_ID", "Visit"]).drop_duplicates("Subject_ID")

    print(f"{args.cohort.upper()}  {src.name}")
    print(f"  {len(d)} sessions, {d.Subject_ID.nunique()} participants, "
          f"{len(first)} used for age\n")

    print(f"  weight on the projection region: median {d.w.median():.3f}, "
          f"IQR {d.w.quantile(.25):.3f} to {d.w.quantile(.75):.3f}")
    rr = stats.pearsonr(first.rho_proj, first.rho_assoc)
    print(f"  the two regions agree at r={rr[0]:+.3f}, so the second region "
          f"adds {100 * (1 - rr[0] ** 2):.0f}% new variance\n")

    rows = []
    print(f"  {'measure':<26s} {'r age':>8s} {'p':>10s} {'ICC':>7s} "
          f"{'r/sqrt(ICC)':>12s}")
    for col, label in (("rho_proj", "projection region alone"),
                       ("rho_assoc", "association region alone"),
                       ("rho_mean", "unweighted average"),
                       ("alps_vox", "the paired index")):
        r, p, n = r_age(first, col)
        icc = icc11(d, col)
        dis = r / np.sqrt(icc) if icc == icc and icc > 0 else float("nan")
        print(f"  {label:<26s} {r:>+8.3f} {p:>10.2e} {icc:>7.3f} {dis:>+12.3f}")
        rows.append(dict(cohort=args.cohort, measure=col, label=label,
                         n=n, r_age=r, p_age=p, icc=icc, r_disattenuated=dis))

    best = max(rows[:2], key=lambda x: abs(x["r_age"]))
    paired = rows[-1]
    gain = abs(paired["r_age"]) - abs(best["r_age"])
    print(f"\n  pairing against the better single region "
          f"({best['label']}): {gain:+.3f}")
    print("  A pairing that adds nothing shows a gain near zero or below.")

    out = pd.DataFrame(rows)
    out["weight_proj_median"] = float(d.w.median())
    out["region_agreement_r"] = float(rr[0])
    p = HERE / f"two_roi_value_{args.cohort}.csv"
    out.to_csv(p, index=False)
    print(f"\n  wrote {p.name}")


if __name__ == "__main__":
    main()
