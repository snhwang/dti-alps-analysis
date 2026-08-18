"""
Is the automated region placement itself reproducible between visits?

The paper argues that removing the operator removes rater variability, and the
manual-versus-automated comparison shows what that substitution costs. Neither
shows that the automated placement is stable. It is not deterministic across
visits: each session is registered separately, from a different image of the
same head, so the regions land wherever that registration puts them.

This asks the question directly. For every participant with repeat visits, how
much do the placed regions vary in size, in what tissue they contain, and in the
tract directions they yield? Reported as ICC(1,1) and within-participant
coefficient of variation, on the same footing as the index itself, so that
placement stability and index stability can be read against each other.

The comparison that matters is placement against index. If a placement property
is markedly more reproducible than the index computed from it, placement is not
what limits the index. If it is comparably unstable, it is.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import atomic_io  # noqa: F401  writes become atomic on import

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from estimator_variants import variance_components

COHORTS = (("HCP-A", "roi_placement_quality_hcpa_b1500.csv"),
           ("DLBS", "roi_placement_quality_dlbs_all.csv"))

# what the measure tells us about the placement
MEASURES = [
    ("n_scr", "projection region size (voxels)"),
    ("n_slf", "association region size (voxels)"),
    ("scr_off_tract", "projection off-tract fraction"),
    ("slf_off_tract", "association off-tract fraction"),
    ("theta_scr", "projection direction vs z (deg)"),
    ("theta_slf", "association direction vs y (deg)"),
    ("theta_pvs", "perivascular axis vs x (deg)"),
    ("theta_interfiber", "inter-fibre angle (deg)"),
]
REFERENCE = [("classic", "classic index"), ("refined_slab", "refined index")]


def summarise(d: pd.DataFrame, col: str) -> dict | None:
    s = d.dropna(subset=[col])
    lon = s[s.Subject_ID.isin(s.Subject_ID.value_counts()[lambda x: x >= 2].index)]
    if len(lon) < 20 or lon[col].std(ddof=1) == 0:
        return None
    v = variance_components(lon, col)
    g = lon.groupby("Subject_ID")[col]
    # Median between-visit range. Reported relative to the participant mean where
    # that mean is meaningfully non-zero, and on the measure's own scale where it
    # is not: the projection off-tract fraction is zero in most sessions, which
    # makes any relative spread meaningless rather than large.
    absolute = g.apply(lambda x: x.max() - x.min())
    centre = abs(lon[col].median())
    relative = centre > 1e-6 and centre > 0.02 * abs(lon[col]).max()
    rng = (g.apply(lambda x: (x.max() - x.min()) / abs(x.mean()) * 100
                   if abs(x.mean()) > 1e-9 else np.nan) if relative else absolute)
    return {"icc": v["icc"], "wcv": v["wcv_pct"] if relative else np.nan,
            "n": v["n"], "n_subj": int(lon.Subject_ID.nunique()),
            "median_range": float(np.nanmedian(rng)), "relative": relative}


def main() -> None:
    rows = []
    for tag, f in COHORTS:
        d = pd.read_csv(HERE / f)
        for col, label in MEASURES + REFERENCE:
            if col not in d:
                continue
            r = summarise(d, col)
            if r:
                rows.append({"cohort": tag, "measure": label, "col": col, **r})

    out = pd.DataFrame(rows)
    out.to_csv(HERE / "placement_reproducibility.csv", index=False)

    for tag, _ in COHORTS:
        s = out[out.cohort == tag]
        if s.empty:
            continue
        print(f"\n{tag}  ({int(s.n_subj.iloc[0])} participants with repeat visits, "
              f"{int(s.n.iloc[0])} sessions)")
        print(f"  {'measure':<34s} {'ICC':>7s} {'wCV%':>7s} {'median visit range':>20s}")
        for r in s.itertuples():
            mark = "  <- index" if r.col in dict(REFERENCE) else ""
            cv = f"{r.wcv:7.2f}" if np.isfinite(r.wcv) else f"{'--':>7s}"
            rng = f"{r.median_range:8.1f}%" if r.relative else f"{r.median_range:8.3f} abs"
            print(f"  {r.measure:<34s} {r.icc:7.3f} {cv} {rng:>20s}{mark}")

    print("\nRead the placement rows against the index rows. A placement property that")
    print("is more reproducible than the index it feeds is not what limits the index.")
    print(f"\nwrote {HERE / 'placement_reproducibility.csv'}")


if __name__ == "__main__":
    main()
