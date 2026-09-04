"""Does Refined+ differ from the refined index enough to keep?

Refined+ projects the PVS axis onto each voxel's radial plane before averaging,
so it is perpendicular to the local fiber rather than to the region means. The
manuscript carried it in tbl:variants with a rotation departure and four dashes,
and the appendix asserted it "does not outperform the refined index" on the
strength of columns that were never computed.

decoupled_roi_*.csv still holds paired per-session values for both, from the
pipeline as it stood before the regions were redrawn. That is enough to answer
whether the two indices are distinguishable at all, which is what decides
whether the variant earns a place. It is not enough to publish a comparison,
since those values predate the redraw, and this script says so rather than
quietly treating them as current.

    python refined_plus_value.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent


def main() -> None:
    for cohort, fn in (("HCP-A", "decoupled_roi_hcpa_b1500.csv"),
                       ("DLBS", "decoupled_roi_dlbs.csv")):
        p = HERE / fn
        if not p.exists():
            print(f"{cohort}: {fn} missing\n")
            continue
        d = pd.read_csv(p)
        pairs = [(c, c.replace("refined_", "refinedplus_"))
                 for c in d.columns
                 if c.startswith("refined_")
                 and c.replace("refined_", "refinedplus_") in d.columns]
        if not pairs:
            print(f"{cohort}: no paired refined/refinedplus columns\n")
            continue
        print(f"{cohort}, {len(d)} sessions")
        print(f"  {'measure':<22s} {'refined':>9s} {'Refined+':>9s} "
              f"{'r':>7s} {'median |d|':>11s} {'p':>9s}")
        for a, b in pairs:
            x, y = d[a], d[b]
            ok = x.notna() & y.notna()
            if ok.sum() < 10:
                continue
            x, y = x[ok], y[ok]
            r = float(np.corrcoef(x, y)[0, 1])
            t = stats.wilcoxon(x, y)
            print(f"  {a.replace('refined_',''):<22s} {x.mean():>9.4f} "
                  f"{y.mean():>9.4f} {r:>7.4f} "
                  f"{float(np.median(np.abs(x - y))):>11.5f} {t.pvalue:>9.2e}")
        print()

    print("  These come from the pre-redraw pipeline. They answer whether the")
    print("  two are distinguishable, not what either is worth now.")


if __name__ == "__main__":
    main()
