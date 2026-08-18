"""
Does the between-visit change grow with the interval between visits?

The reliability figures are computed on visits separated by years, so the
within-participant variance may hold real biological change as well as
measurement error. If it does, each ICC understates short-interval test-retest
reliability and should be read as a lower bound. That is a convenient thing to
believe and it should not be assumed.

The test is simple. If accumulated change contributes, the size of the
between-visit difference should rise with the gap. If it does not, the
within-participant variance is measurement error and the ICC is an estimate of
reliability rather than a floor under it.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

import atomic_io  # noqa: F401  writes become atomic on import
from scipy import stats

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
COHORTS = (("DLBS", "decoupled_roi_dlbs.csv"),
           ("HCP-A", "decoupled_roi_hcpa_b1500.csv"))
VARIANTS = ("classic", "refined_slab")


def pairs(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, g in d.groupby("Subject_ID"):
        g = g.sort_values("Age")
        for i in range(len(g) - 1):
            a, b = g.iloc[i], g.iloc[i + 1]
            gap = float(b.Age) - float(a.Age)
            if gap <= 0:
                continue
            rec = {"gap": gap}
            for c in VARIANTS:
                mean = (float(a[c]) + float(b[c])) / 2
                rec[c] = abs(float(b[c]) - float(a[c])) / abs(mean) * 100
            rows.append(rec)
    return pd.DataFrame(rows)


def main() -> None:
    out = []
    for tag, f in COHORTS:
        d = pd.read_csv(HERE / f).dropna(subset=["Age", *VARIANTS])
        p = pairs(d)
        print(f"{tag}: {len(p)} consecutive visit pairs, median gap {p.gap.median():.2f} y")
        for c in VARIANTS:
            r, pv = stats.pearsonr(p.gap, p[c])
            verdict = "change accumulates" if (r > 0 and pv < 0.05) else "no accumulation detected"
            print(f"    {c:<13s} r={r:+.3f} p={pv:.3f}  median |change| {p[c].median():5.2f}%"
                  f"   {verdict}")
            out.append({"cohort": tag, "variant": c, "n_pairs": len(p),
                        "median_gap_y": p.gap.median(), "r": r, "p": pv,
                        "median_change_pct": p[c].median()})
    pd.DataFrame(out).to_csv(HERE / "visit_interval_test.csv", index=False)
    print(f"\nwrote {HERE / 'visit_interval_test.csv'}")


if __name__ == "__main__":
    main()
