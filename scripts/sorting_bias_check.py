"""Is the eigenvalue ratio's age association an artifact of sorting bias?

The eigenvalues are sorted, so lambda2 >= lambda3 holds by construction and not
by anatomy. Estimation noise alone therefore drives the ROI mean of lambda2
above that of lambda3 even in tissue that is perfectly transversely isotropic,
and the inflation grows as the signal-to-noise ratio falls (Pierpaoli and
Basser, 1996). Any ratio of sorted eigenvalues carries a noise floor above one.

That matters here because lambda2/lambda3 carries the strongest age association
of any variant in this paper. If data quality declines with age, sorting bias
could manufacture part of that association without any tissue change.

The hypothesis makes a signed prediction, which is what makes it testable.
Poorer data means lower SNR, which means a larger sorting bias, which means a
LARGER lambda2/lambda3. So under the artifact hypothesis the ratio should rise
with motion and with outlier slices. If age also brings poorer data, the induced
age association is POSITIVE.

The observed age association is negative. So the artifact does not predict the
finding; it predicts its opposite, and would be suppressing the true effect
rather than creating it. Three tests:

  1. does data quality decline with age in this cohort
  2. does the ratio rise with poorer data, as the artifact requires
  3. does the ratio's age association survive adjustment for data quality

Test 3 is the one that matters. Tests 1 and 2 establish whether the artifact is
even in play.

    python sorting_bias_check.py

Writes sorting_bias_check.csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import atomic_io  # noqa: F401  writes become atomic on import
from scipy import stats

HERE = Path(__file__).resolve().parent
DIFF = HERE.parent.parent / "diffusion"
QUALITY = ["motion_rms", "pct_outliers"]
VARIANTS = ["classic", "cross", "v2_slab", "pv_perp"]


def partial(y, x, covs) -> tuple[float, float]:
    C = np.column_stack([np.ones(len(y))] + [np.asarray(c, float) for c in covs])

    def rz(v):
        b, *_ = np.linalg.lstsq(C, np.asarray(v, float), rcond=None)
        return np.asarray(v, float) - C @ b

    a, b = rz(x), rz(y)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan"), float("nan")
    r = float(np.corrcoef(a, b)[0, 1])
    dof = len(y) - C.shape[1] - 1
    return r, float(2 * stats.t.sf(abs(r * np.sqrt(dof / max(1 - r * r, 1e-12))), dof))


def main() -> None:
    argparse.ArgumentParser().parse_args()

    d = pd.read_csv(HERE / "measured_pvs_axis_hcpa_b1500_all.csv")
    q = pd.read_csv(DIFF / "HCP" / "motion_rms_n1379.csv").rename(
        columns={"subject_id": "Subject_ID", "visit": "Visit"})
    for f in (d, q):
        f["Subject_ID"] = f.Subject_ID.astype(str)
        f["Visit"] = f.Visit.astype(str)
    m = d.merge(q[["Subject_ID", "Visit"] + QUALITY], on=["Subject_ID", "Visit"],
                how="inner")
    m = m.sort_values(["Subject_ID", "Visit"]).groupby("Subject_ID").first().reset_index()
    m = m.replace([np.inf, -np.inf], np.nan).dropna(subset=["Age"] + QUALITY + VARIANTS)
    print(f"HCP-A, {len(m)} participants with both an index and a quality measure\n")

    rows = []

    print("1. does data quality decline with age?\n")
    for c in QUALITY:
        r, p = stats.pearsonr(m[c], m.Age)
        rows.append(dict(test="quality_vs_age", variant="", quality=c,
                         r=float(r), p=float(p)))
        print(f"   {c:14s} vs age   r = {r:+.4f}   p = {p:.3g}")
    print("\n   Positive means older participants give poorer data, which is the")
    print("   condition under which sorting bias could act as a confound.\n")

    print("2. does the ratio rise with poorer data, as the artifact requires?\n")
    print("   (partialling age, so this is the quality effect at fixed age)")
    for v in VARIANTS:
        for c in QUALITY:
            r, p = partial(m[v], m[c], [m.Age])
            rows.append(dict(test="index_vs_quality_given_age", variant=v, quality=c,
                             r=float(r), p=float(p)))
            flag = "  <-- as artifact predicts" if r > 0 and p < 0.05 else ""
            print(f"   {v:10s} vs {c:14s}  r = {r:+.4f}  p = {p:8.3g}{flag}")
    print()

    print("3. does the age association survive adjustment for data quality?\n")
    print(f"   {'variant':10s} {'raw':>9s} {'| quality':>10s} {'p':>10s}   shift")
    for v in VARIANTS:
        r0 = float(stats.pearsonr(m[v], m.Age)[0])
        r1, p1 = partial(m.Age, m[v], [m[c] for c in QUALITY])
        rows.append(dict(test="age_given_quality", variant=v, quality="+".join(QUALITY),
                         r=r1, p=p1, raw=r0))
        print(f"   {v:10s} {r0:+9.4f} {r1:+10.4f} {p1:10.3g}   {r1 - r0:+.4f}")

    pd.DataFrame(rows).to_csv(HERE / "sorting_bias_check.csv", index=False)

    pv = [x for x in rows if x["test"] == "age_given_quality" and x["variant"] == "pv_perp"][0]
    print(f"\n   The eigenvalue ratio's age association moves by {pv['r'] - pv['raw']:+.4f}")
    print("   when data quality is adjusted for. Sorting bias predicts a positive")
    print("   contribution to the ratio in poorer data, so it cannot account for a")
    print("   negative age association, and would if anything be masking it.")


if __name__ == "__main__":
    main()
