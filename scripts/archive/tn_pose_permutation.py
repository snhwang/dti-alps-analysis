"""Is the trigeminal pose absorption more than an adjustment artifact?

Section 3 states the principle plainly: adjusting a coefficient for any
covariate can shrink it, so the head-pose adjustment has to be tested against a
permutation null. That test is reported for DLBS, where the observed 45.0%
absorption sits against a null whose mean is -0.09%, whose 95th percentile is
4.6% and whose maximum over 2000 permutations is 15.1%. It is also reported for
HCP-A, where it correctly returns nothing.

It was never run for the trigeminal cohort, whose headline number is a 20%
absorption of the group coefficient. That number is therefore undefended by the
paper's own standard, and 20% is not obviously outside a null that reached 15.1%
at n = 156.

This runs the same test. Pose values are shuffled across participants, which
preserves their marginal distribution exactly while breaking their relation to
group and to the index, and the same adjustment is re-run. The observed
absorption is then placed in that distribution.

    python tn_pose_permutation.py

Writes tn_pose_permutation.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import atomic_io  # noqa: F401  writes become atomic on import

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from data_paths import winpath  # noqa: E402

PARTICIPANTS = "M:/ds005713-download/participants_v2.0.1.tsv"


def group_beta(y, g, covs):
    """Partial correlation of group with the index, given the covariates.

    This is the estimator the manuscript uses, not a standardized regression
    coefficient. Both variables are residualized on the covariate matrix and the
    residuals correlated. The two differ in scale, so the absorbed percentage
    differs too, and the permutation has to use whichever the paper reports.
    """
    C = np.column_stack([np.ones(len(y))] + [np.asarray(c, float) for c in covs])

    def rz(v):
        b, *_ = np.linalg.lstsq(C, np.asarray(v, float), rcond=None)
        return np.asarray(v, float) - C @ b

    return float(np.corrcoef(rz(g), rz(y))[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--metric", default="classic")
    args = ap.parse_args()

    alps = pd.read_csv(HERE / "tn_alps.csv")
    rot = pd.read_csv(HERE / "head_rotation_tn.csv")
    par = pd.read_csv(winpath(PARTICIPANTS), sep="\t")
    par.columns = [c.strip().lstrip("\ufeff") for c in par.columns]

    d = (alps.merge(rot, on="BIDS_ID")
             .merge(par[["BIDS_ID", "age", "sex", "Clinical_finding"]], on="BIDS_ID"))
    d["patient"] = (d.Clinical_finding.astype(str).str.strip().str.upper()
                    .str.startswith("TN").astype(float))
    d["sex_n"] = pd.to_numeric(d.sex, errors="coerce").fillna(0)
    d = d.dropna(subset=[args.metric, "pitch", "total", "age"])
    print(f"{len(d)} sessions: {int(d.patient.sum())} patients, "
          f"{int((1 - d.patient).sum())} controls\n")

    y = d[args.metric].to_numpy()
    g = d.patient.to_numpy()
    base_cov = [d.age.to_numpy(), d.sex_n.to_numpy()]
    pose = [d.pitch.abs().to_numpy(), d.total.abs().to_numpy()]

    b0 = group_beta(y, g, base_cov)
    b1 = group_beta(y, g, base_cov + pose)
    observed = 100 * (1 - abs(b1) / abs(b0))
    print(f"   group coefficient, age and sex adjusted      {b0:+.4f}")
    print(f"   after adding head pose                       {b1:+.4f}")
    print(f"   absorbed by pose                             {observed:.1f}%\n")

    rng = np.random.default_rng(0)
    null = np.empty(args.n_perm)
    for k in range(args.n_perm):
        idx = rng.permutation(len(d))
        shuffled = [p[idx] for p in pose]
        bk = group_beta(y, g, base_cov + shuffled)
        null[k] = 100 * (1 - abs(bk) / abs(b0))

    p = float((null >= observed).sum() + 1) / (args.n_perm + 1)
    print(f"   permutation null over {args.n_perm} shuffles")
    print(f"      mean            {null.mean():+.2f}%")
    print(f"      95th percentile {np.percentile(null, 95):+.2f}%")
    print(f"      maximum         {null.max():+.2f}%")
    print(f"\n   observed {observed:.1f}% against that null: p = {p:.4f}")
    print("   " + ("survives" if p < 0.05
                   else "DOES NOT survive: the absorption is within what an"
                        " irrelevant covariate produces"))

    pd.DataFrame([dict(metric=args.metric, n=len(d),
                       beta_unadjusted=round(b0, 5),
                       beta_pose_adjusted=round(b1, 5),
                       absorbed_pct=round(observed, 3),
                       null_mean=round(float(null.mean()), 3),
                       null_p95=round(float(np.percentile(null, 95)), 3),
                       null_max=round(float(null.max()), 3),
                       p_permutation=p, n_perm=args.n_perm)]
                 ).to_csv(HERE / "tn_pose_permutation.csv", index=False)


if __name__ == "__main__":
    main()
