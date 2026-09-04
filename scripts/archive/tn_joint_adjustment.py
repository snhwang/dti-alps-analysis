"""What is left of the trigeminal group difference after pose and the ratio?

Two reductions of that contrast are already reported separately and they are
not the same size. Pose adjustment absorbs about a fifth of the classic
index's patient-control coefficient (tn_pose_permutation.py). Partialling the
eigenvalue ratio takes it from significant to not (beyond_eigenvalue_ratio.py).
Neither says what remains when both are removed, and that is the question a
reader asks on being told the difference is postural: how much of it actually
is.

Both terms are entered together here, on the same participants, against the
same contrast. The order matters for interpretation and not for the result, so
the nested sequence is reported: unadjusted, then pose, then ratio, then both.

The expectation going in was that pose and the ratio would overlap, since pitch
lowers the index by turning the measurement axis away from the ratio, so that
the joint model would remove less than the sum of the two separately. The data
says the opposite: jointly they remove more than separately. That is
suppression rather than redundancy. Each covariate explains variance in the
index that is unrelated to the group contrast, and removing one sharpens the
other, so the two together leave less than either predicts alone. The output
reports both figures so the direction is visible rather than assumed.

    python tn_joint_adjustment.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from data_paths import winpath

HERE = Path(__file__).resolve().parent
RATIO = "pv_perp"


def partial_r(y, x, covs) -> tuple[float, float]:
    """Correlation of y and x with covs removed from both."""
    Y, X = np.asarray(y, float), np.asarray(x, float)
    C = [np.asarray(c, float) for c in covs]
    m = np.isfinite(Y) & np.isfinite(X)
    for c in C:
        m &= np.isfinite(c)
    Y, X, C = Y[m], X[m], [c[m] for c in C]
    A = np.column_stack([np.ones(len(Y))] + C) if C else np.ones((len(Y), 1))
    ry = Y - A @ np.linalg.lstsq(A, Y, rcond=None)[0]
    rx = X - A @ np.linalg.lstsq(A, X, rcond=None)[0]
    if rx.std() <= 1e-8 * max(X.std(), 1e-30):
        return float("nan"), float("nan")
    r, _ = stats.pearsonr(rx, ry)
    df = len(Y) - 2 - len(C)
    t = r * np.sqrt(df / max(1 - r ** 2, 1e-300))
    return float(r), float(2 * stats.t.sf(abs(t), df))


def main() -> None:
    tn = pd.read_csv(HERE / "tn_alps.csv")
    pose = pd.read_csv(HERE / "head_rotation_tn.csv")
    par = pd.read_csv(winpath("M:/ds005713-download/participants_v2.0.1.tsv"), sep="\t")
    m = tn.merge(par, on="BIDS_ID").merge(pose[["BIDS_ID", "pitch"]], on="BIDS_ID")
    # the same patient rule the rest of the project uses
    m["patient"] = (m.BIDS_ID.astype(str).str.extract(r"sub-(\d+)")[0]
                    .str.len() >= 3).astype(float)
    m["age"] = pd.to_numeric(m.age, errors="coerce")
    m["sex_n"] = pd.to_numeric(m.sex, errors="coerce")
    m["abs_pitch"] = m.pitch.abs()
    m = m.dropna(subset=["age", "sex_n", "abs_pitch", RATIO, "classic"])

    base = [m.age, m.sex_n]
    arms = [
        ("age and sex only", base),
        ("+ head pitch", base + [m.abs_pitch]),
        ("+ eigenvalue ratio", base + [m[RATIO]]),
        ("+ both", base + [m.abs_pitch, m[RATIO]]),
    ]
    print(f"trigeminal patient-control contrast, n={len(m)}\n")
    print(f"  {'model':<22s} {'r':>8s} {'p':>9s} {'absorbed':>10s}")
    r0 = None
    rows = []
    for name, covs in arms:
        r, p = partial_r(m.patient, m.classic, covs)
        if r0 is None:
            r0 = r
        pct = 100 * (1 - abs(r) / abs(r0))
        print(f"  {name:<22s} {r:>+8.3f} {p:>9.4f} {pct:>9.1f}%")
        rows.append(dict(model=name, r=r, p=p, absorbed_pct=pct, n=len(m)))

    sep = rows[1]["absorbed_pct"] + rows[2]["absorbed_pct"]
    print(f"\n  pose alone {rows[1]['absorbed_pct']:.1f}% + ratio alone "
          f"{rows[2]['absorbed_pct']:.1f}% = {sep:.1f}%, "
          f"against {rows[3]['absorbed_pct']:.1f}% jointly")
    gap = rows[3]["absorbed_pct"] - sep
    if gap > 0:
        print(f"  jointly they remove {gap:.1f} points MORE than separately, which is")
        print("  suppression, not redundancy: each explains index variance unrelated")
        print("  to the group contrast, so removing one sharpens the other.")
    else:
        print(f"  jointly they remove {-gap:.1f} points LESS than separately, so that "
              "much of what each explains is the same variance.")

    pd.DataFrame(rows).to_csv(HERE / "tn_joint_adjustment.csv", index=False)
    print("\n  wrote tn_joint_adjustment.csv")


if __name__ == "__main__":
    main()
