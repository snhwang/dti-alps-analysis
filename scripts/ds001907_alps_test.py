"""Does head position distort the Parkinson's group comparison in DTI-ALPS?

This is the claim the trigeminal cohort was carrying, tested on a cohort where
the positioning difference is real rather than marginal. Patients in ds001907
sit with about six degrees more head rotation than controls, Hedges g near one,
with within-scan motion excluded as the cause.

Two questions:

  1. Do the variants separate patients from controls at all?
  2. How much of whatever separation exists is head position rather than
     physiology? That is measured the way the paper measures it everywhere
     else, by how far the group coefficient falls when pose enters the model.

The estimator is a partial correlation of group with the index given the
covariates, NOT a standardized regression coefficient. The two differ by enough
to change the reported absorption by several points, and every other number in
the paper is the partial correlation, so this matches it.

Absorption is compared against a permutation null. Adding any two covariates to
a model moves a coefficient a little, so the question is never whether the
coefficient moved, it is whether it moved further than shuffled pose would move
it. The null shuffles pose across participants, keeping group and the
covariates fixed.

    python ds001907_alps_test.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ds001907_common import assert_group_mapping, demographics

HERE = Path(__file__).resolve().parent
RNG = np.random.default_rng(20260822)
NPERM = 2000


def partial_r(C: np.ndarray, g: np.ndarray, y: np.ndarray) -> float:
    """Partial correlation of g with y given the columns of C.

    Both variables are residualized on C and then correlated, which is the
    estimator the manuscript reports for every group and age contrast.
    """
    def rz(v):
        b, *_ = np.linalg.lstsq(C, np.asarray(v, float), rcond=None)
        return np.asarray(v, float) - C @ b
    return float(np.corrcoef(rz(g), rz(y))[0, 1])


def r_to_p(r: float, n: int, k: int) -> float:
    """Two-sided p for a partial correlation with k covariates."""
    dof = n - k - 2
    if dof <= 0 or abs(r) >= 1:
        return np.nan
    t = r * np.sqrt(dof / (1 - r ** 2))
    return float(2 * stats.t.sf(abs(t), dof))


def bh(p):
    p = np.asarray(p, float)
    ok = ~np.isnan(p)
    q = np.full_like(p, np.nan)
    v = p[ok]
    o = np.argsort(v)
    adj = np.empty_like(v)
    adj[o] = np.minimum.accumulate(
        (v[o] * len(v) / np.arange(1, len(v) + 1))[::-1])[::-1]
    q[ok] = np.clip(adj, 0, 1)
    return q


def main() -> None:
    argparse.ArgumentParser().parse_args()
    assert_group_mapping()

    a = pd.read_csv(HERE / "ds001907_alps.csv")
    pose = pd.read_csv(HERE / "ds001907_pose.csv")
    m = a.merge(pose[["subject", "session", "pitch", "roll", "yaw", "total",
                      "motion_tr_rms"]], on=["subject", "session"], how="inner")
    m["sid"] = m.subject.str.replace("sub-", "", regex=False)
    dem = demographics().rename(columns={"subject": "sid"}).drop(columns=["group"])
    m = m.merge(dem, on="sid", how="left")
    m["patient"] = (m.group == "patient").astype(int)
    m["male"] = (m.sex == "Male").astype(float)
    m["abs_pitch"] = m.pitch.abs()

    variants = [c for c in a.columns
                if c not in ("subject", "session", "group")
                and not c.endswith(("_L", "_R")) and c != "n_slab"]
    # One row per participant. Two sessions per person are not independent, and
    # the group contrast is a between-participant quantity.
    keep = variants + ["abs_pitch", "total", "age", "male", "motion_tr_rms"]
    s = m.groupby(["sid", "patient"], as_index=False)[keep].mean()
    print(f"{len(m)} scans -> {len(s)} participants "
          f"({int(s.patient.sum())} patient, {int((1 - s.patient).sum())} control)")
    print(f"{len(variants)} variants\n")

    one = np.ones(len(s))
    base = np.column_stack([one, s.age.to_numpy(float), s.male.to_numpy(float)])
    posed = np.column_stack([base, s.abs_pitch.to_numpy(float),
                             s.total.to_numpy(float)])
    g = s.patient.to_numpy(float)

    rows = []
    for v in variants:
        y = s[v].to_numpy(float)
        if np.isnan(y).any():
            ok = ~np.isnan(y)
            b0 = partial_r(base[ok], g[ok], y[ok])
            b1 = partial_r(posed[ok], g[ok], y[ok])
            n = int(ok.sum())
        else:
            b0, b1, n = partial_r(base, g, y), partial_r(posed, g, y), len(s)
        rows.append({"variant": v, "n": n, "r_adj": b0, "p_adj": r_to_p(b0, n, 2),
                     "r_pose": b1, "p_pose": r_to_p(b1, n, 4),
                     "absorbed_pct": 100 * (1 - abs(b1) / abs(b0)) if b0 else np.nan})
    t = pd.DataFrame(rows)
    t["q_adj"] = bh(t.p_adj)

    # The partial correlations below are adjusted, which is right for the
    # inference but hides the raw picture. The unadjusted contrast is what a
    # conventional ALPS paper would report, and it is the number the thesis
    # speaks to, so it is shown first along with what each adjustment costs.
    print("=== 0. Unadjusted, as a conventional study would report it ===")
    print(f"{'variant':<16s} {'patient':>16s} {'control':>16s} {'d':>7s} {'p':>8s}")
    for v in ["classic"] + [x for x in variants if x != "classic"]:
        a_ = s.loc[s.patient == 1, v].dropna()
        b_ = s.loc[s.patient == 0, v].dropna()
        if len(a_) < 5 or len(b_) < 5:
            continue
        _, pv = stats.ttest_ind(a_, b_, equal_var=False)
        sd = np.sqrt(((len(a_) - 1) * a_.var(ddof=1) + (len(b_) - 1) * b_.var(ddof=1))
                     / (len(a_) + len(b_) - 2))
        print(f"{v:<16s} {a_.mean():8.3f} (SD{a_.std():5.3f}) "
              f"{b_.mean():8.3f} (SD{b_.std():5.3f}) "
              f"{(a_.mean()-b_.mean())/sd:7.2f} {pv:8.3f}")

    print("\n   the classic index, one adjustment at a time:")
    for lab, C in (("unadjusted", one[:, None]),
                   ("age", np.column_stack([one, s.age])),
                   ("sex", np.column_stack([one, s.male])),
                   ("age+sex", base),
                   ("age+sex+pose", posed)):
        y = s["classic"].to_numpy(float)
        r = partial_r(np.asarray(C, float), g, y)
        print(f"      {lab:<14s} r={r:+.3f}  p={r_to_p(r, len(s), C.shape[1]-1):.3f}")

    print("\n=== 1. Does any variant separate patients from controls? ===")
    print("   (partial correlation of group with the index, given age and sex)\n")
    print(f"{'variant':<16s} {'r':>7s} {'p':>8s} {'BH q':>7s}")
    for r in t.sort_values("p_adj").itertuples():
        star = "  *" if r.q_adj < 0.05 else ""
        print(f"{r.variant:<16s} {r.r_adj:7.3f} {r.p_adj:8.3f} {r.q_adj:7.3f}{star}")
    sig = t[t.q_adj < 0.05]
    print(f"\n   {len(sig)} of {len(t)} variants separate the groups after "
          f"BH correction.")

    print("\n=== 2. How much of the group contrast is head position? ===")
    print(f"{'variant':<16s} {'age+sex':>9s} {'+pose':>9s} {'absorbed':>9s}")
    for r in t.sort_values("absorbed_pct", ascending=False).itertuples():
        print(f"{r.variant:<16s} {r.r_adj:9.3f} {r.r_pose:9.3f} "
              f"{r.absorbed_pct:8.1f}%")

    # --- permutation null for the absorption --------------------------------
    print(f"\n=== 3. Permutation control, {NPERM} shuffles of pose ===")
    print("   Shuffling pose across participants, keeping group, age and sex.")
    print(f"{'variant':<16s} {'absorbed':>9s} {'null mean':>10s} "
          f"{'null max':>9s} {'p':>8s}")
    perm_rows = []
    for v in (list(sig.variant) if len(sig) else list(t.nlargest(4, "absorbed_pct").variant)):
        y = s[v].to_numpy(float)
        ok = ~np.isnan(y)
        b0 = partial_r(base[ok], g[ok], y[ok])
        obs = 100 * (1 - abs(partial_r(posed[ok], g[ok], y[ok])) / abs(b0))
        null = np.empty(NPERM)
        pp = np.column_stack([s.abs_pitch.to_numpy(float), s.total.to_numpy(float)])
        for i in range(NPERM):
            perm = RNG.permutation(len(s))
            P = np.column_stack([base, pp[perm]])
            null[i] = 100 * (1 - abs(partial_r(P[ok], g[ok], y[ok])) / abs(b0))
        pv = (1 + (null >= obs).sum()) / (NPERM + 1)
        print(f"{v:<16s} {obs:8.1f}% {null.mean():9.2f}% {null.max():8.1f}% {pv:8.4f}")
        perm_rows.append({"variant": v, "absorbed_pct": obs,
                          "null_mean": null.mean(), "null_max": null.max(),
                          "perm_p": pv})

    t.to_csv(HERE / "ds001907_alps_contrast.csv", index=False)
    if perm_rows:
        pd.DataFrame(perm_rows).to_csv(HERE / "ds001907_alps_permutation.csv",
                                       index=False)
    print(f"\n   wrote {HERE / 'ds001907_alps_contrast.csv'}")


if __name__ == "__main__":
    main()
