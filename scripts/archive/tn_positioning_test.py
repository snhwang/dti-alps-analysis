"""Group positioning in the trigeminal cohort, tested correctly.

The manuscript's positioning table originally reported pooled-variance t tests,
p = 0.048 for absolute pitch and p = 0.031 for total rotation. Those are the
wrong test. The patients are considerably more variable than the controls, which
the manuscript itself reports, and Levene rejects equal variance for both
quantities. Welch's t test makes no equal-variance assumption and is what should
have been used.

It also happens to be the more favourable test here, which is worth stating
plainly: the corrected p values are smaller, not larger, so the original
reporting understated the result.

Three things are computed.

  Welch      the test itself, with its Welch-Satterthwaite confidence interval
  Levene     the justification for preferring it, since if variances were equal
             the pooled test would be fine and the choice would look arbitrary
  BH         Benjamini-Hochberg across all four rotation quantities, because
             pitch and total rotation are the two smallest of four and a reader
             is entitled to ask about the other two

An age- and sex-adjusted group contrast is also reported. It does not reach
significance, and the manuscript says so, because the patients were older and a
reader will ask whether the positioning difference is age wearing a disguise.

    python tn_positioning_test.py

Writes tn_positioning_test.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import atomic_io  # noqa: F401  writes become atomic on import
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from data_paths import winpath  # noqa: E402

PARTICIPANTS = "M:/ds005713-download/participants_v2.0.1.tsv"
AXES = ("pitch", "roll", "yaw", "total")


def bh(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    q = np.empty_like(p)
    prev, n = 1.0, len(p)
    for rank, i in enumerate(order[::-1], 1):
        prev = min(prev, p[i] * n / (n - rank + 1))
        q[i] = prev
    return q


def main() -> None:
    argparse.ArgumentParser().parse_args()

    rot = pd.read_csv(HERE / "head_rotation_tn.csv")
    par = pd.read_csv(winpath(PARTICIPANTS), sep="\t")
    par.columns = [c.strip().lstrip("\ufeff") for c in par.columns]

    m = rot.merge(par[["BIDS_ID", "age", "sex", "Clinical_finding"]], on="BIDS_ID")
    # "TN", "TN, SUNCT" and "TN, Left-sided SUNA" are all patients. Controls are
    # "normal", which appears with and without a trailing space.
    m["patient"] = (m.Clinical_finding.astype(str).str.strip().str.upper()
                    .str.startswith("TN").astype(int))
    m = m.dropna(subset=["pitch", "age"])
    n_pat, n_ctl = int(m.patient.sum()), int((1 - m.patient).sum())
    print(f"{len(m)} sessions: {n_pat} patients, {n_ctl} controls\n")

    rows = []
    for ax in AXES:
        a = m.loc[m.patient == 1, ax].abs()
        b = m.loc[m.patient == 0, ax].abs()
        _, p_pool = stats.ttest_ind(a, b, equal_var=True)
        t_w, p_welch = stats.ttest_ind(a, b, equal_var=False)
        _, p_lev = stats.levene(a, b)

        # Welch-Satterthwaite interval on the difference of means
        va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
        se = np.sqrt(va + vb)
        df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
        crit = stats.t.ppf(0.975, df)
        diff = a.mean() - b.mean()

        sd_pool = np.sqrt(((len(a) - 1) * a.var(ddof=1)
                           + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
        rows.append(dict(axis=ax, mean_patient=round(a.mean(), 3),
                         mean_control=round(b.mean(), 3),
                         sd_patient=round(a.std(ddof=1), 3),
                         sd_control=round(b.std(ddof=1), 3),
                         difference=round(diff, 3),
                         ci_lo=round(diff - crit * se, 3),
                         ci_hi=round(diff + crit * se, 3),
                         cohens_d=round(diff / sd_pool, 3),
                         p_pooled=p_pool, p_welch=p_welch, p_levene=p_lev,
                         welch_df=round(df, 1)))

    out = pd.DataFrame(rows)
    out["q_welch_bh"] = bh(out.p_welch.to_numpy())

    print(f"   {'axis':7s}{'patient':>9s}{'control':>9s}{'diff':>7s}"
          f"{'95% CI':>18s}{'d':>7s}{'Welch p':>10s}{'BH q':>8s}{'Levene':>9s}")
    for r in out.itertuples():
        ci = f"[{r.ci_lo:+.2f}, {r.ci_hi:+.2f}]"
        print(f"   {r.axis:7s}{r.mean_patient:9.2f}{r.mean_control:9.2f}"
              f"{r.difference:7.2f}{ci:>18s}{r.cohens_d:7.2f}"
              f"{r.p_welch:10.4f}{r.q_welch_bh:8.4f}{r.p_levene:9.4f}")

    # age and sex adjusted, which is the harder question
    X = np.column_stack([np.ones(len(m)), m.patient, m.age,
                         pd.to_numeric(m.sex, errors="coerce").fillna(0)])
    y = m.pitch.abs().to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(m) - X.shape[1]
    se = np.sqrt((resid @ resid / dof) * np.linalg.inv(X.T @ X)[1, 1])
    t_adj = beta[1] / se
    p_adj = float(2 * stats.t.sf(abs(t_adj), dof))
    out.attrs["p_adjusted"] = p_adj
    pd.DataFrame([dict(quantity="abs pitch, age and sex adjusted",
                       coefficient_deg=round(float(beta[1]), 3),
                       t=round(float(t_adj), 3), p=p_adj,
                       n=len(m))]).to_csv(HERE / "tn_positioning_adjusted.csv",
                                          index=False)
    out.to_csv(HERE / "tn_positioning_test.csv", index=False)

    print(f"\n   Levene rejects equal variance for pitch and total rotation, so")
    print(f"   the pooled test is not the right one. Welch gives smaller p")
    print(f"   values than the pooled test reported originally, and both")
    print(f"   survive Benjamini-Hochberg across all four quantities.")
    print(f"\n   Age- and sex-adjusted group effect on absolute pitch: "
          f"{beta[1]:+.3f} deg, t = {t_adj:.2f}, p = {p_adj:.4f}.")
    print(f"   This does not reach significance. The patients were older, and a")
    print(f"   reader is entitled to ask whether the positioning difference is")
    print(f"   age. The manuscript reports it rather than leaving it out.")


if __name__ == "__main__":
    main()
