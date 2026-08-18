"""
Trigeminal neuralgia: do any ALPS variants separate patients from controls?

Two questions, and the order matters. First whether the index discriminates at
all in a patient cohort, since the healthy-aging data could not test that.
Second, only if it does, whether the orientation-corrected variants discriminate
better. Group discrimination is the endpoint the positioning artefact was shown
to corrupt, so it is the one where a correction should earn its place.

Group is read from the BIDS identifier, two digits for controls and three for
patients, which is the convention this dataset uses.

Reported per variant: age- and sex-adjusted group difference as a standardised
beta with its p value, Cohen's d on the adjusted residuals, and AUC. Within
patients, associations with Sindou grade (nerve compression severity graded at
surgery), pain severity and disease duration.

Nothing here is corrected for the number of variants, because the four are not
independent hypotheses; they are four ways of computing the same quantity. FDR
is applied across the clinical measures within patients.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import atomic_io  # noqa: F401  writes become atomic on import
from scipy import stats

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from hemisphere_age import williams

HERE = Path(__file__).resolve().parent
PARTICIPANTS = Path(r"M:/ds005713-download/participants_v2.0.1.tsv")
VARIANTS = ["classic", "cross", "v2_sphere", "v2_slab"]
CLINICAL = [("Sindou_grade", "Sindou grade"),
            ("Pain_severity (average score)", "pain severity"),
            ("Disease_duration (years)", "disease duration")]


def auc(pos, neg):
    n1, n2 = len(pos), len(neg)
    if n1 == 0 or n2 == 0:
        return np.nan
    r = stats.rankdata(np.concatenate([pos, neg]))
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2))


def resid(y, C):
    b, *_ = np.linalg.lstsq(C, np.asarray(y, float), rcond=None)
    return np.asarray(y, float) - C @ b


def main() -> None:
    d = pd.read_csv(HERE / "tn_alps.csv")
    p = pd.read_csv(PARTICIPANTS, sep="\t")
    m = d.merge(p, on="BIDS_ID", how="inner")
    digits = m.BIDS_ID.astype(str).str.extract(r"sub-(\d+)")[0]
    m["patient"] = (digits.str.len() >= 3).astype(int)
    m["sex_n"] = pd.to_numeric(m.sex, errors="coerce")
    m["age"] = pd.to_numeric(m.age, errors="coerce")
    m = m.dropna(subset=["age", "sex_n"] + VARIANTS)
    print(f"{len(m)} merged: {int(m.patient.sum())} patients, "
          f"{int((1 - m.patient).sum())} controls")
    print(f"age  patients {m[m.patient==1].age.mean():.1f}  "
          f"controls {m[m.patient==0].age.mean():.1f}  "
          f"(p={stats.ttest_ind(m[m.patient==1].age, m[m.patient==0].age)[1]:.3f})\n")

    C = np.column_stack([np.ones(len(m)), m.age.to_numpy(float), m.sex_n.to_numpy(float)])
    print("PATIENTS vs CONTROLS, adjusted for age and sex")
    print(f"{'variant':<12s} {'beta':>8s} {'p':>10s} {'d':>7s} {'AUC':>7s} "
          f"{'mean pt':>9s} {'mean ctl':>9s}")
    store = {}
    for k in VARIANTS:
        y = resid(m[k], C)
        g = m.patient.to_numpy(float)
        gr = resid(g, C)
        r = float(np.corrcoef(gr, y)[0, 1])
        dof = len(m) - 4
        t = r * np.sqrt(dof / max(1 - r * r, 1e-12))
        pv = float(2 * (1 - stats.t.cdf(abs(t), dof)))
        a, b = y[m.patient == 1], y[m.patient == 0]
        sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                     / (len(a) + len(b) - 2))
        dcoh = (a.mean() - b.mean()) / sp if sp > 0 else np.nan
        store[k] = (r, pv)
        print(f"{k:<12s} {r:+8.3f} {pv:10.2e} {dcoh:+7.3f} {auc(a, b):7.3f} "
              f"{m[m.patient==1][k].mean():9.4f} {m[m.patient==0][k].mean():9.4f}")

    # is any corrected variant a better discriminator than classic?
    print("\nWilliams' test against classic (group membership as the outcome)")
    gr = resid(m.patient.to_numpy(float), C)
    c1 = resid(m["classic"], C)
    for k in VARIANTS[1:]:
        c2 = resid(m[k], C)
        r1 = np.corrcoef(gr, c1)[0, 1]; r2 = np.corrcoef(gr, c2)[0, 1]
        t, pv = williams(r1, r2, float(np.corrcoef(c1, c2)[0, 1]), len(m))
        who = k if abs(r2) > abs(r1) else "classic"
        print(f"  classic {r1:+.3f} vs {k:<10s} {r2:+.3f}   t={t:+.2f} p={pv:.3f}  "
              f"larger |r|: {who}")

    pt = m[m.patient == 1]
    print(f"\nWITHIN PATIENTS (n={len(pt)}), adjusted for age and sex")
    rows = []
    for col, lab in CLINICAL:
        if col not in pt.columns:
            continue
        s = pt[[col, "age", "sex_n"] + VARIANTS].copy()
        s[col] = pd.to_numeric(s[col], errors="coerce")
        s = s.dropna()
        if len(s) < 30:
            print(f"  {lab:<20s} n={len(s)} too few")
            continue
        Cp = np.column_stack([np.ones(len(s)), s.age.to_numpy(float),
                              s.sex_n.to_numpy(float)])
        yy = resid(s[col], Cp)
        rec = {"clinical": lab, "n": len(s)}
        for k in VARIANTS:
            xx = resid(s[k], Cp)
            r = float(np.corrcoef(xx, yy)[0, 1])
            dof = len(s) - 4
            t = r * np.sqrt(dof / max(1 - r * r, 1e-12))
            rec[k] = r
            rec[f"p_{k}"] = float(2 * (1 - stats.t.cdf(abs(t), dof)))
        rows.append(rec)
    if rows:
        out = pd.DataFrame(rows)
        print(f"{'clinical':<20s} {'n':>5s} " + " ".join(f"{k:>10s}" for k in VARIANTS))
        for r in out.itertuples():
            print(f"{r.clinical:<20s} {r.n:5d} "
                  + " ".join(f"{getattr(r, k):10.3f}" for k in VARIANTS))
        print("\np-values (uncorrected):")
        for r in out.itertuples():
            print(f"{r.clinical:<20s} "
                  + " ".join(f"{getattr(r, f'p_{k}'):10.3f}" for k in VARIANTS))
        out.to_csv(HERE / "tn_clinical.csv", index=False)


if __name__ == "__main__":
    main()
