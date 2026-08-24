"""Does head position covary with anything besides age?

The paper's confound argument needs head pose to be non-random with respect to
the variables a study cares about. That is shown for age. Whether it extends to
clinical and biomarker variables is a separate question, and it is the one that
decides whether the Introduction can claim more than age.

The test is pose against each phenotype, adjusted for age and sex. Age
adjustment is the point rather than a formality: pose covaries with age and so
does almost every phenotype in an aging cohort, so an unadjusted correlation
between them is guaranteed and means nothing. What matters is whether pose
tracks a phenotype over and above age.

The two cohorts answer different halves and neither answers both.

  DLBS  obliquely acquired, so it retains real positioning variation, but its
        released participants file carries only MMSE, BMI, education,
        handedness and the ages at each wave. It has no amyloid or tau values,
        despite the study having collected PET.
  HCP-A rich phenotyping, 223 repeated measures including neurofilament light,
        inflammatory markers and the full cognitive battery, but anatomical
        alignment during preprocessing removed most of the head position, so
        the exposure is small by construction.

A null in HCP-A is therefore expected and is itself worth stating, because it
is what "alignment removes the exposure" predicts. A positive in DLBS would be
the stronger result despite the thinner phenotyping.

    python pose_phenotype.py --cohort dlbs
    python pose_phenotype.py --cohort hcpa
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
DIFF = HERE.parent.parent / "diffusion"
AABC = DIFF / "HCP" / "AABC2_subjects_2026_02_05_14_29_11.csv"
DLBS_TSV = DIFF / "DLBS" / "ds004856_participants.tsv"
# Identifiers, dates and ages are excluded for the usual reasons. So are the
# wave-interval columns, CogW1toW2 and its siblings, which record how long
# elapsed between a participant's assessments. They are study logistics rather
# than phenotypes, and they dominate the DLBS sweep if left in: pose correlates
# with them at up to r=-0.38, which is a statement about who came back and when,
# not about what head position tracks.
DROP = re.compile(r"id|date|visit|event|site|scanner|version|complete|_dt$|days|^age"
                  r"|W\dto W?\d|W\dto\w?\d", re.I)
POSE = ["abs_pitch", "abs_roll", "abs_yaw", "total"]


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


def partial(x, y, C):
    """Partial correlation of x and y given the columns of C."""
    ok = ~(np.isnan(x) | np.isnan(y) | np.isnan(C).any(axis=1))
    x, y, C = x[ok], y[ok], C[ok]
    if len(x) < 30 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan, len(x)
    A = np.column_stack([np.ones(len(C)), C])

    def rz(v):
        b, *_ = np.linalg.lstsq(A, v, rcond=None)
        return v - A @ b
    rx, ry = rz(x), rz(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return np.nan, np.nan, len(x)
    r = float(np.corrcoef(rx, ry)[0, 1])
    dof = len(x) - A.shape[1] - 1
    t = r * np.sqrt(dof / max(1 - r ** 2, 1e-12))
    return r, float(2 * stats.t.sf(abs(t), dof)), len(x)


def load(cohort):
    hr = pd.read_csv(HERE / f"head_rotation_{cohort}.csv")
    hr["Subject_ID"] = hr.Subject_ID.astype(str)
    hr["Visit"] = hr.Visit.astype(str)
    for a in ("pitch", "roll", "yaw"):
        hr[f"abs_{a}"] = hr[a].abs()
    f = ("measured_pvs_axis_hcpa_b1500_all.csv" if cohort == "hcpa"
         else "measured_pvs_axis_dlbs.csv")
    d = pd.read_csv(HERE / f)
    d["Subject_ID"] = d.Subject_ID.astype(str)
    d["Visit"] = d.Visit.astype(str)
    m = hr.merge(d[["Subject_ID", "Visit", "Age"]], on=["Subject_ID", "Visit"])
    # One session per participant: pose is a session property but the phenotypes
    # here are participant-level, and repeat visits are not independent.
    m = m.sort_values(["Subject_ID", "Visit"]).groupby("Subject_ID").first().reset_index()

    if cohort == "hcpa":
        a = pd.read_csv(AABC, low_memory=False)
        a["Subject_ID"] = a.id_event.astype(str).str.split("_").str[0]
        num = [c for c in a.columns
               if pd.api.types.is_numeric_dtype(a[c]) and not DROP.search(c)
               and a[c].notna().sum() >= 100 and a[c].nunique() > 4]
        ph = a.groupby("Subject_ID")[num].first().reset_index()
        sx = a.groupby("Subject_ID")["sex"].first().reset_index()
        m = m.merge(ph, on="Subject_ID", how="inner").merge(sx, on="Subject_ID",
                                                            how="left")
        m["sex_n"] = (m.sex.astype(str).str.upper().str[0] == "M").astype(float)
    else:
        t = pd.read_csv(DLBS_TSV, sep="\t", low_memory=False)
        t["Subject_ID"] = t.participant_id.astype(str)
        m = m.merge(t, on="Subject_ID", how="inner")
        sexcol = next(c for c in m.columns if c.lower() == "sex")
        m["sex_n"] = (m[sexcol].astype(str).str.lower().str[0] == "m").astype(float)
        num = [c for c in t.columns
               if pd.api.types.is_numeric_dtype(t[c]) and not DROP.search(c)]
    return m, [c for c in num if c in m.columns]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=["hcpa", "dlbs"], default="dlbs")
    args = ap.parse_args()

    m, phenos = load(args.cohort)
    print(f"{args.cohort}: {len(m)} participants, {len(phenos)} phenotypes\n")

    # Body habitus is the obvious mechanism, and it is the one phenotype both
    # cohorts share, so a second arm adds BMI to the covariates. Anything that
    # survives it is not simply a consequence of how a larger person lies in a
    # head coil. BMI itself is dropped from the sweep in that arm, since
    # partialling a variable out of itself is degenerate.
    bmi_col = next((c for c in ("bmi", "BMI_W1") if c in m.columns), None)
    arms = {"age+sex": np.column_stack([m.Age.to_numpy(float),
                                        m.sex_n.to_numpy(float)])}
    if bmi_col is not None:
        arms["age+sex+bmi"] = np.column_stack([m.Age.to_numpy(float),
                                               m.sex_n.to_numpy(float),
                                               m[bmi_col].to_numpy(float)])
        print(f"body-habitus covariate: {bmi_col}\n")

    rows = []
    for arm, C in arms.items():
        for pose in POSE:
            for c in phenos:
                if m[c].notna().sum() < 40:
                    continue
                if arm.endswith("bmi") and c in (bmi_col, "BMI_W2",
                                                 "Weight_W1", "Weight_W2",
                                                 "Height_W1", "Height_W2"):
                    continue
                r, p, n = partial(m[pose].to_numpy(float),
                                  m[c].to_numpy(float), C)
                if not np.isnan(r):
                    rows.append({"arm": arm, "pose": pose, "phenotype": c,
                                 "n": n, "r": r, "p": p})
    out = pd.DataFrame(rows)
    if out.empty:
        print("nothing testable")
        return
    out["q"] = out.groupby(["arm", "pose"]).p.transform(lambda s: bh(s.to_numpy()))
    out = out.sort_values(["arm", "pose", "p"])
    out.to_csv(HERE / f"pose_phenotype_{args.cohort}.csv", index=False)

    for arm in out.arm.unique():
        print(f"=== pose against phenotype, given {arm} ===")
        print(f"{'pose':<12s} {'tested':>7s} {'BH q<.05':>9s}   strongest")
        for pose in POSE:
            g = out[(out.arm == arm) & (out.pose == pose)]
            if not len(g):
                continue
            t = g.iloc[0]
            print(f"{pose:<12s} {len(g):7d} {int((g.q<.05).sum()):9d}   "
                  f"{t.phenotype[:32]:<32s} r={t.r:+.3f} q={t.q:.3f}")
        print()

    sig = out[(out.q < 0.05) & (out.arm == "age+sex")]
    print(f"\n{len(sig)} pose-phenotype pairs survive BH correction.")
    if len(sig):
        print(sig.head(20).to_string(index=False))
    else:
        print("Head position does not track any phenotype here beyond age and sex.")
    print(f"\n   wrote pose_phenotype_{args.cohort}.csv")


if __name__ == "__main__":
    main()
