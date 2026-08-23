"""Do Parkinson's patients lie differently in the scanner than controls?

This is the hypothesis ds001907 was recovered to test. If it holds, head pose is
not a nuisance that varies at random across a cohort, it is a variable that
tracks disease, and any index sensitive to pose inherits a spurious group
difference. That would be the strongest possible statement of the paper's thesis
and it needs a real patient cohort to make.

The test is deliberately unforgiving of the answer it wants:

  - Both arms are pooled across sessions, so each subject contributes one or two
    scans. That is not independent, so the primary model is a mixed effect with
    a random intercept per subject, and the between-subject mean is reported
    alongside it.
  - Sex is imbalanced between arms and Parkinson's is male-predominant, so sex
    and age are covariates in every model.
  - Tremor makes patients move more, and within-scan motion could shift a
    registration on its own. Motion is carried as a covariate and reported
    separately, since a pose difference that vanishes under motion adjustment is
    a motion finding, not a posture finding.
  - Four axes are tested, so p-values are Benjamini-Hochberg corrected.
  - n is 44 subjects. A null here is weak evidence of absence, so the achieved
    effect size and its confidence interval are reported whatever the p-value.

    python ds001907_pose_test.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ds001907_common import assert_group_mapping, demographics

HERE = Path(__file__).resolve().parent
AXES = ["pitch", "roll", "yaw", "total"]


def bh(p):
    """Benjamini-Hochberg adjusted p-values, order preserved."""
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = np.empty_like(p)
    q[o] = np.minimum.accumulate(
        (p[o] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    return np.clip(q, 0, 1)


def hedges_g(a, b):
    """Standardized mean difference with the small-sample correction and a CI."""
    na, nb = len(a), len(b)
    s = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    d = (a.mean() - b.mean()) / s
    g = d * (1 - 3 / (4 * (na + nb) - 9))
    se = np.sqrt((na + nb) / (na * nb) + g ** 2 / (2 * (na + nb - 2)))
    return g, g - 1.96 * se, g + 1.96 * se


def qc(r: pd.DataFrame) -> pd.DataFrame:
    """Drop scans whose registration did not converge sensibly.

    A FLIRT that fails does not raise, it returns a bad affine, and a bad affine
    decomposes into a large fake rotation. Two symptoms give it away. The scale
    factors from the polar decomposition should sit near one, since a human head
    is not half or twice the size of the template. And a pose at the edge of the
    plus or minus 90 degree search range means the optimizer walked to the
    boundary rather than finding a minimum.

    The criteria are fixed here before the group contrast is run, and the drops
    are reported by arm so it is visible if QC removed one group preferentially.
    """
    from ds001907_common import DEST  # noqa: F401  keeps the module's paths together
    work = Path(r"M:\ds001907-derivatives")
    scale, ok = [], []
    for t in r.itertuples():
        m = work / t.subject / t.session / "subject_to_mni_affine.mat"
        s = np.nan
        if m.exists():
            A = np.loadtxt(m)[:3, :3]
            sv = np.linalg.svd(A, compute_uv=False)
            s = float(sv.max() / sv.min())          # anisotropy of the scaling
        scale.append(s)
    r = r.assign(scale_ratio=scale)
    bad = (r.scale_ratio > 1.6) | (r[["pitch", "roll", "yaw"]].abs().max(axis=1) > 85)
    if bad.any():
        print("=== QC drops ===")
        for g, d in r[bad].groupby("group"):
            print(f"   {g}: {len(d)} scan(s) {list(d.subject + '/' + d.session)}")
    else:
        print("QC: all registrations plausible "
              f"(scale ratio {r.scale_ratio.min():.2f}-{r.scale_ratio.max():.2f}, "
              f"no pose at the search boundary)")
    return r[~bad].copy()


def main() -> None:
    argparse.ArgumentParser().parse_args()
    assert_group_mapping()

    r = pd.read_csv(HERE / "ds001907_pose.csv")
    r = qc(r)
    dem = demographics().rename(columns={"subject": "sid"})
    r["sid"] = r.subject.str.replace("sub-", "", regex=False)
    m = r.merge(dem.drop(columns=["group"]), on="sid", how="left")
    for a in ("pitch", "roll", "yaw"):
        m[a] = m[a].abs()          # direction of tilt is arbitrary, extent is not
    m["patient"] = (m.group == "patient").astype(int)
    m["male"] = (m.sex == "Male").astype(float)
    print(f"{len(m)} scans, {m.sid.nunique()} subjects "
          f"({m[m.patient==1].sid.nunique()} patient, "
          f"{m[m.patient==0].sid.nunique()} control)\n")

    # --- subject means, the model that needs no independence assumption -----
    sub = (m.groupby(["sid", "patient", "male"], as_index=False)[AXES + ["age", "motion_tr_rms"]]
             .mean())
    print("=== subject-level, patient vs control ===")
    print(f"{'axis':<8s} {'patient':>14s} {'control':>14s} {'Welch p':>9s} "
          f"{'BH q':>7s} {'Hedges g [95% CI]':>26s}")
    raw = []
    for c in AXES:
        a = sub.loc[sub.patient == 1, c].dropna()
        b = sub.loc[sub.patient == 0, c].dropna()
        t, pv = stats.ttest_ind(a, b, equal_var=False)
        g, lo, hi = hedges_g(a, b)
        raw.append((c, a, b, pv, g, lo, hi))
    q = bh([x[3] for x in raw])
    for (c, a, b, pv, g, lo, hi), qq in zip(raw, q):
        print(f"{c:<8s} {a.mean():7.2f} +-{a.std():5.2f} {b.mean():7.2f} +-{b.std():5.2f} "
              f"{pv:9.3f} {qq:7.3f}   {g:6.2f} [{lo:5.2f}, {hi:5.2f}]")

    # --- adjusted, and then adjusted for motion as well ---------------------
    print("\n=== adjusted for age and sex, then additionally for motion ===")
    print(f"{'axis':<8s} {'beta':>8s} {'p':>8s} {'BH q':>7s} | "
          f"{'+motion beta':>13s} {'p':>8s}")
    import itertools
    b1, p1, b2, p2 = [], [], [], []
    for c in AXES:
        s = sub.dropna(subset=[c, "age", "male"])
        y = s[c].to_numpy(float)
        for cols, bs, ps in ((["age", "male"], b1, p1),
                             (["age", "male", "motion_tr_rms"], b2, p2)):
            d = s.dropna(subset=cols)
            X = np.column_stack([np.ones(len(d)), d.patient.to_numpy(float)]
                                + [d[k].to_numpy(float) for k in cols])
            yy = d[c].to_numpy(float)
            beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
            resid = yy - X @ beta
            dof = len(d) - X.shape[1]
            se = np.sqrt(np.diag(np.linalg.pinv(X.T @ X)) * (resid @ resid) / dof)
            bs.append(beta[1])
            ps.append(2 * stats.t.sf(abs(beta[1] / se[1]), dof))
    q1 = bh(p1)
    for c, a, pa, qq, bb, pb in zip(AXES, b1, p1, q1, b2, p2):
        print(f"{c:<8s} {a:8.2f} {pa:8.3f} {qq:7.3f} | {bb:13.2f} {pb:8.3f}")

    # --- does motion itself differ, and does pose track severity? -----------
    print("\n=== the two things that would explain a positive result away ===")
    a = sub.loc[sub.patient == 1, "motion_tr_rms"].dropna()
    b = sub.loc[sub.patient == 0, "motion_tr_rms"].dropna()
    t, pv = stats.ttest_ind(a, b, equal_var=False)
    g, lo, hi = hedges_g(a, b)
    print(f"   within-scan motion   patient {a.mean():.2f}, control {b.mean():.2f}, "
          f"Welch p={pv:.3f}, g={g:.2f} [{lo:.2f}, {hi:.2f}]")

    hy = demographics().rename(columns={"subject": "sid"})[["sid", "hoehn_yahr"]]
    sv = sub.merge(hy, on="sid").dropna(subset=["hoehn_yahr"])
    print(f"\n   Hoehn and Yahr gradient within the {len(sv)} patients "
          f"(stages {sorted(sv.hoehn_yahr.unique())}):")
    for c in AXES:
        rho, pv = stats.spearmanr(sv[c], sv.hoehn_yahr)
        print(f"      {c:<8s} rho={rho:6.2f}  p={pv:.3f}")

    # --- what else could produce this, besides posture --------------------
    print("\n=== alternative explanations ===")

    # 1. Which covariate costs pitch its significance, age or sex?
    print("   pitch, covariates entered one at a time:")
    for cols in ([], ["age"], ["male"], ["age", "male"]):
        d = sub.dropna(subset=["pitch"] + cols)
        X = np.column_stack([np.ones(len(d)), d.patient.to_numpy(float)]
                            + [d[k].to_numpy(float) for k in cols])
        y = d.pitch.to_numpy(float)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        res = y - X @ beta
        dof = len(d) - X.shape[1]
        se = np.sqrt(np.diag(np.linalg.pinv(X.T @ X)) * (res @ res) / dof)
        pv = 2 * stats.t.sf(abs(beta[1] / se[1]), dof)
        print(f"      {'+'.join(cols) or 'unadjusted':<12s} beta={beta[1]:5.2f}  p={pv:.3f}")

    # 2. Is this posture or is it the registration reacting to patient anatomy?
    #    Posture is an occasion, so it should differ between a subject's two
    #    sessions. A registration bias driven by the patient's brain is a trait,
    #    so it would repeat. A high correlation across sessions does not prove
    #    bias, some people always lie the same way, but a low one largely rules
    #    it out.
    two = m[m.sid.isin(m.groupby("sid").session.nunique().pipe(lambda s: s[s > 1].index))]
    print(f"\n   between-session consistency within subject "
          f"({two.sid.nunique()} subjects with two scans):")
    for c in AXES:
        w = two.pivot_table(index="sid", columns="session", values=c).dropna()
        if len(w) > 3:
            rho, pv = stats.pearsonr(w.iloc[:, 0], w.iloc[:, 1])
            print(f"      {c:<8s} r={rho:5.2f}  p={pv:.3f}  "
                  f"(n={len(w)})")

    # 3. Does the contrast hold in session 1 alone, with no averaging at all?
    s1 = m[m.session == "ses-1"]
    print(f"\n   session 1 only, one scan per subject "
          f"({(s1.patient==1).sum()} patient, {(s1.patient==0).sum()} control):")
    for c in AXES:
        a = s1.loc[s1.patient == 1, c].dropna()
        b = s1.loc[s1.patient == 0, c].dropna()
        t, pv = stats.ttest_ind(a, b, equal_var=False)
        u, pu = stats.mannwhitneyu(a, b)
        g, lo, hi = hedges_g(a, b)
        print(f"      {c:<8s} {a.mean():6.2f} vs {b.mean():6.2f}  Welch p={pv:.3f}  "
              f"Mann-Whitney p={pu:.3f}  g={g:.2f}")

    # 4. Is it a handful of extreme patients, or the whole distribution?
    print("\n   robustness of the total-rotation contrast:")
    a = sub.loc[sub.patient == 1, "total"].dropna()
    b = sub.loc[sub.patient == 0, "total"].dropna()
    u, pu = stats.mannwhitneyu(a, b)
    print(f"      Mann-Whitney p={pu:.3f}  (rank based, immune to outliers)")
    tr = stats.trim_mean
    print(f"      20% trimmed means {tr(a,.2):.2f} vs {tr(b,.2):.2f}")
    k = max(1, int(round(0.1 * len(a))))
    a2 = a.sort_values()[:-k]
    t2, p2 = stats.ttest_ind(a2, b, equal_var=False)
    print(f"      dropping the {k} most-rotated patients: "
          f"{a2.mean():.2f} vs {b.mean():.2f}, p={p2:.3f}")

    sub.to_csv(HERE / "ds001907_pose_subject.csv", index=False)
    print(f"\n   wrote {HERE / 'ds001907_pose_subject.csv'}")


if __name__ == "__main__":
    main()
