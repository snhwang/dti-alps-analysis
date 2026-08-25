"""Does head position manufacture a group difference in real data?

The paper shows that adjusting for head pose removes a large share of the
classic index's association with age. Most DTI-ALPS literature is not an age
correlation, though. It compares a patient group with a control group, and the
question a reader of that literature has is whether a reported group difference
could be postural.

Until now the paper answered that with a simulation: take the same participants
twice, apply a pitch tilt to one copy, and read off the artifactual group
difference. That answer is exposed to the objection that rotating fitted tensors
reproduces the coordinate change and not the physics of acquiring a differently
angled head.

This answers it with real heads instead. Two phenotypes are known to covary with
head position after age and sex are accounted for, body mass index and total
tau. Splitting the cohort on either gives two groups that genuinely differ in
how they were lying in the scanner, which is exactly the situation a case
control study is in when its groups differ in comfort, rigidity or habitus.

The test is the group coefficient before and after pose adjustment. Three
controls decide whether an attenuation means anything.

  age and sex enter first, always. Pose covaries with age and so does every
  phenotype here, so an unadjusted attenuation would be removing shared age
  variance rather than removing a confound. That is the same error that made
  region volume look like it mattered.

  the pose difference between groups is reported, because an attenuation
  without one would have no mechanism behind it.

  pose is permuted across participants 2000 times, preserving its distribution
  while breaking its link to group and index. Attenuation against that null is
  what separates a confound from an artifact of adding a covariate.

    python pose_group_difference.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
DIFF = HERE.parent.parent / "diffusion"
AABC = DIFF / "HCP" / "AABC2_subjects_2026_02_05_14_29_11.csv"
VARIANTS = ["classic", "cross", "v2_slab", "pv_perp", "anat_x"]
PHENOTYPES = {"hcpa": [("bmi", "body mass index"),
                       ("tTau_Conc_pg_ml", "total tau")],
              # DLBS is the cohort this test needs. HCP-A is anatomically
              # aligned, so its groups differ by about a quarter of a degree,
              # which at the measured slope is a fifth of a per cent and moves
              # nothing. DLBS was acquired obliquely and its head rotation has
              # a median of about eleven degrees.
              "dlbs": [("BMI_W1", "body mass index"),
                       ("MMSE_W1", "mini-mental state examination")]}
RNG = np.random.default_rng(20260825)
NPERM = 2000


def beta(y, X):
    """Standardised coefficient on the first column of X, with its p value."""
    A = np.column_stack([np.ones(len(X)), X])
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ b
    dof = len(y) - A.shape[1]
    if dof < 10:
        return np.nan, np.nan
    s2 = float(resid @ resid) / dof
    XtX_inv = np.linalg.pinv(A.T @ A)
    se = float(np.sqrt(s2 * XtX_inv[1, 1]))
    t = b[1] / se if se > 0 else np.nan
    return float(b[1]), float(2 * stats.t.sf(abs(t), dof))


def load(cohort: str) -> pd.DataFrame:
    if cohort == "dlbs":
        return load_dlbs()
    d = pd.read_csv(HERE / "measured_pvs_axis_hcpa_b1500_all.csv")
    d["Subject_ID"] = d.Subject_ID.astype(str)
    d["Visit"] = d.Visit.astype(str)
    d = (d.sort_values(["Subject_ID", "Visit"])
          .groupby("Subject_ID").first().reset_index())

    r = pd.read_csv(HERE / "head_rotation_hcpa.csv")
    r["Subject_ID"] = r.Subject_ID.astype(str)
    r["abs_pitch"] = r.pitch.abs()
    r = r.groupby("Subject_ID")[["abs_pitch", "total"]].mean().reset_index()
    d = d.merge(r, on="Subject_ID", how="left")

    sp = pd.read_csv(DIFF / "HCP" / "hcpa_alps_spheres_5mm.csv")
    sp["Subject_ID"] = sp.Subject_ID.astype(str)
    d = d.merge(sp.groupby("Subject_ID")["Sex"].first().reset_index(),
                on="Subject_ID", how="left")
    d["sex_n"] = (d.Sex.astype(str).str.upper().str[0] == "M").astype(float)

    a = pd.read_csv(AABC, low_memory=False)
    a["Subject_ID"] = a.id_event.astype(str).str.split("_").str[0]
    keep = [c for c, _ in PHENOTYPES if c in a.columns]
    a = a.groupby("Subject_ID")[keep].first().reset_index()
    a["Subject_ID"] = a.Subject_ID.astype(str)
    return d.merge(a, on="Subject_ID", how="left")


def load_dlbs() -> pd.DataFrame:
    d = pd.read_csv(HERE / "measured_pvs_axis_dlbs.csv")
    d["Subject_ID"] = d.Subject_ID.astype(str)
    d["Visit"] = d.Visit.astype(str)
    d = (d.sort_values(["Subject_ID", "Visit"])
          .groupby("Subject_ID").first().reset_index())
    r = pd.read_csv(HERE / "head_rotation_dlbs.csv")
    r["Subject_ID"] = r.Subject_ID.astype(str)
    r["abs_pitch"] = r.pitch.abs()
    r = r.groupby("Subject_ID")[["abs_pitch", "total"]].mean().reset_index()
    d = d.merge(r, on="Subject_ID", how="left")
    sp = pd.read_csv(DIFF / "DLBS" / "dlbs_alps_spheres_5mm.csv")
    sp["Subject_ID"] = sp.Subject_ID.astype(str)
    d = d.merge(sp.groupby("Subject_ID")["Sex"].first().reset_index(),
                on="Subject_ID", how="left")
    d["sex_n"] = (d.Sex.astype(str).str.upper().str[0] == "M").astype(float)
    t = pd.read_csv(DIFF / "DLBS" / "ds004856_participants.tsv", sep="	",
                    low_memory=False)
    t["Subject_ID"] = t.participant_id.astype(str)
    keep = [c for c, _ in PHENOTYPES["dlbs"] if c in t.columns]
    return d.merge(t[["Subject_ID"] + keep], on="Subject_ID", how="left")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=["hcpa", "dlbs"], default="dlbs")
    args = ap.parse_args()
    d = load(args.cohort)
    rows = []

    for pheno, label in PHENOTYPES[args.cohort]:
        if pheno not in d.columns:
            print(f"\n{label}: not available")
            continue
        for pose_col in ("abs_pitch", "total"):
            s = d.dropna(subset=[pheno, pose_col, "Age", "sex_n"] + VARIANTS[:1])
            if len(s) < 60:
                continue
            hi = (s[pheno] > s[pheno].median()).astype(float).to_numpy()
            age = s.Age.to_numpy(float)
            sex = s.sex_n.to_numpy(float)
            pose = s[pose_col].to_numpy(float)

            # does the split actually produce a pose difference?
            g0, g1 = pose[hi == 0], pose[hi == 1]
            tp = stats.ttest_ind(g1, g0, equal_var=False)
            print(f"\n{'=' * 70}\n{label}, split at the median, pose = {pose_col}"
                  f"\n{'=' * 70}")
            print(f"   n={len(s)}   pose {np.median(g1):.2f} vs {np.median(g0):.2f} deg"
                  f"   Welch p={tp.pvalue:.3g}")
            if tp.pvalue > 0.05:
                print("   groups do not differ in pose, so there is no confound to remove")

            for v in VARIANTS:
                y = s[v].to_numpy(float)
                ok = ~np.isnan(y)
                b0, p0 = beta(y[ok], np.column_stack([hi, age, sex])[ok])
                b1, p1 = beta(y[ok], np.column_stack([hi, age, sex, pose])[ok])
                att = (b0 - b1) / b0 * 100 if b0 else np.nan

                # permutation null for the attenuation
                null = []
                for _ in range(NPERM):
                    pp = RNG.permutation(pose)
                    bb, _ = beta(y[ok], np.column_stack([hi, age, sex, pp])[ok])
                    null.append((b0 - bb) / b0 * 100 if b0 else np.nan)
                null = np.array(null, float)
                pperm = float((np.abs(null) >= abs(att)).mean())

                print(f"   {v:<10s} group beta {b0:+.4f} (p={p0:.3g})"
                      f" -> {b1:+.4f} (p={p1:.3g})   {att:5.1f}% absorbed"
                      f"   perm p={pperm:.4f}")
                rows.append({"phenotype": pheno, "pose": pose_col, "variant": v,
                             "n": int(ok.sum()), "beta_adj": b0, "p_adj": p0,
                             "beta_pose": b1, "p_pose": p1, "absorbed_pct": att,
                             "perm_p": pperm, "pose_diff_p": float(tp.pvalue),
                             "null_mean": float(np.nanmean(null)),
                             "null_p95": float(np.nanpercentile(np.abs(null), 95))})

    out = pd.DataFrame(rows)
    out.to_csv(HERE / f"pose_group_difference_{args.cohort}.csv", index=False)
    print(f"\n   wrote pose_group_difference.csv ({len(out)} rows)")
    print("   Read the classic row against the corrected rows. A confound that")
    print("   acts through the measurement axis should absorb more from classic")
    print("   than from variants whose axes rotate with the tensor.")


if __name__ == "__main__":
    main()
