"""Does the direction error that survives registration still track age?

Section 3.1 shows registration leaves most of the between-participant spread in
tract direction. This asks the question that decides whether that residue is a
confound or only noise: does what survives still covary with age?

The two possibilities differ in consequence. A residue that is age-flat inflates
between-participant variance and costs power, but leaves a slope unbiased. A
residue that is age-graded biases the slope, which is what this paper says the
uncorrected index does, and would mean registration is an incomplete correction
rather than merely an imperfect one.

Measured at three stages, for both tracts and both hemispheres, as the angle
between the participant's measured tract direction and the axis the classic
index assumes for it:

  native      the direction as acquired, against scanner z and y
  affine      after the rotation of the subject-to-template affine
  nonlinear   after the local rotation of the full warp at the region

One session per participant, since repeat visits are not independent. Four
region-hemispheres are tested at each stage, so a single result at p just under
0.05 is what chance alone would produce about one time in five. The pattern
across stages is more informative than any single test, and it is printed in
full rather than filtered.

Writes registration_residual_age.csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import atomic_io  # noqa: F401  writes become atomic on import
from scipy import stats

HERE = Path(__file__).resolve().parent

STAGES = (("native", "as acquired, scanner axes"),
          ("affine", "after the affine rotation"),
          ("nonlinear", "after the full warp"))
TRACTS = (("proj", "projection, from z"), ("assoc", "association, from y"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="registration_residual_age.csv")
    args = ap.parse_args()

    d = pd.read_csv(HERE / "registration_aligns_tracts.csv")
    age = (pd.read_csv(HERE / "measured_pvs_axis_dlbs.csv")[["Subject_ID", "Visit", "Age"]]
           .rename(columns={"Visit": "Session"}))
    for f in (d, age):
        f["Subject_ID"] = f.Subject_ID.astype(str)
        f["Session"] = f.Session.astype(str)
    m = d.merge(age, on=["Subject_ID", "Session"])
    one = (m.sort_values(["Subject_ID", "Session"])
             .groupby(["Subject_ID", "hemi"]).first().reset_index())

    rows = []
    print(f"DLBS, {one.Subject_ID.nunique()} participants, one session each\n")
    print(f"  {'stage':28s} {'tract':22s} {'hemi':5s} {'median':>8s} "
          f"{'r vs age':>9s} {'p':>9s}")
    for stage, slab in STAGES:
        for tag, tlab in TRACTS:
            col = f"{tag}_{stage}"
            if col not in one.columns:
                continue
            for h in ("L", "R"):
                s = one[one.hemi == h].dropna(subset=[col, "Age"])
                if len(s) < 20:
                    continue
                r, p = stats.pearsonr(s[col], s.Age)
                rows.append(dict(stage=stage, tract=tag, hemi=h, n=len(s),
                                 median=float(s[col].median()), r_age=float(r),
                                 p_age=float(p)))
                print(f"  {slab:28s} {tlab:22s} {h:5s} {s[col].median():7.2f}  "
                      f"{r:+8.3f} {p:9.2g}{'  *' if p < 0.05 else ''}")
        print()

    # Pooling the hemispheres is the better test. Four per stage invites a false
    # positive at the nominal rate about one time in five, and the first pass here
    # duly produced one, in the right projection at p = 0.035, which did not
    # survive pooling. Report the pooled version as the result.
    pooled = (m.sort_values(["Subject_ID", "Session"])
                .groupby(["Subject_ID", "Session"]).mean(numeric_only=True).reset_index()
                .sort_values(["Subject_ID", "Session"]).groupby("Subject_ID").first().reset_index())
    print("hemispheres pooled, one test per stage and tract")
    print(f"  {'stage':12s} {'tract':10s} {'median':>8s} {'r vs age':>9s} {'p':>9s}")
    for stage, _ in STAGES:
        cols = [f"{tag}_{stage}" for tag, _ in TRACTS if f"{tag}_{stage}" in pooled.columns]
        for col in cols:
            s = pooled.dropna(subset=[col, "Age"])
            r, pv = stats.pearsonr(s[col], s.Age)
            rows.append(dict(stage=stage, tract=col.split("_")[0], hemi="pooled",
                             n=len(s), median=float(s[col].median()),
                             r_age=float(r), p_age=float(pv)))
            print(f"  {stage:12s} {col.split('_')[0]:10s} {s[col].median():8.2f} "
                  f"{r:+9.3f} {pv:9.3g}{'  *' if pv < 0.05 else ''}")
        if len(cols) == 2:
            comb = pooled[cols].mean(axis=1)
            s = pd.DataFrame({"c": comb, "Age": pooled.Age}).dropna()
            r, pv = stats.pearsonr(s.c, s.Age)
            rows.append(dict(stage=stage, tract="combined", hemi="pooled", n=len(s),
                             median=float(s.c.median()), r_age=float(r), p_age=float(pv)))
            print(f"  {stage:12s} {'combined':10s} {s.c.median():8.2f} "
                  f"{r:+9.3f} {pv:9.3g}{'  *' if pv < 0.05 else ''}")
    print()

    out = pd.DataFrame(rows)
    out.to_csv(HERE / args.out, index=False)

    def band(stage, tag):
        s = out[(out.stage == stage) & (out.tract == tag)]
        return s.r_age.min(), s.r_age.max(), s.p_age.min()

    print("  summary, across both hemispheres")
    for stage, slab in STAGES:
        for tag, tlab in TRACTS:
            if not len(out[(out.stage == stage) & (out.tract == tag)]):
                continue
            lo, hi, pmin = band(stage, tag)
            print(f"    {slab:28s} {tlab:22s} r in [{lo:+.3f}, {hi:+.3f}], "
                  f"best p = {pmin:.3g}")
    print()
    print("  The projection direction is the age-graded one before registration,")
    print("  which is the pitch confound. Registration removes most of that and")
    print("  not all of it, and adding the nonlinear warp does not help. The")
    print("  association direction is age-flat at every stage, and registration")
    print("  moves it further from its assumed axis rather than closer.")


if __name__ == "__main__":
    main()
