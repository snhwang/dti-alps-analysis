"""Is the transformation a registration applies to the ALPS regions age-dependent?

Reorientation is the recommended fix for the confound this paper describes, and
Section 3.8 finds it works. This asks a different question about it. Not whether
it removes an age-dependent term, but whether the operation it performs is
itself age-dependent, since anything applied differentially across the age range
can carry age into the index by a route that has nothing to do with
perivascular diffusion.

Three quantities, read off the registrations already computed for DLBS, one
session per participant, each of the four region-hemispheres separately.

  1. the rotation the affine applies to the region's measured direction.
     Strongly age-dependent, r about +0.33 everywhere. This is expected and is
     not a criticism. Head pitch tracks age, so a correction that removes head
     pitch must be larger in older participants. It is the confound being
     removed, visible in the correction itself.

  2. the extra rotation contributed by the nonlinear warp beyond the affine.
     Flat with age, r between +0.01 and +0.10, none significant. The warp's
     local rotation is not an age-graded quantity.

  3. the local Jacobian determinant of the atlas-to-subject warp at the region.
     Age-dependent, r about -0.20 to -0.33. The warp is stored template to
     subject, so a smaller determinant means the atlas expands less to reach
     that participant's region, which is what atrophy produces.

The third is the one worth reporting. Reorientation resamples, and it resamples
under a local scaling that varies with age, so the interpolation smoothing and
the partial-volume mixing it induces are not constant across the age range. The
correction is sound but its resampling carries an age-graded term that nobody
reports.

Two things keep this from being an indictment. The residual warp rotation is
age-flat, and Section 3.8 finds reorientation's age association (-0.350)
essentially identical to the closed-form correction's (-0.353), so in these data
reorientation is not manufacturing an age effect. Where it plausibly matters is
between studies rather than within one, since different pipelines apply
different age-graded resampling.

Writes registration_age_dependence.csv.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import atomic_io  # noqa: F401  writes become atomic on import
from scipy import stats

HERE = Path(__file__).resolve().parent


def main() -> None:
    d = pd.read_csv(HERE / "registration_aligns_tracts.csv")
    age = (pd.read_csv(HERE / "measured_pvs_axis_dlbs.csv")[["Subject_ID", "Visit", "Age"]]
             .rename(columns={"Visit": "Session"}))
    d = d.merge(age, on=["Subject_ID", "Session"])

    # The rotation the affine actually applies to the region's direction, as the
    # angle between the native vector and its image under the affine. Axes are
    # sign-ambiguous, so take the acute angle.
    for tag in ("proj", "assoc"):
        n = d[[f"{tag}_native_{c}" for c in "xyz"]].to_numpy()
        a = d[[f"{tag}_affine_{c}" for c in "xyz"]].to_numpy()
        cos = np.abs((n * a).sum(1)) / (np.linalg.norm(n, axis=1) * np.linalg.norm(a, axis=1))
        d[f"{tag}_applied"] = np.degrees(np.arccos(np.clip(cos, -1, 1)))

    one = (d.sort_values(["Subject_ID", "Session"])
             .groupby(["Subject_ID", "hemi"]).first().reset_index())

    rows = []
    print(f"DLBS, one session per participant, {one.Subject_ID.nunique()} participants\n")
    print("  region  quantity                          hemi   median    vs Age r         p")
    for tag in ("proj", "assoc"):
        for q, lab in ((f"{tag}_applied", "rotation applied by affine"),
                       (f"{tag}_local_vs_affine", "extra local warp rotation"),
                       (f"{tag}_jac_det", "local Jacobian determinant")):
            for h in ("L", "R"):
                s = one[one.hemi == h]
                r, p = stats.pearsonr(s[q], s.Age)
                rows.append(dict(region=tag, quantity=q, hemi=h, n=len(s),
                                 median=float(s[q].median()), r_age=float(r), p_age=float(p)))
                print(f"  {tag:6s}  {lab:32s} {h}  {s[q].median():7.3f}  "
                      f"{r:+.3f}  {p:9.2g}{'  *' if p < 0.05 else ''}")

    out = pd.DataFrame(rows)
    out.to_csv(HERE / "registration_age_dependence.csv", index=False)

    def span(q):
        s = out[out.quantity.str.endswith(q)]
        return s.r_age.min(), s.r_age.max(), s.p_age.max()

    print()
    for q, lab in (("applied", "affine rotation applied"),
                   ("local_vs_affine", "extra warp rotation"),
                   ("jac_det", "local Jacobian determinant")):
        lo, hi, pmax = span(q)
        print(f"  {lab:26s} r in [{lo:+.3f}, {hi:+.3f}]  worst p = {pmax:.2g}")
    print("\n  The applied rotation tracks age because head pitch does. That is the")
    print("  confound being removed. The Jacobian tracks age because of atrophy, and")
    print("  that one is a property of the resampling rather than of the correction.")


if __name__ == "__main__":
    main()
