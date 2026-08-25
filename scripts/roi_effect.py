"""Is the region of interest actually affecting the measurement, or not?

Two results look contradictory. Adjusting for region volume used to remove a
third of the age association, which reads as a large region effect. But fixing
region size at source, which is what the redrawn spheres do, barely moved the
index at all: the two placements agree at r=0.97 and the age associations differ
in the third decimal.

Both cannot be describing the same thing. If region size were really carrying a
third of the age signal, fixing it would have changed the answer.

The resolution is testable. Region volume is the count of voxels surviving the
anisotropy criterion, so it falls with age for two reasons that have nothing to
do with the region being wrong: FA declines, and the warp of an atrophied brain
delivers a different mask. A covariate that tracks age for those reasons removes
real age signal when it is adjusted for. That is over-adjustment, not confound
control, and it would produce a large apparent region effect while the actual
measurement was never disturbed.

This separates the two by asking four questions:

  1. does region volume track age, under each placement
  2. how much does the index itself move between placements
  3. does the age association move between placements
  4. how much does adjusting for volume cost, under each placement

A genuine region artifact predicts that (2) and (3) are large. Over-adjustment
predicts that (2) and (3) are near zero while (4) is large under the placement
whose size varies most.

    python roi_effect.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
DIFF = HERE.parent.parent / "diffusion"
IDX = ["classic", "cross", "v2_slab", "pv_perp", "anat_x"]


def one_per(d):
    return (d.sort_values(["Subject_ID", "Visit"])
             .groupby("Subject_ID").first().reset_index())


def partial(y, x, z):
    ok = ~(np.isnan(y) | np.isnan(x) | np.isnan(z))
    y, x, z = y[ok], x[ok], z[ok]
    A = np.column_stack([np.ones(len(z)), z])

    def rz(v):
        b, *_ = np.linalg.lstsq(A, v, rcond=None)
        return v - A @ b
    return float(np.corrcoef(rz(y), rz(x))[0, 1])


def load(cohort, warped):
    stem = ("measured_pvs_axis_hcpa_b1500_all" if cohort == "hcpa"
            else "measured_pvs_axis_dlbs")
    d = pd.read_csv(HERE / f"{stem}{'_warpedmask' if warped else ''}.csv")
    d["Subject_ID"] = d.Subject_ID.astype(str)
    d["Visit"] = d.Visit.astype(str)
    if warped:
        sp = pd.read_csv(DIFF / ("HCP/hcpa_alps_spheres_5mm.csv" if cohort == "hcpa"
                                 else "DLBS/dlbs_alps_spheres_5mm.csv"))
        sp["Subject_ID"] = sp.Subject_ID.astype(str)
        sp["Visit"] = (sp.Visit if "Visit" in sp.columns else sp.Session).astype(str)
        sp["nvox"] = (pd.to_numeric(sp.n_proj, errors="coerce")
                      + pd.to_numeric(sp.n_assoc, errors="coerce"))
        d = d.merge(sp[["Subject_ID", "Visit", "nvox"]], on=["Subject_ID", "Visit"],
                    how="left")
    else:
        d["nvox"] = d.n_proj + d.n_assoc
    return d


def main() -> None:
    argparse.ArgumentParser().parse_args()
    rows = []
    for cohort in ("hcpa", "dlbs"):
        W, S = one_per(load(cohort, True)), one_per(load(cohort, False))
        print(f"\n{'=' * 68}\n{cohort.upper()}   {len(S)} participants\n{'=' * 68}")

        print("\n1. Does region volume track age?")
        for lab, d in (("warped mask", W), ("redrawn sphere", S)):
            s = d[["nvox", "Age"]].dropna()
            r, p = stats.pearsonr(s.nvox, s.Age)
            cv = s.nvox.std() / s.nvox.mean() * 100
            print(f"   {lab:<16s} size CV {cv:5.1f}%   r with age {r:+.3f}  p={p:.2g}")
            rows.append({"cohort": cohort, "placement": lab, "quantity":
                         "volume vs age", "value": r, "size_cv_pct": cv})

        print("\n2. How much does the index itself move between placements?")
        m = W.merge(S, on="Subject_ID", suffixes=("_w", "_s"))
        for v in IDX:
            a, b = m[f"{v}_w"], m[f"{v}_s"]
            ok = a.notna() & b.notna()
            r = float(np.corrcoef(a[ok], b[ok])[0, 1])
            print(f"   {v:<10s} r(warped, redrawn) = {r:.4f}   "
                  f"mean shift {(b[ok] - a[ok]).mean():+.4f}")
            rows.append({"cohort": cohort, "placement": "both", "quantity":
                         f"agreement {v}", "value": r})

        print("\n3. Does the age association move between placements?")
        for v in IDX:
            rw = stats.pearsonr(*W[[v, "Age"]].dropna().T.to_numpy())[0]
            rs = stats.pearsonr(*S[[v, "Age"]].dropna().T.to_numpy())[0]
            print(f"   {v:<10s} warped {rw:+.3f}   redrawn {rs:+.3f}   "
                  f"difference {abs(rs - rw):.3f}")
            rows.append({"cohort": cohort, "placement": "both", "quantity":
                         f"age r shift {v}", "value": abs(rs - rw)})

        print("\n4. What does adjusting for volume cost, under each placement?")
        for lab, d in (("warped mask", W), ("redrawn sphere", S)):
            for v in ("classic", "cross"):
                s = d[[v, "Age", "nvox"]].dropna()
                raw = float(stats.pearsonr(s[v], s.Age)[0])
                adj = partial(s[v].to_numpy(float), s.Age.to_numpy(float),
                              s.nvox.to_numpy(float))
                print(f"   {lab:<16s} {v:<8s} {raw:+.3f} -> {adj:+.3f}   "
                      f"({(raw - adj) / raw * 100:5.1f}% of the coefficient)")
                rows.append({"cohort": cohort, "placement": lab, "quantity":
                             f"volume adjustment {v}", "value": (raw - adj) / raw * 100})

    pd.DataFrame(rows).to_csv(HERE / "roi_effect.csv", index=False)
    print(f"\n{'=' * 68}")
    print("Read questions 2 and 3 first. If the index and its age association")
    print("barely move when the region is fixed, the region was not corrupting")
    print("the measurement, whatever the volume covariate appeared to show.")
    print("\n   wrote roi_effect.csv")


if __name__ == "__main__":
    main()
