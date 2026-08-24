"""Does the beyond-the-ratio result hold once the age association is adjusted?

tbl:beyond reports plain Pearson correlations with age, before and after
partialling out the eigenvalue ratio. That is the right shape for the question
it asks, since both columns use the same model and the comparison between them
is internal. But the paper adjusts its other age associations for sex, motion,
region volume, site and scanner, so reporting this one unadjusted holds the
central claim to a lower standard than the supporting ones.

Covariates enter in nested arms, each a named alternative:

    ratio only        what the table currently reports
    + sex             ALPS differs by sex, and head size accounts for much of
                      that difference
    + site, scanner   HCP-A is four sites on six scanners
    + motion          movement degrades the tensor and the registration
    + region volume   the largest single adjustment in the age models
    + registration    scale, anisotropy and shear of the subject-to-template
                      affine, which the estimated axes inherit

The question is not whether the coefficients move. Adding covariates always
moves them. It is whether the pattern holds: variants that estimate the second
eigenvector fall to near zero, and those whose axes miss it keep a residual.

    python beyond_ratio_adjusted.py --cohort hcpa
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
DIFF = HERE.parent.parent / "diffusion"
VARIANTS = ["classic", "cross", "v2_slab", "anat_x", "pv_perp",
            "ALPS-PAS", "per-voxel", "LD-ALPS"]


def dummies(s):
    d = pd.get_dummies(s.astype(str), drop_first=True)
    return d.to_numpy(float) if d.shape[1] else np.empty((len(s), 0))


def partial(y, x, C):
    ok = ~(np.isnan(y) | np.isnan(x) | np.isnan(C).any(axis=1))
    y, x, C = y[ok], x[ok], C[ok]
    if len(y) < 30:
        return np.nan, np.nan, len(y)
    A = np.column_stack([np.ones(len(C)), C])

    def rz(v):
        b, *_ = np.linalg.lstsq(A, v, rcond=None)
        return v - A @ b
    ry, rx = rz(y), rz(x)
    # A variant that IS one of the covariates, as pv_perp is the ratio, leaves
    # only floating-point residue after regression. Correlating that residue
    # against age returns an arbitrary number, and at n=809 it can clear
    # p<0.05. Testing against zero misses it, because the residue is small but
    # never exactly zero. Compare it to the variable's own spread instead, so
    # a collapsed residual reports as undefined rather than as a finding.
    if np.std(ry) <= 1e-8 * max(np.std(y), 1e-30) or np.std(rx) == 0:
        return np.nan, np.nan, len(y)
    r = float(np.corrcoef(ry, rx)[0, 1])
    dof = len(y) - A.shape[1] - 1
    t = r * np.sqrt(dof / max(1 - r ** 2, 1e-12))
    return r, float(2 * stats.t.sf(abs(t), dof)), len(y)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=["hcpa", "dlbs"], default="hcpa")
    ap.add_argument("--warped", action="store_true",
                    help="use the first submission's warped masks instead of "
                         "the primary re-sphered regions")
    ap.add_argument("--input", default=None,
                    help="explicit index file, overriding --cohort and --sphere")
    args = ap.parse_args()
    hcpa = args.cohort == "hcpa"

    name = ("measured_pvs_axis_hcpa_b1500_all" if hcpa
            else "measured_pvs_axis_dlbs") + ("_warpedmask" if args.warped else "")
    src = Path(args.input) if args.input else HERE / f"{name}.csv"
    print(f"indices from {src.name}")
    base = pd.read_csv(src)
    base["Subject_ID"] = base.Subject_ID.astype(str)
    base["Visit"] = base.Visit.astype(str)
    base["ratio"] = base["pv_perp"]

    comp = pd.read_csv(HERE / f"comparators_{args.cohort}.csv")
    comp["Subject_ID"] = comp.Subject_ID.astype(str)
    comp["Visit"] = comp.Visit.astype(str)
    m = base.merge(comp[["Subject_ID", "Visit", "ALPS-PAS", "per-voxel"]],
                   on=["Subject_ID", "Visit"], how="left")

    ld = HERE / f"ld_alps_{args.cohort}.csv"
    if ld.exists():
        L = pd.read_csv(ld)
        if {"Subject_ID", "Visit", "ALPS_overall"} <= set(L.columns):
            L = L[["Subject_ID", "Visit", "ALPS_overall"]].rename(
                columns={"ALPS_overall": "LD-ALPS"})
            L["Subject_ID"] = L.Subject_ID.astype(str)
            L["Visit"] = L.Visit.astype(str)
            m = m.merge(L, on=["Subject_ID", "Visit"], how="left")

    sp = pd.read_csv(DIFF / ("HCP/hcpa_alps_spheres_5mm.csv" if hcpa
                             else "DLBS/dlbs_alps_spheres_5mm.csv"))
    sp["Subject_ID"] = sp.Subject_ID.astype(str)
    sp["Visit"] = (sp.Visit if "Visit" in sp.columns else sp.Session).astype(str)
    sp["nvox"] = (pd.to_numeric(sp.n_proj, errors="coerce")
                  + pd.to_numeric(sp.n_assoc, errors="coerce"))
    keep = ["Subject_ID", "Visit", "Sex", "nvox"] + (["site", "scanner"] if hcpa else [])
    m = m.merge(sp[keep], on=["Subject_ID", "Visit"], how="left")

    # The sphere file records the warped-mask sizes. Those are the right sizes
    # for the published indices and the wrong ones for the re-sphered indices,
    # which were measured over different voxels entirely. When the index file
    # carries its own counts, they win, so the covariate always describes the
    # regions the indices actually came from.
    if {"n_proj", "n_assoc"} <= set(base.columns):
        m["nvox"] = m.n_proj + m.n_assoc
        print("   region volume from the index file's own counts")
    else:
        print("   region volume from the sphere file (warped-mask sizes)")

    mot = DIFF / ("HCP/hcpa_motion.csv" if hcpa else "DLBS/dlbs_motion.csv")
    if mot.exists():
        mo = pd.read_csv(mot)
        if {"Subject_ID", "Visit"} <= set(mo.columns):
            mo["Subject_ID"] = mo.Subject_ID.astype(str)
            mo["Visit"] = mo.Visit.astype(str)
            m = m.merge(mo[["Subject_ID", "Visit", "Eddy_Mean_RMS"]],
                        on=["Subject_ID", "Visit"], how="left")
    rq = HERE / f"registration_quality_{args.cohort}.csv"
    if rq.exists():
        q = pd.read_csv(rq)
        q["Subject_ID"] = q.Subject_ID.astype(str)
        q["Visit"] = q.Visit.astype(str)
        m = m.merge(q[["Subject_ID", "Visit", "det", "aniso", "shear"]],
                    on=["Subject_ID", "Visit"], how="left")

    m = m.sort_values(["Subject_ID", "Visit"]).groupby("Subject_ID").first().reset_index()
    m["sex_n"] = (m.Sex.astype(str).str.upper().str[0] == "M").astype(float)
    m["Age"] = pd.to_numeric(m.Age, errors="coerce")
    print(f"{args.cohort}: {len(m)} participants\n")

    # Two ladders. The first adjusts for nuisances only, so it shows the age
    # association each variant has net of them. The second adds the ratio on
    # top, so the difference between the two is what the ratio accounts for
    # rather than what the covariates do.
    ratio = m.ratio.to_numpy(float)[:, None]
    empty = np.empty((len(m), 0))
    nuis = [("none", empty), ("sex", np.column_stack([m.sex_n]))]
    if hcpa:
        nuis.append(("sex, site, scanner", np.column_stack(
            [m.sex_n, dummies(m.site), dummies(m.scanner)])))
    _p = nuis[-1][1]
    if "Eddy_Mean_RMS" in m.columns:
        _p = np.column_stack([_p, pd.to_numeric(m.Eddy_Mean_RMS, errors="coerce")])
        nuis.append(("+ motion", _p))
    _p = np.column_stack([_p, m.nvox.to_numpy(float)])
    nuis.append(("+ region volume", _p))
    if "det" in m.columns:
        _p = np.column_stack([_p, m.det, m.aniso, m.shear])
        nuis.append(("+ registration", _p))

    print("=== age association WITHOUT the ratio partialled out ===")
    print(f"{'variant':<14s}" + "".join(f"{a:>17s}" for a, _ in nuis))
    for v in [x for x in VARIANTS if x in m.columns and m[x].notna().sum() > 50]:
        cells = []
        for name, C in nuis:
            r, p, n = partial(m[v].to_numpy(float), m.Age.to_numpy(float),
                              C if C.shape[1] else np.zeros((len(m), 1)))
            cells.append(f"{r:+.3f}{'*' if p < 0.05 else ' '}")
        print(f"{v:<14s}" + "".join(f"{c:>17s}" for c in cells))
    print()

    arms = [("ratio only", ratio),
            ("+ sex", np.column_stack([ratio, m.sex_n]))]
    if hcpa:
        arms.append(("+ site, scanner", np.column_stack(
            [ratio, m.sex_n, dummies(m.site), dummies(m.scanner)])))
    prev = arms[-1][1]
    if "Eddy_Mean_RMS" in m.columns:
        prev = np.column_stack([prev, pd.to_numeric(m.Eddy_Mean_RMS,
                                                    errors="coerce")])
        arms.append(("+ motion", prev))
    prev = np.column_stack([prev, m.nvox.to_numpy(float)])
    arms.append(("+ region volume", prev))
    if "det" in m.columns:
        prev = np.column_stack([prev, m.det, m.aniso, m.shear])
        arms.append(("+ registration", prev))

    have = [v for v in VARIANTS if v in m.columns and m[v].notna().sum() > 50]
    rows = []
    print("=== age association WITH the ratio partialled out ===")
    print(f"{'variant':<14s}" + "".join(f"{a:>17s}" for a, _ in arms))
    for v in have:
        cells = []
        for name, C in arms:
            r, p, n = partial(m[v].to_numpy(float), m.Age.to_numpy(float), C)
            cells.append("   identical " if np.isnan(r)
                         else f"{r:+.3f}{'*' if p < 0.05 else ' '}")
            rows.append({"variant": v, "arm": name, "n": n, "r": r, "p": p})
        print(f"{v:<14s}" + "".join(f"{c:>17s}" for c in cells))
    print("\n   * p<0.05.  Age association after partialling the ratio and the")
    print("   listed covariates. A variant carrying nothing beyond the ratio")
    print("   should sit near zero in every column. 'identical' marks a")
    print("   variant that is the ratio itself, leaving no residual variance.")
    tag = args.cohort + ("_warpedmask" if args.warped else "")
    pd.DataFrame(rows).to_csv(HERE / f"beyond_ratio_adjusted_{tag}.csv",
                              index=False)
    print(f"\n   wrote beyond_ratio_adjusted_{tag}.csv")


if __name__ == "__main__":
    main()
