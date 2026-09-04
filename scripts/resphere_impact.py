"""What changed when the regions became true spheres, claim by claim.

The first submission measured every index inside the template region warped
into native space. That mask arrives distorted in both size and shape, and its
size varies almost eightfold across HCP-A, which is why the age models carried
a region-volume covariate at all. Reviewer 4 asked about exactly this.

The revision draws the region instead: the warped mask's centre is kept, and a
true sphere of fixed radius is drawn around it in native space. Registration
decides where the region sits, which is what registration is good at, and no
longer decides how big it is or what shape.

Changing the regions changes every number measured inside them. This script
reports the ones the manuscript states, warped against re-sphered, so the
revision can say what moved rather than quietly reprinting different digits.
It reads both placements from disk and computes nothing that the manuscript
does not already claim.

    python resphere_impact.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
DIFF = HERE.parent.parent / "diffusion"


def variance_components(d: pd.DataFrame, col: str) -> dict:
    """Between- and within-participant variance, and their ICC."""
    g = d.groupby("Subject_ID")[col]
    n = g.count()
    keep = n[n >= 2].index
    d = d[d.Subject_ID.isin(keep)]
    if not len(d):
        return {"icc": np.nan}
    g = d.groupby("Subject_ID")[col]
    mu = g.transform("mean")
    within = float(((d[col] - mu) ** 2).sum() / max(len(d) - g.ngroups, 1))
    between = float(g.mean().var(ddof=1))
    tot = between + within
    return {"icc": between / tot if tot > 0 else np.nan}


def one_per(d: pd.DataFrame) -> pd.DataFrame:
    return (d.sort_values(["Subject_ID", "Visit"])
             .groupby("Subject_ID").first().reset_index())


def repeats(d: pd.DataFrame) -> pd.DataFrame:
    return d[d.Subject_ID.isin(d.Subject_ID.value_counts()[lambda s: s >= 2].index)]


def load(cohort: str, warped: bool) -> pd.DataFrame:
    stem = ("measured_pvs_axis_hcpa_b1500_all" if cohort == "hcpa"
            else "measured_pvs_axis_dlbs")
    d = pd.read_csv(HERE / f"{stem}{'' if warped else '_sphere5'}.csv")
    d["Subject_ID"] = d.Subject_ID.astype(str)
    d["Visit"] = d.Visit.astype(str)
    return d


def main() -> None:
    argparse.ArgumentParser().parse_args()
    rows = []

    def rec(claim, cohort, warped, sphered, fmt="{:+.3f}"):
        rows.append({"claim": claim, "cohort": cohort,
                     "warped_mask": warped, "resphered": sphered,
                     "delta": (sphered - warped
                               if isinstance(warped, float)
                               and isinstance(sphered, float) else np.nan)})
        w = fmt.format(warped) if isinstance(warped, float) else str(warped)
        s = fmt.format(sphered) if isinstance(sphered, float) else str(sphered)
        print(f"   {claim:<46s} {cohort:<6s} {w:>9s} -> {s:>9s}")

    print("=== region size, the reason for the change ===")
    for cohort, sph in (("hcpa", "HCP/hcpa_alps_spheres_5mm.csv"),
                        ("dlbs", "DLBS/dlbs_alps_spheres_5mm.csv")):
        s = pd.read_csv(DIFF / sph)
        v = (pd.to_numeric(s.n_proj, errors="coerce")
             + pd.to_numeric(s.n_assoc, errors="coerce")).dropna()
        v = v[v > 0]
        n = load(cohort, warped=False)
        g = (n.n_proj_geom + n.n_assoc_geom).dropna()
        rec("region size, max/min", cohort,
            float(v.max() / max(v.min(), 1)), float(g.max() / max(g.min(), 1)),
            fmt="{:.2f}x")
        rec("region size, CV %", cohort,
            float(v.std() / v.mean() * 100), float(g.std() / g.mean() * 100),
            fmt="{:.1f}")

    print("\n=== age associations, one session per participant ===")
    for cohort in ("hcpa", "dlbs"):
        W, S = load(cohort, True), load(cohort, False)
        for col in ("classic", "cross", "v2_slab", "pv_perp", "anat_x"):
            a, b = one_per(W).dropna(subset=[col, "Age"]), one_per(S).dropna(
                subset=[col, "Age"])
            rec(f"age r, {col}", cohort,
                float(stats.pearsonr(a.Age, a[col])[0]),
                float(stats.pearsonr(b.Age, b[col])[0]))

    print("\n=== reliability, participants with repeat visits ===")
    for cohort in ("hcpa", "dlbs"):
        W, S = load(cohort, True), load(cohort, False)
        for col in ("classic", "cross", "v2_slab", "pv_perp", "anat_x"):
            rec(f"ICC, {col}", cohort,
                float(variance_components(repeats(W).dropna(subset=[col]), col)["icc"]),
                float(variance_components(repeats(S).dropna(subset=[col]), col)["icc"]))

    print("\n=== how close each variant sits to the bound ===")
    for cohort in ("hcpa", "dlbs"):
        W, S = load(cohort, True), load(cohort, False)
        for col in ("classic", "cross", "v2_slab"):
            rec(f"median {col} / pv_perp", cohort,
                float((W[col] / W.pv_perp).median()),
                float((S[col] / S.pv_perp).median()), fmt="{:.4f}")

    print("\n=== agreement with the ratio, correlation ===")
    for cohort in ("hcpa", "dlbs"):
        W, S = one_per(load(cohort, True)), one_per(load(cohort, False))
        for col in ("classic", "cross", "v2_slab", "anat_x"):
            a = W.dropna(subset=[col, "pv_perp"])
            b = S.dropna(subset=[col, "pv_perp"])
            rec(f"r with the ratio, {col}", cohort,
                float(stats.pearsonr(a[col], a.pv_perp)[0]),
                float(stats.pearsonr(b[col], b.pv_perp)[0]))

    print("\n=== LD-ALPS, whose own regions did not change ===")
    lo = HERE / "ld_alps_dlbs.csv"
    if lo.exists():
        L = pd.read_csv(lo)[["Subject_ID", "Visit", "ALPS_overall"]].rename(
            columns={"ALPS_overall": "ld_alps"})
        L["Subject_ID"] = L.Subject_ID.astype(str)
        L["Visit"] = L.Visit.astype(str)
        for warped, lab in ((True, "warped"), (False, "resphered")):
            d = one_per(load("dlbs", warped).merge(L, on=["Subject_ID", "Visit"]))
            d = d.dropna(subset=["ld_alps", "pv_perp"])
            viol = float((d.ld_alps > d.pv_perp + 1e-9).mean() * 100)
            corr = float(stats.pearsonr(d.ld_alps, d.pv_perp)[0])
            print(f"   {lab:<10s} n={len(d):3d}  violations {viol:5.2f}%  "
                  f"r with the ratio {corr:+.3f}")
        print("   LD-ALPS places its own regions, so the bound is being read")
        print("   across two different sets of voxels either way. The rise is")
        print("   a measure of that mismatch, not of the bound failing.")

    out = pd.DataFrame(rows)
    out.to_csv(HERE / "resphere_impact.csv", index=False)
    print(f"\n   wrote resphere_impact.csv ({len(out)} quantities)")


if __name__ == "__main__":
    main()
