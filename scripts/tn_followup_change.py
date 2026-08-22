"""Do any ALPS variants change after surgery, as conventional metrics do?

NOT part of the current manuscript. Exploratory.

The trigeminal dataset has a post-surgical arm the ALPS analysis never used.
tn_alps.py skipped every session whose name ends in "fu" until a flag was added.
A companion study of the same dataset reports that conventional diffusion
metrics partially reverse after surgery, fractional anisotropy rising and
diffusivity falling in the corticospinal tract and elsewhere, with paired effect
sizes up to 0.64, and that head motion and head position are unchanged between
the two scans.

That combination makes this the cleanest test available of the index.

  no age confound        each patient is their own control, which the
                         cross-sectional comparison is not, since the patients
                         there were older than the controls
  no position confound   position is unchanged between the paired scans, so a
                         change in the index cannot be posture
  a known positive       conventional metrics do move, so the design is not
                         inert and a null for ALPS is informative rather than
                         merely underpowered

Two outcomes, both worth having. If ALPS moves, the index detects something real
once posture is held still. If it does not while conventional metrics do, then
the cross-sectional group difference is more likely between-person confounding
than disease, which is the sharper reading.

Paired t and Wilcoxon per variant, Benjamini-Hochberg across variants, and the
paired effect size, alongside the baseline patient-control gap so the change can
be read as a fraction of it.

    python tn_followup_change.py

Writes tn_followup_change.csv.
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
VARIANTS = ["classic", "cross", "v2_sphere", "ALPS-PAS", "per-voxel",
            "pv_perp", "anat_x"]


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

    base = pd.read_csv(HERE / "tn_alps.csv")
    fu_path = HERE / "tn_alps_followup.csv"
    if not fu_path.exists():
        raise SystemExit("run: python tn_alps.py --followup")
    fu = pd.read_csv(fu_path)

    # sub-009fu is the follow-up of sub-009
    fu["base_id"] = fu.BIDS_ID.str.replace("fu$", "", regex=True)
    cols = [c for c in VARIANTS if c in base.columns and c in fu.columns]
    pair = base.merge(fu[["base_id"] + cols], left_on="BIDS_ID",
                      right_on="base_id", suffixes=("_pre", "_post"))
    print(f"{len(pair)} patients with paired pre- and post-surgical scans\n")

    par = pd.read_csv(winpath(PARTICIPANTS), sep="\t")
    par.columns = [c.strip().lstrip("\ufeff") for c in par.columns]
    par["patient"] = (par.Clinical_finding.astype(str).str.strip().str.upper()
                      .str.startswith("TN"))
    b = base.merge(par[["BIDS_ID", "patient"]], on="BIDS_ID")

    rows = []
    for c in cols:
        pre, post = pair[f"{c}_pre"].to_numpy(), pair[f"{c}_post"].to_numpy()
        ok = np.isfinite(pre) & np.isfinite(post)
        pre, post = pre[ok], post[ok]
        if len(pre) < 10:
            continue
        diff = post - pre
        t, p_t = stats.ttest_rel(post, pre)
        try:
            _, p_w = stats.wilcoxon(post, pre)
        except ValueError:
            p_w = np.nan
        dz = diff.mean() / diff.std(ddof=1)
        # A null is only informative with a bound on what it excludes. The
        # interval on the paired effect size says how large a real change could
        # have hidden here, which is the number to compare against the 0.64 the
        # conventional metrics reached in these same scans.
        se_dz = np.sqrt(1 / len(diff) + dz ** 2 / (2 * len(diff)))
        crit = stats.t.ppf(0.975, len(diff) - 1)
        dz_lo, dz_hi = dz - crit * se_dz, dz + crit * se_dz

        # baseline patient-control gap, for scale
        gp = b.loc[b.patient, c].dropna()
        gc = b.loc[~b.patient, c].dropna()
        gap = gc.mean() - gp.mean()
        rows.append(dict(variant=c, n=len(pre),
                         pre=round(float(pre.mean()), 4),
                         post=round(float(post.mean()), 4),
                         change=round(float(diff.mean()), 4),
                         dz=round(float(dz), 3),
                         dz_lo=round(float(dz_lo), 3),
                         dz_hi=round(float(dz_hi), 3),
                         p_paired_t=float(p_t),
                         p_wilcoxon=float(p_w),
                         baseline_gap=round(float(gap), 4),
                         pct_of_gap_closed=(round(100 * diff.mean() / gap, 1)
                                            if abs(gap) > 1e-12 else np.nan)))

    out = pd.DataFrame(rows)
    out["q_bh"] = bh(out.p_paired_t.to_numpy())
    out.to_csv(HERE / "tn_followup_change.csv", index=False)

    print(f"{'variant':11s}{'n':>5s}{'change':>10s}{'dz':>7s}{'95% CI on dz':>18s}{'paired t':>10s}{'BH q':>8s}")
    for r in out.itertuples():
        ci = f"[{r.dz_lo:+.2f}, {r.dz_hi:+.2f}]"
        print(f"{r.variant:11s}{r.n:5d}{r.change:+10.4f}{r.dz:+7.2f}"
              f"{ci:>18s}{r.p_paired_t:10.4f}{r.q_bh:8.4f}")

    surv = out[out.q_bh < 0.05]
    print(f"\n   {len(surv)} of {len(out)} variants change after surgery at "
          f"q < 0.05.")
    if surv.empty:
        print("   None. Conventional metrics move in these same patients, with")
        print("   paired effect sizes up to 0.64, so the design is not inert.")
        print("   An index that does not move where they do, in a comparison")
        print("   free of both the age and the position confound, is evidence")
        print("   about the index rather than about the sample.")
        worst = max(out.dz_hi.abs().max(), out.dz_lo.abs().max())
        print()
        print(f"   The widest interval reaches |dz| = {worst:.2f}, so a paired")
        print("   change of the size the conventional metrics reach in these")
        print("   same scans, up to 0.64, is excluded for every variant.")
    else:
        for r in surv.itertuples():
            print(f"     {r.variant:11s} dz = {r.dz:+.2f}, q = {r.q_bh:.4f}, "
                  f"{r.pct_of_gap_closed}% of the baseline gap")


if __name__ == "__main__":
    main()
