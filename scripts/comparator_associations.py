"""Age associations for ALPS-PAS and the per-voxel variant in the aging cohorts.

These two rows were dashes in both the variant table and the beyond-ratio table
because the comparators had only ever been computed in the patient cohort.
aging_cohort_comparators.py supplies the per-session values; this turns them
into the numbers the tables report.

Two conventions are in play and they are not interchangeable, so both are
produced and each is labeled with the table it belongs to.

    tbl:variants   Pearson r with age over ALL sessions, 1706 and 379
    tbl:beyond     Pearson r with age over ONE session per participant, and
                   the same with the eigenvalue ratio partialled out

The classic index is computed the same way in the same pass. Its values are
already published in those tables, so reproducing them is the test that these
conventions have been implemented correctly. If classic does not come back at
the printed value, nothing else here should be used.

    python comparator_associations.py --cohort dlbs
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
VARIANTS = ["classic", "cross", "pv_perp", "anat_x", "ALPS-PAS", "per-voxel"]
# What the manuscript currently prints for classic, as the correctness check.
KNOWN = {"dlbs": {"all_sessions": -0.396, "one_per_participant": -0.328},
         "hcpa": {"all_sessions": -0.465, "one_per_participant": -0.430}}


def partial_out(y, x, z):
    """Correlation of y with x after removing z from both."""
    ok = ~(np.isnan(y) | np.isnan(x) | np.isnan(z))
    y, x, z = y[ok], x[ok], z[ok]
    Z = np.column_stack([np.ones(len(z)), z])

    def rz(v):
        b, *_ = np.linalg.lstsq(Z, v, rcond=None)
        return v - Z @ b
    return float(np.corrcoef(rz(y), rz(x))[0, 1]), int(ok.sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=["hcpa", "dlbs"], default="dlbs")
    args = ap.parse_args()

    c = pd.read_csv(HERE / f"comparators_{args.cohort}.csv")
    f = ("measured_pvs_axis_hcpa_b1500_all.csv" if args.cohort == "hcpa"
         else "measured_pvs_axis_dlbs.csv")
    base = pd.read_csv(HERE / f)
    for d in (c, base):
        d["Subject_ID"] = d.Subject_ID.astype(str)
        d["Visit"] = d.Visit.astype(str)
    # The manuscript's lambda2/lambda3 row is pv_perp itself, not a separately
    # pooled ratio. The two agree to r=0.9999 but not exactly, and partialling
    # the wrong one shifts the printed values by 0.002 to 0.003. Use pv_perp so
    # these numbers sit in the same column as the ones already published.
    base["ratio"] = base["pv_perp"]
    m = c.merge(base[["Subject_ID", "Visit", "Age", "ratio"]],
                on=["Subject_ID", "Visit"], how="inner")
    print(f"{args.cohort}: {len(m)} sessions, {m.Subject_ID.nunique()} participants\n")

    have = [v for v in VARIANTS if v in m.columns and m[v].notna().sum() > 20]
    first = m.sort_values(["Subject_ID", "Visit"]).groupby("Subject_ID").first().reset_index()

    print("=== tbl:variants convention, all sessions ===")
    rows = []
    for v in have:
        ok = m[v].notna() & m.Age.notna()
        r = float(np.corrcoef(m.loc[ok, v], m.loc[ok, "Age"])[0, 1])
        rows.append({"variant": v, "convention": "all_sessions", "n": int(ok.sum()),
                     "r_age": r, "r_age_given_ratio": np.nan})
        print(f"   {v:<12s} n={int(ok.sum()):5d}  r={r:+.3f}")

    print("\n=== tbl:beyond convention, one session per participant ===")
    for v in have:
        ok = first[v].notna() & first.Age.notna()
        r = float(np.corrcoef(first.loc[ok, v], first.loc[ok, "Age"])[0, 1])
        rp, n = partial_out(first[v].to_numpy(float), first.Age.to_numpy(float),
                            first.ratio.to_numpy(float))
        rows.append({"variant": v, "convention": "one_per_participant",
                     "n": int(ok.sum()), "r_age": r, "r_age_given_ratio": rp})
        print(f"   {v:<12s} n={int(ok.sum()):5d}  raw r={r:+.3f}   "
              f"| ratio {rp:+.3f}")

    # tbl:variants also carries ICC(1,1) across visits, on the same all-sessions
    # sample. Participants with a single session carry no within-participant
    # information and are dropped rather than counted as perfectly reliable.
    print("\n=== ICC(1,1) across visits, all sessions ===")
    for v in have:
        d = m[["Subject_ID", v]].dropna()
        g = d.groupby("Subject_ID")[v]
        d = d[d.Subject_ID.isin(g.size()[g.size() > 1].index)]
        if len(d) < 20:
            continue
        grp = d.groupby("Subject_ID")[v]
        k = grp.size().mean()
        grand = d[v].mean()
        msb = (grp.size() * (grp.mean() - grand) ** 2).sum() / (grp.ngroups - 1)
        msw = ((d[v] - grp.transform("mean")) ** 2).sum() / (len(d) - grp.ngroups)
        icc = (msb - msw) / (msb + (k - 1) * msw)
        rows.append({"variant": v, "convention": "icc_1_1", "n": len(d),
                     "r_age": np.nan, "r_age_given_ratio": np.nan, "icc": icc})
        print(f"   {v:<12s} n={len(d):5d} sessions, {grp.ngroups:4d} participants"
              f"  ICC={icc:.3f}")

    # The supplement argues every variant lies at or below lambda2/lambda3, and
    # that ALPS-PAS never exceeds it because per-voxel selection cannot push the
    # numerator above lambda2 or the denominator below lambda3. That is a claim
    # about the data, so it is measured rather than asserted.
    print("\n=== position relative to the eigenvalue ratio ===")
    for v in have:
        if v == "pv_perp":
            continue
        d = m[[v, "ratio"]].dropna()
        exceed = 100 * float((d[v] > d.ratio).mean())
        r = float(np.corrcoef(d[v], d.ratio)[0, 1])
        print(f"   {v:<12s} mean {d[v].mean():.3f} against ratio "
              f"{d.ratio.mean():.3f}, tracks at r={r:.3f}, "
              f"exceeds it in {exceed:.2f}% of sessions")

    out = pd.DataFrame(rows)
    out.to_csv(HERE / f"comparator_associations_{args.cohort}.csv", index=False)

    print("\n=== check: does classic reproduce the printed value? ===")
    good = True
    for conv, want in KNOWN[args.cohort].items():
        got = out[(out.variant == "classic") & (out.convention == conv)].r_age
        if not len(got):
            continue
        got = float(got.iloc[0])
        ok = abs(got - want) < 0.002
        good &= ok
        print(f"   {conv:<20s} printed {want:+.3f}, computed {got:+.3f}  "
              f"{'ok' if ok else 'MISMATCH'}")
    if not good:
        print("\n   Classic does not reproduce. Do not use the comparator values")
        print("   from this pass until the discrepancy is understood.")
    print(f"\n   wrote comparator_associations_{args.cohort}.csv")


if __name__ == "__main__":
    main()
