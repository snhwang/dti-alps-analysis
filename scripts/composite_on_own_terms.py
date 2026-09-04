"""Rank the variants as composites, without partialling the ratio out of them.

Most of this project tests what a variant retains after the eigenvalue ratio is
removed. That is the right test of the glymphatic reading, because it asks
whether the index carries anything the ratio does not. It is the wrong test of
the index as an instrument. A composite is not obliged to beat its own largest
component, and an index that is mostly radial anisotropy can still be the best
available estimator of something.

So this asks the other question. Across the within-participant phenotype set,
with age removed and nothing else, which variant carries the strongest
association? Every variant is evaluated on the same participants, the same
visits and the same endpoints, so the only difference is the measurement axis.

Three summaries, because one number would hide the disagreement between them:

  Per variant, how many phenotypes it wins outright.
  Per variant, how many clear FDR at all.
  The head-to-head against classic and against the ratio, since those are the
  two references that matter: the conventional index, and the quantity the
  paper shows every variant approaches.

The within-participant design is used rather than the cross-sectional one
because it cancels every time-invariant confound, so a difference between
variants there is a difference in measurement rather than in case mix.

    python composite_on_own_terms.py
    python composite_on_own_terms.py --arm age+pose
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
# the variants that are indices, excluding the diagnostic angle columns
VARIANTS = ["classic", "cross", "v2_sphere", "v2_slab", "anat_x", "pv_perp"]
LABEL = {"classic": "Classic", "cross": "Refined (cross product)",
         "v2_sphere": "Measured axis (sphere)", "v2_slab": "Measured axis (slab)",
         "anat_x": "Anatomical axis", "pv_perp": "Eigenvalue ratio"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="age",
                    help="'age' is the composite on its own terms")
    args = ap.parse_args()

    d = pd.read_csv(HERE / "phenotype_longitudinal_hcpa.csv")
    d = d[(d.arm == args.arm) & (d.variant.isin(VARIANTS))]
    if d.empty:
        raise SystemExit(f"no rows for arm {args.arm!r}")

    wide = d.pivot_table(index="phenotype", columns="variant", values="r")
    qs = d.pivot_table(index="phenotype", columns="variant", values="q")
    wide = wide.dropna(subset=VARIANTS)
    qs = qs.loc[wide.index]

    print(f"arm={args.arm}, {len(wide)} phenotypes, within participants\n")

    print(f"  {'variant':<24s} {'wins':>6s} {'q<0.05':>8s} {'median |r|':>11s} "
          f"{'max |r|':>9s}")
    winner = wide.abs().idxmax(axis=1)
    for v in VARIANTS:
        print(f"  {LABEL[v]:<24s} {int((winner == v).sum()):>6d} "
              f"{int((qs[v] < 0.05).sum()):>8d} {wide[v].abs().median():>11.4f} "
              f"{wide[v].abs().max():>9.4f}")

    print("\n  head to head, phenotypes where the row beats the column on |r|")
    for ref in ("classic", "pv_perp"):
        print(f"\n   against {LABEL[ref]}:")
        for v in VARIANTS:
            if v == ref:
                continue
            beats = (wide[v].abs() > wide[ref].abs())
            k, n = int(beats.sum()), len(wide)
            p = stats.binomtest(k, n, 0.5).pvalue
            gain = (wide[v].abs() - wide[ref].abs()).median()
            flag = "  *" if p < 0.05 else "   "
            print(f"     {LABEL[v]:<24s} {k:>4d}/{n:<4d} p={p:<8.4f} "
                  f"median gain {gain:+.4f}{flag}")

    out = pd.DataFrame({
        "variant": VARIANTS,
        "label": [LABEL[v] for v in VARIANTS],
        "wins": [int((winner == v).sum()) for v in VARIANTS],
        "n_fdr": [int((qs[v] < 0.05).sum()) for v in VARIANTS],
        "median_abs_r": [float(wide[v].abs().median()) for v in VARIANTS],
        "beats_classic": [int((wide[v].abs() > wide["classic"].abs()).sum())
                          for v in VARIANTS],
        "beats_ratio": [int((wide[v].abs() > wide["pv_perp"].abs()).sum())
                        for v in VARIANTS],
        "n_phenotypes": len(wide),
        "arm": args.arm,
    })
    p = HERE / f"composite_on_own_terms_{args.arm.replace('+', '_')}.csv"
    out.to_csv(p, index=False)
    print(f"\n  wrote {p.name}")


if __name__ == "__main__":
    main()
