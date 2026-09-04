"""Does correcting the axes buy anything on phenotypes, as it does on age?

The age result is that corrected variants associate more strongly than classic
in the aligned cohort, and that the gain tracks how closely each variant
approaches lambda2/lambda3. Age is one endpoint, and the obvious question is
whether the same holds for the phenotypes the index is used to study.

The manuscript already reports that no variant clears FDR on any phenotype after
age and sex adjustment. That is a statement about absolute significance. It does
not answer the comparative question, which is whether corrected beats classic,
and a null endpoint can still order the variants consistently.

Two things are asked, on the same phenotypes and the same participants so that
nothing differs but the measurement axis.

  Per phenotype, does the corrected variant carry a larger association than
  classic? Reported as the fraction of phenotypes where it does, against the
  50% a coin would give, with a sign test.

  Does the gain, where it exists, track the ordering that age gives? If the
  ratio-approaching variants win on phenotypes as they do on age, the same
  mechanism is operating. If the ordering is absent, the age gain is specific
  to age and should not be generalised.

Phenotypes are screened to those with a usable sample, and the unadjusted arm is
used, since adjustment for age removes the variance the variants differ on.

    python phenotype_variant_gain.py
    python phenotype_variant_gain.py --arm age
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
VARIANTS = ["cross", "v2_sphere", "v2_slab", "pv_perp", "anat_x"]
# how closely each variant attains the ratio, from Section "How Closely Each
# Variant Approaches the Ratio". pv_perp is the ratio itself.
ATTAIN = {"classic": 0.81, "cross": 0.87, "anat_x": 0.90, "v2_sphere": 0.92,
          "v2_slab": 0.92, "pv_perp": 1.00}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="none",
                    help="'none' for unadjusted, or an arm in phenotype_arms")
    ap.add_argument("--min-n", type=int, default=60)
    args = ap.parse_args()

    if args.arm == "none":
        d = pd.read_csv(HERE / "phenotype_sweep.csv")
    else:
        a = pd.read_csv(HERE / "phenotype_arms_hcpa.csv")
        d = a[a.arm == args.arm].copy()
        if d.empty:
            raise SystemExit(f"no arm {args.arm!r}; have {sorted(a.arm.unique())}")
    d = d[d.n >= args.min_n].dropna(subset=["classic"] + VARIANTS)

    print(f"arm={args.arm}  {len(d)} phenotypes, n>={args.min_n}\n")
    print(f"  {'variant':<12s} {'attains':>8s} {'|r| beats classic':>18s} "
          f"{'sign p':>9s} {'median |r|':>11s} {'median gain':>12s}")
    print(f"  {'classic':<12s} {ATTAIN['classic']:>8.2f} {'--':>18s} "
          f"{'--':>9s} {d.classic.abs().median():>11.3f} {'--':>12s}")

    rows = []
    for v in VARIANTS:
        beats = (d[v].abs() > d.classic.abs())
        k, n = int(beats.sum()), len(d)
        # sign test against a coin
        p = stats.binomtest(k, n, 0.5).pvalue
        gain = (d[v].abs() - d.classic.abs()).median()
        print(f"  {v:<12s} {ATTAIN[v]:>8.2f} {k:>10d}/{n:<7d} "
              f"{p:>9.2g} {d[v].abs().median():>11.3f} {gain:>+12.4f}")
        rows.append(dict(arm=args.arm, variant=v, attains=ATTAIN[v],
                         n_phenotypes=n, n_beats_classic=k,
                         frac_beats=k / n, sign_p=p,
                         median_abs_r=float(d[v].abs().median()),
                         median_gain=float(gain)))

    out = pd.DataFrame(rows)
    # Does the gain follow the attainment ordering, as it does for age?
    rho, prho = stats.spearmanr(out.attains, out.median_gain)
    print(f"\n  gain against ratio attainment: Spearman {rho:+.3f} (p={prho:.3f})")
    if prho < 0.05 and rho > 0:
        print("  The gain rises with attainment, as it does for age, so the same")
        print("  ordering holds on these endpoints.")
    else:
        print("  No ordering by attainment here, so the age result should not")
        print("  be generalised to these endpoints.")

    # And the absolute question the manuscript already answers, restated so the
    # comparative and absolute readings sit together.
    qcols = [c for c in d.columns if c.startswith("q_")]
    if qcols:
        sig = {c[2:]: int((d[c] < 0.05).sum()) for c in qcols}
        print(f"\n  phenotypes clearing FDR q<0.05: "
              + ", ".join(f"{k} {v}" for k, v in sig.items()))

    p = HERE / f"phenotype_variant_gain_{args.arm}.csv"
    out.to_csv(p, index=False)
    print(f"\n  wrote {p.name}")


if __name__ == "__main__":
    main()
