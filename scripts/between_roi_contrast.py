"""Does a contrast between regions carry more than a single region does?

The ALPS index uses two regions and averages them. Section on the two-region
design shows the averaging adds nothing in the precise cohort: the projection
region alone matches the pair. That leaves the other thing two regions could be
for, which is a contrast rather than a mean.

The motivation is the same one that makes ALPS a ratio in the first place. A
ratio cancels whatever multiplies both of its terms, so overall diffusivity
drops out and only the asymmetry survives. Applied between regions, a ratio
would cancel whatever scales both regions equally: global tissue aging, protocol
and scanner scaling, head size, whole-brain atrophy, session data quality. What
survives is what differs regionally, which is where a locally specific effect
such as a perivascular contribution would have to live.

Three tests, all on the same participants and the same twelve JHU labels.

  Every single region against age, as the baseline to beat.

  Every region against age after partialling out the average of all twelve.
  This is the cleanest form of the question: is there any regional signal at
  all, or is every label reporting the same global quantity? If the partials
  collapse, no contrast between regions can help, because there is nothing
  regional to contrast.

  Every pairwise log ratio against age. The log makes it symmetric and makes a
  ratio of ratios additive. A contrast that carries more than a region must
  beat both of its own constituents, not just beat the average one, so that is
  what is reported.

A caution the output repeats. Age is a globally shared endpoint, so differencing
two regions removes signal by construction and this test is close to the worst
case for a contrast. A null here bounds the idea for age and says little about
a regionally specific endpoint.

    python between_roi_contrast.py
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
SRC = HERE / "alps_location_special.csv"
ALPS = ["SCR (ALPS proj)", "SLF (ALPS assoc)"]


def r_age(x, age) -> tuple[float, float]:
    m = np.isfinite(x) & np.isfinite(age)
    r, p = stats.pearsonr(np.asarray(x)[m], np.asarray(age)[m])
    return float(r), float(p)


def partial_r(x, y, z) -> float:
    """r(x, y) with z removed from both."""
    X, Y, Z = (np.asarray(v, float) for v in (x, y, z))
    m = np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z)
    X, Y, Z = X[m], Y[m], Z[m]
    A = np.column_stack([np.ones(len(Z)), Z])
    rx = X - A @ np.linalg.lstsq(A, X, rcond=None)[0]
    ry = Y - A @ np.linalg.lstsq(A, Y, rcond=None)[0]
    if rx.std() <= 1e-8 * max(X.std(), 1e-30):
        return float("nan")
    return float(stats.pearsonr(rx, ry)[0])


def main() -> None:
    d = pd.read_csv(SRC)
    labels = [c for c in d.columns
              if not c.endswith(" CP") and c not in ("sid", "Age")]
    d = d.dropna(subset=["Age"] + labels).drop_duplicates("sid")
    age = d.Age.values
    glob = d[labels].mean(axis=1).values

    print(f"{len(d)} participants, {len(labels)} labels\n")

    print("  single regions, and what survives removing the 12-label average")
    print(f"  {'region':<20s} {'r age':>8s} {'r | global':>11s}")
    singles = {}
    for lab in labels:
        r, _ = r_age(d[lab].values, age)
        rp = partial_r(d[lab].values, age, glob)
        singles[lab] = r
        mark = "  <- ALPS" if lab in ALPS else ""
        print(f"  {lab:<20s} {r:>+8.3f} {rp:>+11.3f}{mark}")
    rg, pg = r_age(glob, age)
    print(f"  {'GLOBAL (all 12)':<20s} {rg:>+8.3f} {'--':>11s}")

    # every pairwise log ratio
    rows = []
    for a, b in itertools.combinations(labels, 2):
        c = np.log(d[a].values) - np.log(d[b].values)
        r, p = r_age(c, age)
        # a contrast earns its place only by beating both constituents
        best_single = max(abs(singles[a]), abs(singles[b]))
        rows.append(dict(a=a, b=b, r_age=r, p_age=p, abs_r=abs(r),
                         best_constituent=best_single,
                         gain=abs(r) - best_single))
    con = pd.DataFrame(rows).sort_values("abs_r", ascending=False)

    print(f"\n  {len(con)} pairwise log ratios, strongest five against age")
    print(f"  {'contrast':<42s} {'r age':>8s} {'best part':>10s} {'gain':>8s}")
    for _, x in con.head(5).iterrows():
        print(f"  {x.a[:19]:<19s} / {x.b[:19]:<19s} {x.r_age:>+8.3f} "
              f"{x.best_constituent:>10.3f} {x.gain:>+8.3f}")

    beat = con[con.gain > 0]
    print(f"\n  contrasts beating both of their own constituents: "
          f"{len(beat)} of {len(con)}")
    if len(beat):
        for _, x in beat.sort_values("gain", ascending=False).head(5).iterrows():
            print(f"     {x.a} / {x.b}: {x.r_age:+.3f} vs {x.best_constituent:.3f}")
    print(f"  best single region {max(abs(v) for v in singles.values()):.3f}, "
          f"global average {abs(rg):.3f}")

    con.to_csv(HERE / "between_roi_contrast.csv", index=False)
    print("\n  Age is globally shared, so a contrast removes signal by")
    print("  construction. This bounds the idea for age, not for an endpoint")
    print("  that is regionally specific.")
    print("  wrote between_roi_contrast.csv")


if __name__ == "__main__":
    main()
