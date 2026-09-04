"""Would any white-matter ROI do, or a global average?

alps_location_special.py asks whether the ALPS regions are unusual in their
perpendicular anisotropy. This asks the next question, which is the one a reader
reaches once the index has reduced to lambda2/lambda3 averaged over two chosen
ROIs: if the quantity exists everywhere, does the choice of where to measure it
matter at all?

Three arms, on the same sessions so nothing differs but the region.

  Each JHU label alone, ranked by its age association. If the ALPS regions sit
  mid-pack, their anatomical claim rests on placement rather than on signal.

  The ALPS pair, which is what the index computes.

  A global average across all twelve labels, weighted and unweighted. This is
  the "measure it anywhere" limit. It could go either way in advance. Averaging
  more tissue suppresses noise, which helps, but it also pools regions where
  lambda2 is close to lambda3 and the ratio carries little, which dilutes.

The planar coefficient CP = (lambda2 - lambda3)/lambda1 is reported alongside,
because it says how well-defined the ratio is in each label and therefore which
regions can carry signal at all.

    python alps_location_needed.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
SRC = HERE / "alps_location_special.csv"
ALPS = ["SCR (ALPS proj)", "SLF (ALPS assoc)"]


def main() -> None:
    d = pd.read_csv(SRC)
    labels = [c for c in d.columns
              if not c.endswith(" CP") and c not in ("sid", "Age")]
    d = d.dropna(subset=["Age"] + labels).copy()
    # one session per participant, matching the age convention elsewhere
    first = d.drop_duplicates("sid")

    print(f"{len(d)} sessions, {d.sid.nunique()} participants, "
          f"{len(first)} used for age, {len(labels)} labels\n")

    rows = []
    for lab in labels:
        r, p = stats.pearsonr(first[lab], first.Age)
        cp = float(d[f"{lab} CP"].median()) if f"{lab} CP" in d else np.nan
        rows.append(dict(region=lab, r_age=r, p_age=p, median_cp=cp,
                         median_ratio=float(d[lab].median()),
                         is_alps=lab in ALPS))

    # the index itself: the two ALPS regions averaged
    first = first.copy()
    first["ALPS pair"] = first[ALPS].mean(axis=1)
    # the measure-it-anywhere limit
    first["global (all 12)"] = first[labels].mean(axis=1)
    # and weighted by how well-defined the ratio is, since a label where
    # lambda2 is close to lambda3 contributes a nearly constant 1
    w = np.array([d[f"{lab} CP"].median() for lab in labels], float)
    first["global (CP-weighted)"] = (first[labels].values * w).sum(1) / w.sum()

    for name in ("ALPS pair", "global (all 12)", "global (CP-weighted)"):
        r, p = stats.pearsonr(first[name], first.Age)
        rows.append(dict(region=name, r_age=r, p_age=p, median_cp=np.nan,
                         median_ratio=float(first[name].median()),
                         is_alps=False))

    out = pd.DataFrame(rows).sort_values("r_age")
    print(f"  {'region':<26s} {'r age':>8s} {'p':>9s} {'CP':>7s} {'ratio':>7s}")
    for _, x in out.iterrows():
        mark = " <- ALPS" if x.is_alps else ""
        cp = f"{x.median_cp:.3f}" if x.median_cp == x.median_cp else "    --"
        print(f"  {x.region:<26s} {x.r_age:>+8.3f} {x.p_age:>9.1e} "
              f"{cp:>7s} {x.median_ratio:>7.3f}{mark}")

    singles = out[out.region.isin(labels)]
    ranks = {lab: int((singles.r_age <= singles.loc[singles.region == lab,
                                                   "r_age"].iloc[0]).sum())
             for lab in ALPS}
    print(f"\n  ALPS regions rank {ranks[ALPS[0]]} and {ranks[ALPS[1]]} "
          f"of {len(singles)} on the strength of the decline "
          f"(1 = strongest negative).")
    best = singles.iloc[0]
    pair = out[out.region == "ALPS pair"].iloc[0]
    glob = out[out.region == "global (all 12)"].iloc[0]
    print(f"  strongest single label: {best.region} at {best.r_age:+.3f}")
    print(f"  the ALPS pair: {pair.r_age:+.3f}")
    print(f"  a global average: {glob.r_age:+.3f}")

    out.to_csv(HERE / "alps_location_needed.csv", index=False)
    print(f"\n  wrote alps_location_needed.csv")


if __name__ == "__main__":
    main()
