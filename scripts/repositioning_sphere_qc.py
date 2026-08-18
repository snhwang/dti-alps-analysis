"""Repositioning sensitivity on the sample the manuscript reports.

Section 3.5 and the Figure 7 caption give the change in each index per degree of
change in the scanner-to-anatomy angle, over 580 visit-pair by hemisphere
observations from 156 participants. That sample could not be reproduced from any
committed output, because it needs three things together and no single script
did all three:

  the sphere cohort            longitudinal_reliability.py --cohort spheres
  the DLBS analysis sample     the 156 participants used elsewhere in the paper
  motion QC at 0.5 mm          from dlbs_motion.csv, which the sphere table
                               does not carry inline, so the script's own QC
                               step silently applied nothing

longitudinal_reliability.py hardcoded the hand-drawn cohort, so the committed
repositioning_table.csv held 78 observations from 19 subjects under a name that
looked like the figure's source. This applies all three conditions and writes
the slopes the manuscript quotes.

Run longitudinal_reliability.py --cohort spheres first; this reads its pairs.

    python repositioning_sphere_qc.py

Writes repositioning_sphere_qc.csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import atomic_io  # noqa: F401  writes become atomic on import

HERE = Path(__file__).resolve().parent
DIFF = HERE.parent.parent / "diffusion"
MOTION_MM = 0.5
WAVE = {"ses-wave1": 1, "ses-wave2": 2, "ses-wave3": 3}
METRICS = ("Classic", "Refined", "Refined+", "ALPS-PAS")
BOOT = 2000


def main() -> None:
    argparse.ArgumentParser().parse_args()
    src = HERE / "repositioning_pairs_spheres.csv"
    if not src.exists():
        raise SystemExit("run: python longitudinal_reliability.py --cohort spheres")

    pairs = pd.read_csv(src)
    pairs["Subject_ID"] = pairs.Subject_ID.astype(str)
    core = set(pd.read_csv(HERE / "measured_pvs_axis_dlbs.csv").Subject_ID.astype(str))
    mot = pd.read_csv(DIFF / "DLBS" / "dlbs_motion.csv")
    mot["Subject_ID"] = mot.Subject_ID.astype(str)
    mot["wave"] = mot.Session.map(WAVE)
    good = {(r.Subject_ID, r.wave) for r in mot.itertuples()
            if r.Eddy_Mean_RMS <= MOTION_MM}

    d = pairs[pairs.Subject_ID.isin(core)]
    d = d[[(a, wa) in good and (a, wb) in good
           for a, wa, wb in zip(d.Subject_ID, d.wave_a, d.wave_b)]]
    print(f"{len(d)} visit-pair by hemisphere observations from "
          f"{d.Subject_ID.nunique()} participants\n")

    rng = np.random.default_rng(0)
    subs = d.Subject_ID.unique()
    rows = []
    for m in METRICS:
        col = f"rel_{m}"
        if col not in d.columns:
            continue
        s = d[["Subject_ID", "d_theta_SCR", col]].dropna()
        slope = float(np.polyfit(s.d_theta_SCR, s[col], 1)[0])
        # subject-clustered bootstrap, as the reliability analyses use
        boot = []
        for _ in range(BOOT):
            pick = rng.choice(subs, len(subs), replace=True)
            r = pd.concat([s[s.Subject_ID == p] for p in pick])
            if len(r) > 2:
                boot.append(float(np.polyfit(r.d_theta_SCR, r[col], 1)[0]))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append(dict(metric=m, slope_pct_per_deg=round(slope, 4),
                         ci_lo=round(float(lo), 4), ci_hi=round(float(hi), 4),
                         n_obs=len(s), n_subjects=int(s.Subject_ID.nunique())))
        print(f"   {m:10s} {slope:+.3f}% per degree   "
              f"CI [{lo:+.2f}, {hi:+.2f}]")

    pd.DataFrame(rows).to_csv(HERE / "repositioning_sphere_qc.csv", index=False)


if __name__ == "__main__":
    main()
