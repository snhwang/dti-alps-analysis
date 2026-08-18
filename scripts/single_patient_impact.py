"""
Does head positioning move a single patient across a clinical decision boundary?

Group studies and single-patient reads fail differently. In a cohort the
estimation variance the refined index adds costs statistical power, while a
positioning bias that happens to correlate with group membership manufactures
an effect. For one patient there are no repeated measurements, so the variance
penalty only widens the normative interval slightly, whereas a positioning bias
displaces that patient against the reference range, which is what is actually
being read. There is also no cohort against which an outlier would stand out,
so a silently failed registration on a single case goes unnoticed.

This measures the clinically interpretable quantity: how far a patient's
percentile against a normative distribution moves when their head is tilted,
and how often that crosses a decision threshold.

The normative distribution is built from unrotated HCP-A, which is AC-PC
aligned and therefore as close to a positioning-free reference as the data
allow. Each participant is then re-read with a tilt applied and their
percentile recomputed against the same normative distribution.

Usage:
    python single_patient_impact.py
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import atomic_io  # noqa: F401  writes become atomic on import

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rotation_dose_response import euler_rotation, load_sessions
from rotation_study import METHODS, evaluate

HERE = Path(__file__).resolve().parent
THRESHOLD_PCTILE = 10.0     # below this counts as an abnormal read


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--repeats", type=int, default=6)
    args = ap.parse_args()

    sessions = load_sessions()[: args.limit]
    print(f"normative sample: {len(sessions)} sessions "
          f"(unrotated, AC-PC aligned HCP-A)\n")

    norm = {m: np.array([evaluate(s, fn) for s in sessions])
            for m, fn in METHODS.items()}

    def pct(m, v):
        return 100.0 * (norm[m] < v).mean()

    base_pct = {m: np.array([pct(m, v) for v in norm[m]]) for m in METHODS}
    thr = {m: np.percentile(norm[m], THRESHOLD_PCTILE) for m in METHODS}

    rng = np.random.default_rng(20260728)
    print("Median |shift| in normative percentile after a pitch tilt,")
    print(f"and share of patients whose read crosses the {THRESHOLD_PCTILE:.0f}th "
          f"percentile boundary\n")
    print(f"{'tilt':>6s} " + " ".join(f"{m:>20s}" for m in METHODS))
    rows = []
    for tilt in (5, 10, 15, 20):
        shifts = {m: [] for m in METHODS}
        flips = {m: [] for m in METHODS}
        for _ in range(args.repeats):
            # sign of the tilt varies between patients, magnitude fixed
            for i, s in enumerate(sessions):
                sgn = 1 if rng.random() < 0.5 else -1
                R = euler_rotation(sgn * tilt, 0, 0)
                for m, fn in METHODS.items():
                    v = evaluate(s, fn, R)
                    shifts[m].append(abs(pct(m, v) - base_pct[m][i]))
                    was = norm[m][i] < thr[m]
                    now = v < thr[m]
                    flips[m].append(was != now)
        row = {"tilt_deg": tilt}
        cells = []
        for m in METHODS:
            md = float(np.median(shifts[m]))
            fl = 100 * float(np.mean(flips[m]))
            row[f"{m}_pctile_shift"] = md
            row[f"{m}_flip_pct"] = fl
            cells.append(f"{md:6.1f} pts / {fl:4.1f}%")
        rows.append(row)
        print(f"{tilt:>4}deg " + " ".join(f"{c:>20s}" for c in cells))

    pd.DataFrame(rows).to_csv(HERE / "single_patient_impact.csv", index=False)
    print("\nEach cell is median percentile shift / percent of reads that cross "
          f"the {THRESHOLD_PCTILE:.0f}th-percentile boundary.")
    print("Wrote single_patient_impact.csv")


if __name__ == "__main__":
    main()
