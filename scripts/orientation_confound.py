"""
Is Classic ALPS's stronger age association real signal, or an orientation confound?

With automated ROIs and n=506 sessions, Classic ALPS correlates with age more
strongly than the refined variants (Williams p ~ 0.001). Two readings are
possible. Either orientation correction discards genuine age-related signal, or
the scanner-to-anatomy deviation itself varies with age, in which case part of
Classic's association is head positioning and tract geometry rather than
diffusion.

The distinction is testable, because the deviation angles are measured:

  1. Do the deviation angles themselves correlate with age?
  2. Does the Classic minus Refined difference correlate with the deviation
     angles, and with age?
  3. Does controlling for deviation angle shrink the Classic age association
     toward the Refined one?
  4. Does the deviation angle add explanatory power to Classic ALPS beyond age?

Also reports the variance-component decomposition behind the ICC comparison,
since a lower ICC with a lower within-subject CV means the between-subject
variance fell, not that measurement got noisier.

Usage:
    python orientation_confound.py
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import atomic_io  # noqa: F401  writes become atomic on import
import statsmodels.formula.api as smf
from scipy import stats

from alps_common import parse_age

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
DIFF = HERE.parent.parent / "diffusion"

_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("--alps", default=str(DIFF / "DLBS" / "dlbs_alps_auto_cubic.csv"))
_ap.add_argument("--dev", default=str(DIFF / "DLBS" / "dlbs_alps_auto_axis_deviations.csv"))
_ap.add_argument("--motion", default=str(DIFF / "DLBS" / "dlbs_motion.csv"))
_ap.add_argument("--label", default="")
_ap.add_argument("--motion-threshold", type=float, default=0.5)
_ap.add_argument("--motion-exclude-pct", type=float, default=None,
                 help="exclude this percent of the cohort by motion instead of "
                      "using an absolute cutoff; use for HCP-A, where the DLBS "
                      "0.5 mm value would exclude 99 percent because RMS "
                      "accumulates over a 399-volume acquisition")
args = _ap.parse_args()

ALPS_CSV = Path(args.alps)
DEV_CSV = Path(args.dev)
MOTION_CSV = Path(args.motion)
SUFFIX = f"_{args.label}" if args.label else ""

MIN_ROI_VOXELS = 4

METRICS = {
    "Classic": ("Traditional_L", "Traditional_R", "Traditional_Avg"),
    "Refined": ("Refined_L", "Refined_R", "Refined_Avg"),
    "Refined+": ("RefinedPlus_L", "RefinedPlus_R", "RefinedPlus_Avg"),
    "ALPS-PAS": ("ALPS_PAS_L", "ALPS_PAS_R", "ALPS_PAS_Avg"),
}

lines: list[str] = []


def say(t: str = "") -> None:
    print(t)
    lines.append(t)


def header(t: str) -> None:
    say()
    say("=" * 78)
    say(t)
    say("=" * 78)


# ---------------------------------------------------------------------------

alps = pd.read_csv(ALPS_CSV)
dev = pd.read_csv(DEV_CSV)
motion = pd.read_csv(MOTION_CSV)

df = alps[alps["status"].astype(str) == "ok"].copy()
df = df.merge(
    dev.drop(columns=[c for c in ("Age", "Subject_ID", "Session", "Visit") if c in dev]),
    on="DTI_Session_ID",
    how="left",
)
# DLBS motion keys on session id, HCP-A motion keys on participant and visit.
if "DTI_Session_ID" in motion.columns:
    df = df.merge(motion[["DTI_Session_ID", "Eddy_Mean_RMS"]],
                  on="DTI_Session_ID", how="left")
else:
    df = df.drop(columns=[c for c in ("Eddy_Mean_RMS",) if c in df.columns])
    df = df.merge(motion[["Subject_ID", "Visit", "Eddy_Mean_RMS"]],
                  on=["Subject_ID", "Visit"], how="left")

for cols in METRICS.values():
    for c in cols[:2]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
for name, (l, r, a) in METRICS.items():
    df[a] = df[[l, r]].mean(axis=1)
df["Age"] = parse_age(df["Age"])

df = df.dropna(subset=["Age"] + [c for cols in METRICS.values() for c in cols[:2]])
_rms = pd.to_numeric(df["Eddy_Mean_RMS"], errors="coerce")
if args.motion_exclude_pct is not None:
    _thr = float(np.nanpercentile(_rms.dropna(), 100.0 - args.motion_exclude_pct))
else:
    _thr = args.motion_threshold
df = df[_rms <= _thr]
for c in ("n_proj", "n_assoc"):
    df = df[pd.to_numeric(df[c], errors="coerce") >= MIN_ROI_VOXELS]

for ang in ("theta_PVS", "theta_SCR", "theta_SLF"):
    df[f"{ang}_avg"] = df[[f"{ang}_L", f"{ang}_R"]].mean(axis=1)
df = df.dropna(subset=["theta_PVS_avg", "theta_SCR_avg", "theta_SLF_avg"])

df["diff_CR"] = df["Traditional_Avg"] - df["Refined_Avg"]
df["absdiff_CR"] = df["diff_CR"].abs()

header("COHORT")
say(f"Sessions after QC and with deviation angles: {len(df)}")
say(f"Subjects: {df.Subject_ID.nunique()}")

# ---------------------------------------------------------------------------
header("1. DO THE DEVIATION ANGLES THEMSELVES VARY WITH AGE?")
say()
say("If the scanner-to-anatomy angle tracks age, then part of any age")
say("association measured along fixed scanner axes is geometry, not diffusion.")
say()
say(f"{'Angle':<16s} {'median deg':>11s} {'r with age':>11s} {'p':>11s} {'slope deg/yr':>13s}")
for ang in ("theta_PVS_avg", "theta_SCR_avg", "theta_SLF_avg"):
    sl, ic, r, p, se = stats.linregress(df["Age"], df[ang])
    say(f"{ang:<16s} {df[ang].median():11.2f} {r:11.3f} {p:11.3e} {sl:13.5f}")

# ---------------------------------------------------------------------------
header("2. DOES THE CLASSIC MINUS REFINED DIFFERENCE TRACK ORIENTATION AND AGE?")
say()
say(f"{'Predictor':<16s} {'r with |Classic-Refined|':>26s} {'p':>11s}")
for ang in ("theta_PVS_avg", "theta_SCR_avg", "theta_SLF_avg", "Age"):
    sl, ic, r, p, se = stats.linregress(df[ang], df["absdiff_CR"])
    say(f"{ang:<16s} {r:26.3f} {p:11.3e}")
say()
sl, ic, r, p, se = stats.linregress(df["Age"], df["diff_CR"])
say(f"Signed Classic minus Refined vs age: r={r:.3f}, p={p:.3e}, slope={sl:.6f}/yr")
say("A non-zero signed trend means the two metrics diverge systematically with")
say("age, which is what an age-dependent orientation bias would produce.")

# ---------------------------------------------------------------------------
header("3. DOES CONTROLLING FOR ORIENTATION SHRINK THE CLASSIC AGE ASSOCIATION?")
say()
say("Standardised age coefficient for each metric, before and after adding the")
say("three deviation angles as covariates. Subject-clustered robust SEs.")
say()
say(f"{'Metric':<10s} {'beta_age raw':>13s} {'beta_age adj':>13s} {'change %':>10s} "
    f"{'p adj':>11s}")
rows = []
for name, (_, _, acol) in METRICS.items():
    d = df[["Subject_ID", "Age", acol, "theta_PVS_avg", "theta_SCR_avg",
            "theta_SLF_avg"]].dropna().rename(columns={acol: "y"}).copy()
    for c in ("y", "Age", "theta_PVS_avg", "theta_SCR_avg", "theta_SLF_avg"):
        d[c + "_z"] = (d[c] - d[c].mean()) / d[c].std(ddof=1)
    raw = smf.ols("y_z ~ Age_z", d).fit(
        cov_type="cluster", cov_kwds={"groups": d["Subject_ID"]})
    adj = smf.ols(
        "y_z ~ Age_z + theta_PVS_avg_z + theta_SCR_avg_z + theta_SLF_avg_z", d
    ).fit(cov_type="cluster", cov_kwds={"groups": d["Subject_ID"]})
    b0, b1 = float(raw.params["Age_z"]), float(adj.params["Age_z"])
    say(f"{name:<10s} {b0:13.4f} {b1:13.4f} {100*(b1-b0)/abs(b0):10.1f} "
        f"{float(adj.pvalues['Age_z']):11.3e}")
    rows.append({"metric": name, "beta_age_raw": b0, "beta_age_adj": b1,
                 "pct_change": 100 * (b1 - b0) / abs(b0)})

# ---------------------------------------------------------------------------
header("4. DOES ORIENTATION EXPLAIN CLASSIC ALPS BEYOND AGE?")
say()
say("Incremental R2 from adding the deviation angles to an age-only model.")
say()
say(f"{'Metric':<10s} {'R2 age':>9s} {'R2 age+angles':>15s} {'delta R2':>10s} {'F p':>11s}")
for name, (_, _, acol) in METRICS.items():
    d = df[["Subject_ID", "Age", acol, "theta_PVS_avg", "theta_SCR_avg",
            "theta_SLF_avg"]].dropna().rename(columns={acol: "y"})
    m0 = smf.ols("y ~ Age", d).fit()
    m1 = smf.ols("y ~ Age + theta_PVS_avg + theta_SCR_avg + theta_SLF_avg", d).fit()
    lr = m1.compare_f_test(m0)
    say(f"{name:<10s} {m0.rsquared:9.4f} {m1.rsquared:15.4f} "
        f"{m1.rsquared - m0.rsquared:10.4f} {lr[1]:11.3e}")

# ---------------------------------------------------------------------------
header("5. VARIANCE COMPONENTS BEHIND THE ICC COMPARISON")
say()
say("ICC is between-subject variance over total. A lower ICC with an equal or")
say("lower within-subject CV means between-subject spread shrank, not that")
say("measurement got noisier.")

counts = df["Subject_ID"].value_counts()
lon = df[df["Subject_ID"].isin(counts[counts >= 2].index)]
say()
say(f"Longitudinal subset: {lon.Subject_ID.nunique()} subjects, {len(lon)} sessions")
say()
say(f"{'Metric':<10s} {'var_between':>12s} {'var_within':>11s} {'ICC':>7s} "
    f"{'sd_within':>10s} {'wCV %':>7s} {'sd_between':>11s}")
for name, (_, _, acol) in METRICS.items():
    d = lon[["Subject_ID", acol]].dropna().rename(columns={acol: "y"})
    k = d["Subject_ID"].nunique()
    n_i = d.groupby("Subject_ID")["y"].size().to_numpy(dtype=float)
    m_i = d.groupby("Subject_ID")["y"].mean().to_numpy(dtype=float)
    N = float(n_i.sum())
    ybar = float((n_i * m_i).sum() / N)
    ms_b = float((n_i * (m_i - ybar) ** 2).sum()) / (k - 1)
    ms_w = float(d.groupby("Subject_ID")["y"].transform(lambda s: s - s.mean())
                 .pow(2).sum()) / (N - k)
    n0 = (N - (n_i**2).sum() / N) / (k - 1)
    vb = max((ms_b - ms_w) / n0, 0.0)
    grand = float(d["y"].mean())
    say(f"{name:<10s} {vb:12.6f} {ms_w:11.6f} {vb/(vb+ms_w):7.3f} "
        f"{np.sqrt(ms_w):10.5f} {100*np.sqrt(ms_w)/grand:7.2f} {np.sqrt(vb):11.5f}")

pd.DataFrame(rows).to_csv(HERE / f"orientation_confound{SUFFIX}.csv", index=False)
(HERE / f"report_orientation_confound{SUFFIX}.txt").write_text("\n".join(lines), encoding="utf-8")
print(f"\nWrote {HERE / 'report_orientation_confound.txt'}")
