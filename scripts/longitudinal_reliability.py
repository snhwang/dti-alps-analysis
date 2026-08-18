"""
Revision analyses for MAGRESIMAGING-D-26-00371 (Reviewers 1 and 4).

Addresses the shared core criticism that rotation invariance is not the same as
measurable benefit. Uses the DLBS multi-wave structure that Reviewer 4 pointed
to, from the existing per-session result tables. No reprocessing.

Analyses
  1. Between-wave reliability. Variance components from a random-intercept
     mixed model, giving ICC and within-subject CV for each ALPS variant, with
     and without adjustment for age. Subject-level bootstrap for CIs and for
     paired differences against Classic.
  2. Repositioning sensitivity. Within subject, anatomy is fixed, so the
     between-wave change in the scanner-to-anatomy angles is repositioning.
     Regress relative |change in ALPS| on |change in angle| across wave pairs,
     pooled over hemispheres, with subject-clustered bootstrap.
  3. Left and right analysed separately against age, instead of averaged.
  4. Williams/Steiger tests for the difference between the dependent
     age correlations of Classic and each corrected variant.
  5. Non-independence check on the published age correlation, which pools 62
     sessions drawn from 33 subjects.

Inputs
  ../../diffusion/HCP/lifespan_alps_results.csv
  ../../diffusion/HCP/alps_axis_deviations.csv

Outputs (written next to this script)
  longitudinal_reliability_report.txt
  reliability_table.csv
  repositioning_table.csv
  age_lr_table.csv
"""

from __future__ import annotations

import argparse
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

import atomic_io  # noqa: F401  writes become atomic on import
import statsmodels.formula.api as smf
from scipy import stats

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
DIFF = HERE.parent.parent / "diffusion"
HCP = DIFF / "HCP"
_COHORTS = {
    "manual": (HCP / "lifespan_alps_results.csv",
               HCP / "alps_axis_deviations.csv"),
    "auto": (DIFF / "DLBS" / "dlbs_alps_auto_cubic.csv",
             DIFF / "DLBS" / "dlbs_alps_auto_axis_deviations.csv"),
    "spheres": (DIFF / "DLBS" / "dlbs_alps_spheres_5mm.csv",
                DIFF / "DLBS" / "dlbs_alps_spheres_axis_deviations.csv"),
}
_ap = argparse.ArgumentParser()
_ap.add_argument("--cohort", choices=sorted(_COHORTS), default="manual",
                 help="which region set to analyse; outputs are suffixed by it")
_args, _ = _ap.parse_known_args()
COHORT = _args.cohort
SUF = "" if COHORT == "manual" else f"_{COHORT}"
ALPS_CSV, DEV_CSV = _COHORTS[COHORT]

FA_DROP_THRESHOLD = 20.0
MOTION_THRESHOLD = 0.5
N_BOOT = 2000
RNG = np.random.default_rng(20260726)

# label -> (left column, right column, average column)
METRICS = {
    "Classic": ("Traditional_L", "Traditional_R", "Traditional_Avg"),
    "Refined": ("Refined_L", "Refined_R", "Refined_Avg"),
    "Refined+": ("RefinedPlus_L", "RefinedPlus_R", "RefinedPlus_Avg"),
    "ALPS-PAS": ("ALPS_PAS_L", "ALPS_PAS_R", "ALPS_PAS_Avg"),
}
REFERENCE = "Classic"

lines: list[str] = []


def say(text: str = "") -> None:
    print(text)
    lines.append(text)


def header(text: str) -> None:
    say()
    say("=" * 78)
    say(text)
    say("=" * 78)


# ---------------------------------------------------------------------------
# Load and apply the same QC as the submitted figures
# ---------------------------------------------------------------------------

alps = pd.read_csv(ALPS_CSV)
dev = pd.read_csv(DEV_CSV)
merged = alps.merge(
    dev.drop(columns=["Age"]), on=["Subject_ID", "Session", "DTI_Session_ID"], how="left"
)

need_lr = [c for cols in METRICS.values() for c in cols[:2]]
cc = merged.dropna(subset=["Age"] + need_lr).copy()
n_complete = len(cc)
_qc = {"Max_Pct_Dropped": FA_DROP_THRESHOLD, "Eddy_Mean_RMS": MOTION_THRESHOLD}
_applied = [c for c in _qc if c in cc.columns]
for _c in _applied:
    cc = cc[cc[_c] <= _qc[_c]]
cc = cc.copy()
QC_APPLIED = _applied
print(f"QC filters applied: {_applied or 'none, columns absent from this cohort'}")

wave_order = {"ses-wave1": 1, "ses-wave2": 2, "ses-wave3": 3}
cc["wave"] = cc["Session"].map(wave_order)
cc = cc.sort_values(["Subject_ID", "wave"]).reset_index(drop=True)

counts = cc["Subject_ID"].value_counts()
repeat_ids = sorted(counts[counts >= 2].index)
lon = cc[cc["Subject_ID"].isin(repeat_ids)].copy()

header("COHORT")
say(f"Complete cases before QC          : {n_complete} sessions")
say(f"After motion and FA QC            : {len(cc)} sessions, {cc.Subject_ID.nunique()} subjects")
say(f"Subjects with >= 2 post-QC waves  : {len(repeat_ids)}")
say(f"Subjects with 3 post-QC waves     : {int((counts >= 3).sum())}")
say(f"Sessions in the longitudinal set  : {len(lon)}")
gaps = (
    lon.groupby("Subject_ID")["Age"].agg(lambda s: s.max() - s.min()).astype(float)
)
say(
    f"Within-subject age span (years)   : median {gaps.median():.1f}, "
    f"range {gaps.min():.1f} to {gaps.max():.1f}"
)
say()
say("The between-wave interval is years, so within-subject variance contains real")
say("biological change as well as measurement error. Every ALPS variant is exposed")
say("to the identical biological change in the identical sessions, so the ranking")
say("across variants is a fair comparison even though each absolute ICC is a lower")
say("bound on true short-interval test-retest reliability. Age-adjusted models below")
say("remove the linear part of that biological change.")


# ---------------------------------------------------------------------------
# 1. Variance components, ICC, within-subject CV
# ---------------------------------------------------------------------------


def variance_components(df: pd.DataFrame, col: str, adjust_age: bool) -> dict | None:
    """
    Variance components for one metric by the unbalanced one-way random-effects
    ANOVA estimator, giving ICC(1,1) and the within-subject CV.

    Closed form, so it always returns and is cheap enough to bootstrap. When
    adjust_age is set, the linear age trend is removed across all sessions
    first, so the remaining between-subject variance is not inflated by the
    lifespan age range and the within-subject variance excludes the linear part
    of real aging change over the wave interval.
    """
    d = df[["Subject_ID", "Age", col]].dropna().rename(columns={col: "y"}).copy()
    k = d["Subject_ID"].nunique()
    if k < 3:
        return None

    grand = float(d["y"].mean())
    if adjust_age:
        sl, ic, *_ = stats.linregress(d["Age"], d["y"])
        d["y"] = d["y"] - (sl * d["Age"] + ic) + grand

    n_i = d.groupby("Subject_ID")["y"].size().to_numpy(dtype=float)
    m_i = d.groupby("Subject_ID")["y"].mean().to_numpy(dtype=float)
    N = float(n_i.sum())
    ybar = float((n_i * m_i).sum() / N)

    ss_b = float((n_i * (m_i - ybar) ** 2).sum())
    ss_w = float(
        d.groupby("Subject_ID")["y"].transform(lambda s: s - s.mean()).pow(2).sum()
    )
    if N - k <= 0:
        return None
    ms_b = ss_b / (k - 1)
    ms_w = ss_w / (N - k)
    n0 = (N - (n_i**2).sum() / N) / (k - 1)

    var_b = max((ms_b - ms_w) / n0, 0.0)
    var_w = ms_w
    if not np.isfinite(var_b) or not np.isfinite(var_w) or (var_b + var_w) <= 0:
        return None
    return {
        "var_between": var_b,
        "var_within": var_w,
        "icc": var_b / (var_b + var_w),
        "sd_within": np.sqrt(var_w),
        "wcv_pct": 100.0 * np.sqrt(var_w) / grand,
    }


def bootstrap_subjects(df: pd.DataFrame, ids: list[str]) -> pd.DataFrame:
    """Resample subjects with replacement, relabelling so duplicates stay distinct."""
    picks = RNG.choice(len(ids), size=len(ids), replace=True)
    out = []
    for k, idx in enumerate(picks):
        block = df[df["Subject_ID"] == ids[idx]].copy()
        block["Subject_ID"] = f"boot{k:03d}"
        out.append(block)
    return pd.concat(out, ignore_index=True)


header("1. BETWEEN-WAVE RELIABILITY")

rel_rows = []
for adjust_age in (False, True):
    label = "age-adjusted" if adjust_age else "unadjusted"

    point = {}
    for name, (lcol, rcol, acol) in METRICS.items():
        vc = variance_components(lon, acol, adjust_age)
        point[name] = vc

    boot_icc = {name: [] for name in METRICS}
    boot_wcv = {name: [] for name in METRICS}
    for _ in range(N_BOOT):
        bs = bootstrap_subjects(lon, repeat_ids)
        bs_ids = sorted(bs["Subject_ID"].unique())
        for name, (_, _, acol) in METRICS.items():
            vc = variance_components(bs, acol, adjust_age)
            boot_icc[name].append(np.nan if vc is None else vc["icc"])
            boot_wcv[name].append(np.nan if vc is None else vc["wcv_pct"])
    boot_icc = {k: np.asarray(v, dtype=float) for k, v in boot_icc.items()}
    boot_wcv = {k: np.asarray(v, dtype=float) for k, v in boot_wcv.items()}

    say()
    say(f"--- L/R average, {label} (n={lon.Subject_ID.nunique()} subjects, "
        f"{len(lon)} sessions) ---")
    say(f"{'Metric':<10s} {'ICC':>6s} {'95% CI':>16s} {'wCV %':>7s} {'95% CI':>15s} "
        f"{'dICC vs Classic':>18s} {'p':>7s}")
    for name in METRICS:
        vc = point[name]
        if vc is None:
            say(f"{name:<10s}   model did not converge")
            continue
        icc_lo, icc_hi = np.nanpercentile(boot_icc[name], [2.5, 97.5])
        cv_lo, cv_hi = np.nanpercentile(boot_wcv[name], [2.5, 97.5])
        if name == REFERENCE:
            diff_txt, p_txt = "reference", ""
        else:
            d = boot_icc[name] - boot_icc[REFERENCE]
            d = d[np.isfinite(d)]
            d_point = vc["icc"] - point[REFERENCE]["icc"]
            d_lo, d_hi = np.nanpercentile(d, [2.5, 97.5])
            # two-sided bootstrap p for the paired difference
            p = 2.0 * min((d <= 0).mean(), (d >= 0).mean())
            p = min(1.0, max(p, 1.0 / len(d)))
            diff_txt = f"{d_point:+.3f} [{d_lo:+.3f},{d_hi:+.3f}]"
            p_txt = f"{p:.3f}"
        say(
            f"{name:<10s} {vc['icc']:6.3f} [{icc_lo:5.3f},{icc_hi:5.3f}] "
            f"{vc['wcv_pct']:7.2f} [{cv_lo:4.2f},{cv_hi:4.2f}] {diff_txt:>18s} {p_txt:>7s}"
        )
        rel_rows.append(
            {
                "model": label,
                "side": "avg",
                "metric": name,
                "n_subjects": lon["Subject_ID"].nunique(),
                "n_sessions": len(lon),
                "icc": vc["icc"],
                "icc_lo": icc_lo,
                "icc_hi": icc_hi,
                "wcv_pct": vc["wcv_pct"],
                "wcv_lo": cv_lo,
                "wcv_hi": cv_hi,
                "sd_within": vc["sd_within"],
                "var_between": vc["var_between"],
                "var_within": vc["var_within"],
            }
        )

# Per-hemisphere reliability, unadjusted and age-adjusted, no bootstrap
say()
say("--- Per hemisphere (point estimates) ---")
say(f"{'Metric':<10s} {'side':>5s} {'ICC unadj':>10s} {'ICC age-adj':>12s} "
    f"{'wCV% unadj':>11s} {'wCV% age-adj':>13s}")
for name, (lcol, rcol, _) in METRICS.items():
    for side, col in (("L", lcol), ("R", rcol)):
        a = variance_components(lon, col, False)
        b = variance_components(lon, col, True)
        if a is None or b is None:
            continue
        say(f"{name:<10s} {side:>5s} {a['icc']:10.3f} {b['icc']:12.3f} "
            f"{a['wcv_pct']:11.2f} {b['wcv_pct']:13.2f}")
        rel_rows.append(
            {
                "model": "per-hemisphere",
                "side": side,
                "metric": name,
                "n_subjects": lon["Subject_ID"].nunique(),
                "n_sessions": len(lon),
                "icc": a["icc"],
                "icc_lo": np.nan,
                "icc_hi": np.nan,
                "wcv_pct": a["wcv_pct"],
                "wcv_lo": np.nan,
                "wcv_hi": np.nan,
                "sd_within": a["sd_within"],
                "var_between": a["var_between"],
                "var_within": a["var_within"],
                "icc_age_adj": b["icc"],
                "wcv_pct_age_adj": b["wcv_pct"],
            }
        )

pd.DataFrame(rel_rows).to_csv(HERE / f"reliability_table{SUF}.csv", index=False)


# ---------------------------------------------------------------------------
# 2. Repositioning sensitivity across wave pairs
# ---------------------------------------------------------------------------

header("2. REPOSITIONING SENSITIVITY ACROSS WAVE PAIRS")
say()
say("Within a subject the underlying tract anatomy is fixed, so the between-wave")
say("change in the angle between a tract axis and its assumed scanner axis is head")
say("repositioning at a real re-acquisition, not a numerical rotation. This is the")
say("acquisition-stage evidence Reviewer 4 asks for in point 3.")
say()
say("|dtheta| understates repositioning, since two different head orientations can")
say("share a tilt magnitude relative to a scanner axis. The proxy is therefore")
say("conservative and biases every slope toward zero.")

pair_rows = []
for sid, block in lon.groupby("Subject_ID"):
    block = block.sort_values("wave")
    for (_, a), (_, b) in combinations(block.iterrows(), 2):
        for side in ("L", "R"):
            row = {
                "Subject_ID": sid,
                "wave_a": a["wave"],
                "wave_b": b["wave"],
                "side": side,
                "d_age": abs(float(b["Age"]) - float(a["Age"])),
                "d_motion": abs(float(b["Eddy_Mean_RMS"]) - float(a["Eddy_Mean_RMS"])),
            }
            ok = True
            for ang in ("theta_PVS", "theta_SCR", "theta_SLF"):
                col = f"{ang}_{side}"
                if col not in a or pd.isna(a[col]) or pd.isna(b[col]):
                    ok = False
                    break
                row[f"d_{ang}"] = abs(float(b[col]) - float(a[col]))
            if not ok:
                continue
            for name, (lcol, rcol, _) in METRICS.items():
                col = lcol if side == "L" else rcol
                va, vb = float(a[col]), float(b[col])
                row[f"rel_{name}"] = 100.0 * abs(vb - va) / ((va + vb) / 2.0)
            pair_rows.append(row)

pairs = pd.DataFrame(pair_rows)
pair_ids = sorted(pairs["Subject_ID"].unique())
say()
say(f"Wave pairs x hemispheres: {len(pairs)} observations from {len(pair_ids)} subjects")
say(f"|dtheta_PVS| median {pairs.d_theta_PVS.median():.2f} deg, "
    f"IQR [{pairs.d_theta_PVS.quantile(.25):.2f}, {pairs.d_theta_PVS.quantile(.75):.2f}], "
    f"max {pairs.d_theta_PVS.max():.2f}")
say(f"|dtheta_SCR| median {pairs.d_theta_SCR.median():.2f} deg, "
    f"IQR [{pairs.d_theta_SCR.quantile(.25):.2f}, {pairs.d_theta_SCR.quantile(.75):.2f}], "
    f"max {pairs.d_theta_SCR.max():.2f}")


def cluster_boot_slopes(df: pd.DataFrame, xcol: str) -> dict:
    """OLS slope of relative |change| on |dtheta| per metric, subject-clustered CI."""
    ids = sorted(df["Subject_ID"].unique())
    point = {}
    for name in METRICS:
        sl, ic, r, p, se = stats.linregress(df[xcol], df[f"rel_{name}"])
        point[name] = {"slope": sl, "r": r, "p": p, "intercept": ic}

    boots = {name: [] for name in METRICS}
    for _ in range(N_BOOT):
        picks = RNG.choice(len(ids), size=len(ids), replace=True)
        bs = pd.concat([df[df["Subject_ID"] == ids[i]] for i in picks], ignore_index=True)
        for name in METRICS:
            sl, *_ = stats.linregress(bs[xcol], bs[f"rel_{name}"])
            boots[name].append(sl)
    return point, {k: np.asarray(v) for k, v in boots.items()}


for xcol, xlabel in (("d_theta_PVS", "|dtheta_PVS|"), ("d_theta_SCR", "|dtheta_SCR|")):
    point, boots = cluster_boot_slopes(pairs, xcol)
    say()
    say(f"--- relative |change in ALPS| (%) vs {xlabel} (deg) ---")
    say(f"{'Metric':<10s} {'slope %/deg':>12s} {'95% CI':>18s} {'r':>7s} "
        f"{'vs Classic':>20s} {'p':>7s}")
    for name in METRICS:
        lo, hi = np.percentile(boots[name], [2.5, 97.5])
        if name == REFERENCE:
            diff_txt, p_txt = "reference", ""
        else:
            d = boots[name] - boots[REFERENCE]
            d_point = point[name]["slope"] - point[REFERENCE]["slope"]
            d_lo, d_hi = np.percentile(d, [2.5, 97.5])
            p = 2.0 * min((d <= 0).mean(), (d >= 0).mean())
            p = min(1.0, max(p, 1.0 / len(d)))
            diff_txt = f"{d_point:+.4f} [{d_lo:+.3f},{d_hi:+.3f}]"
            p_txt = f"{p:.3f}"
        say(
            f"{name:<10s} {point[name]['slope']:12.4f} [{lo:7.4f},{hi:7.4f}] "
            f"{point[name]['r']:7.3f} {diff_txt:>20s} {p_txt:>7s}"
        )
        pair_rows_out = {
            "predictor": xcol,
            "metric": name,
            "slope_pct_per_deg": point[name]["slope"],
            "slope_lo": lo,
            "slope_hi": hi,
            "pearson_r": point[name]["r"],
            "p_ols": point[name]["p"],
            "n_obs": len(pairs),
            "n_subjects": len(pair_ids),
        }
        pairs.attrs.setdefault("summary", []).append(pair_rows_out)

pd.DataFrame(pairs.attrs["summary"]).to_csv(HERE / f"repositioning_table{SUF}.csv", index=False)
pairs.to_csv(HERE / f"repositioning_pairs{SUF}.csv", index=False)

say()
say("--- Paired between-wave change, each variant against Classic ---")
say("Same subject, same wave pair, same hemisphere, so this is a paired contrast.")
say()
say(f"{'Metric':<10s} {'median %':>9s} {'mean %':>8s} {'mean diff vs Classic':>22s} "
    f"{'95% CI':>18s} {'p_boot':>8s} {'p_wilcox':>9s}")
pair_ids_arr = np.asarray(pair_ids)
boot_idx = [
    RNG.choice(len(pair_ids_arr), size=len(pair_ids_arr), replace=True)
    for _ in range(N_BOOT)
]
blocks = {sid: pairs[pairs["Subject_ID"] == sid] for sid in pair_ids}
change_rows = []
for name in METRICS:
    med = pairs[f"rel_{name}"].median()
    mean = pairs[f"rel_{name}"].mean()
    if name == REFERENCE:
        say(f"{name:<10s} {med:9.2f} {mean:8.2f} {'reference':>22s}")
        change_rows.append(
            {"metric": name, "median_pct": med, "mean_pct": mean,
             "mean_diff_vs_classic": 0.0, "ci_lo": np.nan, "ci_hi": np.nan,
             "p_boot": np.nan, "p_wilcoxon": np.nan}
        )
        continue
    diff_obs = pairs[f"rel_{name}"] - pairs[f"rel_{REFERENCE}"]
    d_point = float(diff_obs.mean())
    boot = np.empty(N_BOOT)
    for b, picks in enumerate(boot_idx):
        bs = pd.concat([blocks[pair_ids_arr[i]] for i in picks], ignore_index=True)
        boot[b] = float((bs[f"rel_{name}"] - bs[f"rel_{REFERENCE}"]).mean())
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p_boot = 2.0 * min((boot <= 0).mean(), (boot >= 0).mean())
    p_boot = min(1.0, max(p_boot, 1.0 / N_BOOT))
    p_wil = float(stats.wilcoxon(pairs[f"rel_{name}"], pairs[f"rel_{REFERENCE}"]).pvalue)
    say(f"{name:<10s} {med:9.2f} {mean:8.2f} {d_point:22.3f} "
        f"[{lo:7.3f},{hi:7.3f}] {p_boot:8.3f} {p_wil:9.4f}")
    change_rows.append(
        {"metric": name, "median_pct": med, "mean_pct": mean,
         "mean_diff_vs_classic": d_point, "ci_lo": lo, "ci_hi": hi,
         "p_boot": p_boot, "p_wilcoxon": p_wil}
    )
say()
say("Negative differences favour the corrected variant (smaller between-wave change).")
say("The Wilcoxon p treats the 78 observations as independent and so is anti-")
say("conservative; the clustered bootstrap CI is the one to quote.")
pd.DataFrame(change_rows).to_csv(HERE / f"between_wave_change{SUF}.csv", index=False)


# ---------------------------------------------------------------------------
# 3 and 4. Left/right separate age correlations, Williams test
# ---------------------------------------------------------------------------


def williams_t(r_jk: float, r_jh: float, r_kh: float, n: int) -> tuple[float, float]:
    """Williams test for two dependent overlapping correlations sharing variable j."""
    det = 1 - r_jk**2 - r_jh**2 - r_kh**2 + 2 * r_jk * r_jh * r_kh
    rbar = (r_jk + r_jh) / 2.0
    denom = (2 * (n - 1) / (n - 3)) * det + (rbar**2) * (1 - r_kh) ** 3
    t = (r_jk - r_jh) * np.sqrt(((n - 1) * (1 + r_kh)) / denom)
    p = 2 * stats.t.sf(abs(t), df=n - 3)
    return float(t), float(p)


header("3. LEFT AND RIGHT ANALYSED SEPARATELY AGAINST AGE")
say()
say("Reviewer 4 point 4. The submitted analysis averaged L and R before correlating.")

age_rows = []
for side_label, key in (("L", 0), ("R", 1), ("avg", 2)):
    say()
    say(f"--- {side_label} (n={len(cc)} sessions) ---")
    say(f"{'Metric':<10s} {'r':>8s} {'p':>10s} {'slope/yr':>10s}")
    rs = {}
    for name, cols in METRICS.items():
        col = cols[key]
        sl, ic, r, p, se = stats.linregress(cc["Age"], cc[col])
        rs[name] = r
        say(f"{name:<10s} {r:8.3f} {p:10.6f} {sl:10.6f}")
        age_rows.append(
            {"side": side_label, "metric": name, "n": len(cc), "r": r, "p": p, "slope": sl}
        )
    say()
    say(f"{'Williams test vs Classic':<28s} {'t':>8s} {'p':>8s}")
    for name in METRICS:
        if name == REFERENCE:
            continue
        col_a = METRICS[REFERENCE][key]
        col_b = METRICS[name][key]
        r_kh = float(np.corrcoef(cc[col_a], cc[col_b])[0, 1])
        t, p = williams_t(rs[REFERENCE], rs[name], r_kh, len(cc))
        say(f"  Classic vs {name:<15s} {t:8.3f} {p:8.3f}   "
            f"(r between metrics = {r_kh:.3f})")
        age_rows.append(
            {
                "side": side_label,
                "metric": f"Williams Classic vs {name}",
                "n": len(cc),
                "r": r_kh,
                "p": p,
                "slope": t,
            }
        )

say()
say("--- Left versus right asymmetry (L minus R) vs age ---")
say(f"{'Metric':<10s} {'r':>8s} {'p':>10s}")
for name, (lcol, rcol, _) in METRICS.items():
    asym = cc[lcol] - cc[rcol]
    sl, ic, r, p, se = stats.linregress(cc["Age"], asym)
    say(f"{name:<10s} {r:8.3f} {p:10.6f}")
    age_rows.append(
        {"side": "L-R", "metric": name, "n": len(cc), "r": r, "p": p, "slope": sl}
    )

pd.DataFrame(age_rows).to_csv(HERE / f"age_lr_table{SUF}.csv", index=False)


# ---------------------------------------------------------------------------
# 5. Non-independence of the published age correlation
# ---------------------------------------------------------------------------

header("5. NON-INDEPENDENCE CHECK ON THE PUBLISHED AGE CORRELATION")
say()
say(f"The submitted Table 1 treats {len(cc)} sessions as independent, but they come")
say(f"from {cc.Subject_ID.nunique()} subjects. Two robustness versions follow.")

first = cc.sort_values(["Subject_ID", "wave"]).groupby("Subject_ID", as_index=False).first()
say()
say(f"--- One session per subject, earliest wave (n={len(first)}) ---")
say(f"{'Metric':<10s} {'r':>8s} {'p':>10s}")
for name, (_, _, acol) in METRICS.items():
    sl, ic, r, p, se = stats.linregress(first["Age"], first[acol])
    say(f"{name:<10s} {r:8.3f} {p:10.6f}")

say()
say(f"--- OLS with subject-clustered robust SEs, all {len(cc)} sessions ---")
say(f"{'Metric':<10s} {'beta/yr':>10s} {'SE_naive':>10s} {'SE_clust':>10s} "
    f"{'t':>7s} {'p':>10s}")
for name, (_, _, acol) in METRICS.items():
    d = cc[["Subject_ID", "Age", acol]].dropna().rename(columns={acol: "y"}).copy()
    d["Age_c"] = d["Age"] - d["Age"].mean()
    naive = smf.ols("y ~ Age_c", d).fit()
    clust = smf.ols("y ~ Age_c", d).fit(
        cov_type="cluster", cov_kwds={"groups": d["Subject_ID"]}
    )
    b = float(clust.params["Age_c"])
    say(
        f"{name:<10s} {b:10.6f} {float(naive.bse['Age_c']):10.6f} "
        f"{float(clust.bse['Age_c']):10.6f} {float(clust.tvalues['Age_c']):7.3f} "
        f"{float(clust.pvalues['Age_c']):10.6f}"
    )

say()
say(f"--- Mixed model, metric ~ Age + (1|Subject), all {len(cc)} sessions ---")
say(f"{'Metric':<10s} {'beta/yr':>10s} {'SE':>9s} {'z':>7s} {'p':>10s}")
for name, (_, _, acol) in METRICS.items():
    d = cc[["Subject_ID", "Age", acol]].dropna().rename(columns={acol: "y"}).copy()
    d["Age_c"] = d["Age"] - d["Age"].mean()
    fit = None
    for method in ("powell", "bfgs", "cg", "nm"):
        try:
            fit = smf.mixedlm("y ~ Age_c", d, groups=d["Subject_ID"]).fit(
                reml=True, method=method
            )
            break
        except Exception:
            continue
    if fit is None:
        say(f"{name:<10s}   did not converge, see clustered OLS above")
        continue
    b = float(fit.params["Age_c"])
    se = float(fit.bse["Age_c"])
    say(f"{name:<10s} {b:10.6f} {se:9.6f} {b/se:7.3f} {float(fit.pvalues['Age_c']):10.6f}")

(HERE / f"longitudinal_reliability_report{SUF}.txt").write_text("\n".join(lines), encoding="utf-8")
print(f"\nWrote {HERE / 'longitudinal_reliability_report.txt'}")
