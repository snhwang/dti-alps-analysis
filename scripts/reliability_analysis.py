"""
Reliability, repositioning and age analyses for any DLBS ALPS result table.

Parameterised so the manually drawn ROIs used in the MRI submission and the
atlas-derived automated ROIs can be run through an identical pipeline, making
the two directly comparable. Reviewers 1 and 4 both ask whether orientation
correction buys anything measurable; with automated ROIs the between-wave
comparison is no longer contaminated by ROI redrawing.

Usage
    python reliability_analysis.py --input ../../diffusion/HCP/lifespan_alps_results.csv \
        --label manual
    python reliability_analysis.py --input ../../diffusion/DLBS/dlbs_alps_auto_cubic.csv \
        --label auto --motion ../../diffusion/DLBS/dlbs_motion.csv

Outputs, suffixed by label:
    reliability_<label>.csv, repositioning_<label>.csv, age_<label>.csv,
    report_<label>.txt
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

from alps_common import parse_age

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
N_BOOT = 2000
MOTION_THRESHOLD = 0.5
FA_DROP_THRESHOLD = 20.0
MIN_ROI_VOXELS = 4

METRICS = {
    "Classic": ("Traditional_L", "Traditional_R", "Traditional_Avg"),
    "Refined": ("Refined_L", "Refined_R", "Refined_Avg"),
    "Refined+": ("RefinedPlus_L", "RefinedPlus_R", "RefinedPlus_Avg"),
    "ALPS-PAS": ("ALPS_PAS_L", "ALPS_PAS_R", "ALPS_PAS_Avg"),
}
REFERENCE = "Classic"
WAVE_ORDER = {"ses-wave1": 1, "ses-wave2": 2, "ses-wave3": 3}

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
# Loading and QC
# ---------------------------------------------------------------------------


def load_cohort(args) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    df = pd.read_csv(args.input)

    if args.motion:
        motion = pd.read_csv(args.motion)
        mcols = [c for c in ("Eddy_Mean_RMS", "Eddy_Max_RMS", "Eddy_Mean_RMS_relprev")
                 if c in motion]
        df = df.drop(columns=[c for c in mcols if c in df.columns], errors="ignore")
        # DLBS motion is keyed by session id; HCP-A motion is keyed by
        # participant and visit, because it is read from the source zips.
        if "DTI_Session_ID" in motion.columns and "DTI_Session_ID" in df.columns:
            on = ["DTI_Session_ID"]
        elif {"Subject_ID", "Visit"} <= set(motion.columns) <= set(motion.columns) and "Visit" in df.columns:
            on = ["Subject_ID", "Visit"]
        else:
            raise SystemExit("cannot key the motion table to the ALPS table")
        df = df.merge(motion[on + mcols], on=on, how="left")

    dev = pd.read_csv(args.dev) if args.dev else None

    n0 = len(df)
    if "status" in df.columns:
        df = df[df["status"].astype(str) == "ok"]
    n_status = len(df)

    for cols in METRICS.values():
        for c in cols[:2]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Age"] = parse_age(df["Age"])

    # Recompute the L/R average so it is consistent across input tables.
    for name, (lcol, rcol, acol) in METRICS.items():
        df[acol] = df[[lcol, rcol]].mean(axis=1)

    need = ["Age"] + [c for cols in METRICS.values() for c in cols[:2]]
    df = df.dropna(subset=need)
    n_complete = len(df)

    if "Max_Pct_Dropped" in df.columns:
        df = df[pd.to_numeric(df["Max_Pct_Dropped"], errors="coerce") <= FA_DROP_THRESHOLD]

    # Motion QC. The 0.5 mm criterion published for DLBS cannot be carried over
    # to HCP-A unchanged: RMS is measured relative to the first volume and so
    # accumulates with acquisition length, and HCP-A runs 399 volumes against
    # DLBS's 31. At 0.5 mm it would exclude 99% of HCP-A. Passing
    # --motion-exclude-pct instead sets the threshold at the cohort's own
    # distribution so the same fraction is excluded, matching the stringency of
    # the published QC rather than its numeric value.
    motion_note = ""
    if "Eddy_Mean_RMS" in df.columns:
        rms = pd.to_numeric(df["Eddy_Mean_RMS"], errors="coerce")
        if args.motion_exclude_pct is not None:
            thr = float(np.nanpercentile(rms.dropna(), 100.0 - args.motion_exclude_pct))
            motion_note = (f"top {args.motion_exclude_pct:.1f}% by mean RMS "
                           f"(threshold {thr:.3f} mm)")
        else:
            thr = args.motion_threshold
            motion_note = f"mean RMS > {thr:.3f} mm"
        df = df[rms <= thr]
    for c in ("n_proj", "n_assoc"):
        if c in df.columns:
            df = df[pd.to_numeric(df[c], errors="coerce") >= MIN_ROI_VOXELS]

    df = df.copy()
    # DLBS labels visits "ses-waveN" in a Session column; HCP-A labels them
    # "VN" in a Visit column. Accept either so both cohorts run unchanged.
    if "Session" in df.columns:
        df["wave"] = df["Session"].map(WAVE_ORDER)
        if df["wave"].isna().all():
            df["wave"] = df["Session"].astype(str).str.extract(r"(\d+)").astype(float)
    elif "Visit" in df.columns:
        df["Session"] = df["Visit"]
        df["wave"] = df["Visit"].astype(str).str.extract(r"(\d+)").astype(float)
    else:
        raise SystemExit("input needs a Session or Visit column")
    df = df.dropna(subset=["wave"])
    df = df.sort_values(["Subject_ID", "wave"]).reset_index(drop=True)

    header("COHORT")
    say(f"Input                       : {args.input}")
    say(f"Rows in file                : {n0}")
    if "status" in pd.read_csv(args.input, nrows=1).columns:
        say(f"After status == ok          : {n_status}")
    say(f"Complete age and L/R values : {n_complete}")
    say(f"Motion QC                   : excluded {motion_note}")
    say(f"After motion and ROI QC     : {len(df)} sessions, "
        f"{df.Subject_ID.nunique()} subjects")
    return df, dev


# ---------------------------------------------------------------------------
# Variance components
# ---------------------------------------------------------------------------


def variance_components(df: pd.DataFrame, col: str, adjust_age: bool) -> dict | None:
    """Unbalanced one-way random-effects ANOVA estimator, closed form."""
    d = df[["Subject_ID", "Age", col]].dropna().rename(columns={col: "y"}).copy()
    k = d["Subject_ID"].nunique()
    if k < 3 or len(d) <= k:
        return None

    grand = float(d["y"].mean())
    if adjust_age:
        sl, ic, *_ = stats.linregress(d["Age"], d["y"])
        d["y"] = d["y"] - (sl * d["Age"] + ic) + grand

    n_i = d.groupby("Subject_ID")["y"].size().to_numpy(dtype=float)
    m_i = d.groupby("Subject_ID")["y"].mean().to_numpy(dtype=float)
    N = float(n_i.sum())
    ybar = float((n_i * m_i).sum() / N)

    ms_b = float((n_i * (m_i - ybar) ** 2).sum()) / (k - 1)
    ms_w = float(
        d.groupby("Subject_ID")["y"].transform(lambda s: s - s.mean()).pow(2).sum()
    ) / (N - k)
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


def williams_t(r_jk: float, r_jh: float, r_kh: float, n: int) -> tuple[float, float]:
    """Williams test for dependent overlapping correlations sharing variable j."""
    det = 1 - r_jk**2 - r_jh**2 - r_kh**2 + 2 * r_jk * r_jh * r_kh
    rbar = (r_jk + r_jh) / 2.0
    denom = (2 * (n - 1) / (n - 3)) * det + (rbar**2) * (1 - r_kh) ** 3
    t = (r_jk - r_jh) * np.sqrt(((n - 1) * (1 + r_kh)) / denom)
    return float(t), float(2 * stats.t.sf(abs(t), df=n - 3))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--motion", default=None)
    ap.add_argument("--dev", default=None)
    ap.add_argument("--motion-threshold", type=float, default=MOTION_THRESHOLD,
                    help="absolute mean-RMS cutoff in mm (DLBS published value 0.5)")
    ap.add_argument("--motion-exclude-pct", type=float, default=None,
                    help="instead exclude this percent of the cohort by motion, "
                         "matching the stringency of the DLBS QC (23.6)")
    args = ap.parse_args()

    rng = np.random.default_rng(20260726)
    cc, dev = load_cohort(args)

    counts = cc["Subject_ID"].value_counts()
    repeat_ids = sorted(counts[counts >= 2].index)
    lon = cc[cc["Subject_ID"].isin(repeat_ids)].copy()

    say(f"Subjects with >= 2 waves    : {len(repeat_ids)}")
    say(f"Subjects with 3 waves       : {int((counts >= 3).sum())}")
    say(f"Sessions in longitudinal set: {len(lon)}")
    if len(lon):
        gaps = lon.groupby("Subject_ID")["Age"].agg(lambda s: s.max() - s.min())
        say(f"Within-subject age span     : median {gaps.median():.1f} y, "
            f"range {gaps.min():.1f} to {gaps.max():.1f}")

    # ---------------- 1. Between-wave reliability ----------------
    header("1. BETWEEN-WAVE RELIABILITY")
    rel_rows = []
    for adjust_age in (False, True):
        label = "age-adjusted" if adjust_age else "unadjusted"
        point = {n: variance_components(lon, c[2], adjust_age) for n, c in METRICS.items()}

        boot = {n: np.full(N_BOOT, np.nan) for n in METRICS}
        for b in range(N_BOOT):
            picks = rng.choice(len(repeat_ids), size=len(repeat_ids), replace=True)
            bs = pd.concat(
                [
                    lon[lon["Subject_ID"] == repeat_ids[i]].assign(Subject_ID=f"b{k:04d}")
                    for k, i in enumerate(picks)
                ],
                ignore_index=True,
            )
            for n, c in METRICS.items():
                vc = variance_components(bs, c[2], adjust_age)
                if vc:
                    boot[n][b] = vc["icc"]

        say()
        say(f"--- L/R average, {label} "
            f"({lon.Subject_ID.nunique()} subjects, {len(lon)} sessions) ---")
        say(f"{'Metric':<10s} {'ICC':>6s} {'95% CI':>16s} {'wCV %':>7s} "
            f"{'dICC vs Classic':>24s} {'p':>7s}")
        for n in METRICS:
            vc = point[n]
            if vc is None:
                say(f"{n:<10s}   not estimable")
                continue
            lo, hi = np.nanpercentile(boot[n], [2.5, 97.5])
            if n == REFERENCE:
                dtxt, ptxt = "reference", ""
            else:
                d = boot[n] - boot[REFERENCE]
                d = d[np.isfinite(d)]
                dp = vc["icc"] - point[REFERENCE]["icc"]
                dlo, dhi = np.percentile(d, [2.5, 97.5])
                p = 2.0 * min((d <= 0).mean(), (d >= 0).mean())
                p = min(1.0, max(p, 1.0 / max(len(d), 1)))
                dtxt, ptxt = f"{dp:+.3f} [{dlo:+.3f},{dhi:+.3f}]", f"{p:.3f}"
            say(f"{n:<10s} {vc['icc']:6.3f} [{lo:5.3f},{hi:5.3f}] "
                f"{vc['wcv_pct']:7.2f} {dtxt:>24s} {ptxt:>7s}")
            rel_rows.append({"model": label, "metric": n, "icc": vc["icc"],
                             "icc_lo": lo, "icc_hi": hi, "wcv_pct": vc["wcv_pct"],
                             "n_subjects": lon.Subject_ID.nunique(),
                             "n_sessions": len(lon)})
    pd.DataFrame(rel_rows).to_csv(HERE / f"reliability_{args.label}.csv", index=False)

    # ---------------- 2. Between-wave change and repositioning ----------------
    header("2. BETWEEN-WAVE CHANGE")
    pair_rows = []
    dev_idx = None
    if dev is not None:
        dev_idx = dev.set_index("DTI_Session_ID")
    for sid, block in lon.groupby("Subject_ID"):
        for (_, a), (_, b) in combinations(block.sort_values("wave").iterrows(), 2):
            for side in ("L", "R"):
                row = {"Subject_ID": sid, "side": side,
                       "d_age": abs(float(b["Age"]) - float(a["Age"]))}
                for name, cols in METRICS.items():
                    col = cols[0] if side == "L" else cols[1]
                    va, vb = float(a[col]), float(b[col])
                    row[f"rel_{name}"] = 100.0 * abs(vb - va) / ((va + vb) / 2.0)
                if dev_idx is not None:
                    ok = True
                    for ang in ("theta_PVS", "theta_SCR", "theta_SLF"):
                        c = f"{ang}_{side}"
                        try:
                            x = float(dev_idx.loc[a["DTI_Session_ID"], c])
                            y = float(dev_idx.loc[b["DTI_Session_ID"], c])
                        except Exception:
                            ok = False
                            break
                        if not (np.isfinite(x) and np.isfinite(y)):
                            ok = False
                            break
                        row[f"d_{ang}"] = abs(y - x)
                    if not ok:
                        for ang in ("theta_PVS", "theta_SCR", "theta_SLF"):
                            row[f"d_{ang}"] = np.nan
                pair_rows.append(row)

    pairs = pd.DataFrame(pair_rows)
    pair_ids = sorted(pairs["Subject_ID"].unique())
    say()
    say(f"Wave pairs x hemispheres: {len(pairs)} observations "
        f"from {len(pair_ids)} subjects")
    say()
    say(f"{'Metric':<10s} {'median %':>9s} {'mean %':>8s} "
        f"{'mean diff vs Classic':>21s} {'95% CI':>18s} {'p_boot':>8s}")
    blocks = {s: pairs[pairs["Subject_ID"] == s] for s in pair_ids}
    boot_idx = [rng.choice(len(pair_ids), size=len(pair_ids), replace=True)
                for _ in range(N_BOOT)]
    change_rows = []
    for n in METRICS:
        med, mean = pairs[f"rel_{n}"].median(), pairs[f"rel_{n}"].mean()
        if n == REFERENCE:
            say(f"{n:<10s} {med:9.2f} {mean:8.2f} {'reference':>21s}")
            change_rows.append({"metric": n, "median_pct": med, "mean_pct": mean})
            continue
        dp = float((pairs[f"rel_{n}"] - pairs[f"rel_{REFERENCE}"]).mean())
        bt = np.array([
            float((pd.concat([blocks[pair_ids[i]] for i in picks], ignore_index=True)
                   .pipe(lambda x: x[f"rel_{n}"] - x[f"rel_{REFERENCE}"])).mean())
            for picks in boot_idx
        ])
        lo, hi = np.percentile(bt, [2.5, 97.5])
        p = min(1.0, max(2.0 * min((bt <= 0).mean(), (bt >= 0).mean()), 1.0 / N_BOOT))
        say(f"{n:<10s} {med:9.2f} {mean:8.2f} {dp:21.3f} [{lo:7.3f},{hi:7.3f}] {p:8.3f}")
        change_rows.append({"metric": n, "median_pct": med, "mean_pct": mean,
                            "mean_diff_vs_classic": dp, "ci_lo": lo, "ci_hi": hi,
                            "p_boot": p})
    say()
    say("Negative differences favour the corrected variant.")

    if dev_idx is not None and pairs["d_theta_SCR"].notna().any():
        for xcol, xlab in (("d_theta_PVS", "|dtheta_PVS|"), ("d_theta_SCR", "|dtheta_SCR|")):
            sub = pairs.dropna(subset=[xcol])
            say()
            say(f"--- relative |change| (%) vs {xlab} (deg), n={len(sub)} ---")
            say(f"{'Metric':<10s} {'slope %/deg':>12s} {'95% CI':>18s} {'r':>7s}")
            sids = sorted(sub["Subject_ID"].unique())
            sblocks = {s: sub[sub["Subject_ID"] == s] for s in sids}
            for n in METRICS:
                sl, ic, r, p, se = stats.linregress(sub[xcol], sub[f"rel_{n}"])
                bt = []
                for _ in range(N_BOOT):
                    picks = rng.choice(len(sids), size=len(sids), replace=True)
                    bs = pd.concat([sblocks[sids[i]] for i in picks], ignore_index=True)
                    bt.append(stats.linregress(bs[xcol], bs[f"rel_{n}"])[0])
                lo, hi = np.percentile(bt, [2.5, 97.5])
                say(f"{n:<10s} {sl:12.4f} [{lo:7.4f},{hi:7.4f}] {r:7.3f}")

    pd.DataFrame(change_rows).to_csv(HERE / f"repositioning_{args.label}.csv", index=False)

    # ---------------- 3. Age associations ----------------
    header("3. AGE ASSOCIATIONS")
    age_rows = []
    for side_label, key in (("L", 0), ("R", 1), ("avg", 2)):
        say()
        say(f"--- {side_label} (n={len(cc)} sessions, "
            f"{cc.Subject_ID.nunique()} subjects) ---")
        say(f"{'Metric':<10s} {'r':>8s} {'p':>11s} {'slope/yr':>10s}")
        rs = {}
        for n, cols in METRICS.items():
            sl, ic, r, p, se = stats.linregress(cc["Age"], cc[cols[key]])
            rs[n] = r
            say(f"{n:<10s} {r:8.3f} {p:11.3e} {sl:10.6f}")
            age_rows.append({"side": side_label, "metric": n, "n": len(cc),
                             "r": r, "p": p, "slope": sl})
        say(f"{'Williams vs Classic':<30s} {'t':>8s} {'p':>8s}")
        for n in METRICS:
            if n == REFERENCE:
                continue
            r_kh = float(np.corrcoef(cc[METRICS[REFERENCE][key]], cc[METRICS[n][key]])[0, 1])
            t, p = williams_t(rs[REFERENCE], rs[n], r_kh, len(cc))
            say(f"  Classic vs {n:<17s} {t:8.3f} {p:8.3f}  (r={r_kh:.3f})")
            age_rows.append({"side": side_label, "metric": f"Williams vs {n}",
                             "n": len(cc), "r": r_kh, "p": p, "slope": t})

    say()
    say("--- Subject-clustered robust SEs and one-session-per-subject ---")
    first = cc.groupby("Subject_ID", as_index=False).first()
    say(f"{'Metric':<10s} {'r (wave1 only)':>15s} {'p':>11s} "
        f"{'beta/yr (all)':>14s} {'p (clustered)':>14s}")
    for n, cols in METRICS.items():
        sl1, _, r1, p1, _ = stats.linregress(first["Age"], first[cols[2]])
        d = cc[["Subject_ID", "Age", cols[2]]].dropna().rename(columns={cols[2]: "y"}).copy()
        d["Age_c"] = d["Age"] - d["Age"].mean()
        cl = smf.ols("y ~ Age_c", d).fit(
            cov_type="cluster", cov_kwds={"groups": d["Subject_ID"]}
        )
        say(f"{n:<10s} {r1:15.3f} {p1:11.3e} {float(cl.params['Age_c']):14.6f} "
            f"{float(cl.pvalues['Age_c']):14.3e}")
    say(f"(one session per subject: n={len(first)})")

    pd.DataFrame(age_rows).to_csv(HERE / f"age_{args.label}.csv", index=False)

    out = HERE / f"report_{args.label}.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
