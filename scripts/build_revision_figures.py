"""
Figures for the MRI revision, built from the spherical-ROI DLBS cohort.

Figure R1  Repositioning sensitivity
  (a) Relative between-wave |change| in ALPS against the between-wave change in
      the SCR-to-scanner-z angle, for Classic and Refined, with OLS fits and
      subject-clustered bootstrap bands.
  (b) Slope with 95% CI for all four variants, so Refined+ and ALPS-PAS are
      reported without needing four hues in the scatter.

Figure R2  Orientation as a confound of the age association
  (a) SCR-to-scanner-z angle against age.
  (b) Standardised age coefficient before and after adjusting for the three
      scanner-to-anatomy deviation angles.

Sized at final print width so \\includegraphics[width=\\linewidth] is 1:1.
Colours are the Classic-blue / Refined-orange pair, which passes CVD
separation (worst adjacent dE 24.6 protan); every series is also directly
labelled so identity never rests on colour alone.

Output: fig_repositioning.png, fig_orientation_confound.png
"""

from __future__ import annotations

import warnings
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from alps_common import parse_age

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
DIFF = HERE.parent.parent / "diffusion"

MOTION_THRESHOLD = 0.5
MIN_ROI_VOXELS = 4
N_BOOT = 2000
RNG = np.random.default_rng(20260727)

C_CLASSIC = "#1f77b4"
C_REFINED = "#ff7f0e"
C_INK = "#222222"
C_MUTED = "#6b6b6b"
C_GRID = "#dddddd"

METRICS = {
    "Classic": ("Traditional_L", "Traditional_R", "Traditional_Avg"),
    "Refined": ("Refined_L", "Refined_R", "Refined_Avg"),
    "Refined+": ("RefinedPlus_L", "RefinedPlus_R", "RefinedPlus_Avg"),
    "ALPS-PAS": ("ALPS_PAS_L", "ALPS_PAS_R", "ALPS_PAS_Avg"),
}

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.edgecolor": "#999999",
    "axes.linewidth": 0.8,
    "figure.dpi": 300,
})


def load() -> pd.DataFrame:
    # Spherical ROIs, not the superseded cubes. The manuscript reports the
    # sphere-based analysis throughout, and these figures illustrate it.
    alps = pd.read_csv(DIFF / "DLBS" / "dlbs_alps_spheres_5mm.csv")
    dev = pd.read_csv(DIFF / "DLBS" / "dlbs_alps_spheres_axis_deviations.csv")
    motion = pd.read_csv(DIFF / "DLBS" / "dlbs_motion.csv")

    df = alps[alps["status"].astype(str) == "ok"].copy()
    df = df.merge(
        dev.drop(columns=[c for c in ("Age", "Subject_ID", "Session") if c in dev]),
        on="DTI_Session_ID", how="left")
    df = df.merge(motion[["DTI_Session_ID", "Eddy_Mean_RMS"]],
                  on="DTI_Session_ID", how="left")

    for cols in METRICS.values():
        for c in cols[:2]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for name, (l, r, a) in METRICS.items():
        df[a] = df[[l, r]].mean(axis=1)
    df["Age"] = parse_age(df["Age"])

    df = df.dropna(subset=["Age"] + [c for cols in METRICS.values() for c in cols[:2]])
    df = df[pd.to_numeric(df["Eddy_Mean_RMS"], errors="coerce") <= MOTION_THRESHOLD]
    for c in ("n_proj", "n_assoc"):
        df = df[pd.to_numeric(df[c], errors="coerce") >= MIN_ROI_VOXELS]

    df["wave"] = df["Session"].map({"ses-wave1": 1, "ses-wave2": 2, "ses-wave3": 3})
    return df.sort_values(["Subject_ID", "wave"]).reset_index(drop=True)


def build_pairs(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["Subject_ID"].value_counts()
    lon = df[df["Subject_ID"].isin(counts[counts >= 2].index)]
    rows = []
    for sid, block in lon.groupby("Subject_ID"):
        for (_, a), (_, b) in combinations(block.sort_values("wave").iterrows(), 2):
            for side in ("L", "R"):
                row = {"Subject_ID": sid, "side": side}
                ok = True
                for ang in ("theta_PVS", "theta_SCR", "theta_SLF"):
                    c = f"{ang}_{side}"
                    x, y = a.get(c, np.nan), b.get(c, np.nan)
                    if not (np.isfinite(x) and np.isfinite(y)):
                        ok = False
                        break
                    row[f"d_{ang}"] = abs(float(y) - float(x))
                if not ok:
                    continue
                for name, cols in METRICS.items():
                    col = cols[0] if side == "L" else cols[1]
                    va, vb = float(a[col]), float(b[col])
                    row[f"rel_{name}"] = 100.0 * abs(vb - va) / ((va + vb) / 2.0)
                rows.append(row)
    return pd.DataFrame(rows)


def cluster_boot(sub: pd.DataFrame, xcol: str, ycol: str) -> tuple:
    ids = sorted(sub["Subject_ID"].unique())
    blocks = {s: sub[sub["Subject_ID"] == s] for s in ids}
    sl, ic, r, p, se = stats.linregress(sub[xcol], sub[ycol])
    boots = np.empty((N_BOOT, 2))
    for b in range(N_BOOT):
        picks = RNG.choice(len(ids), size=len(ids), replace=True)
        bs = pd.concat([blocks[ids[i]] for i in picks], ignore_index=True)
        s2, i2, *_ = stats.linregress(bs[xcol], bs[ycol])
        boots[b] = (s2, i2)
    lo, hi = np.percentile(boots[:, 0], [2.5, 97.5])
    return sl, ic, r, lo, hi, boots


# ---------------------------------------------------------------------------

df = load()
pairs = build_pairs(df)
print(f"QC sessions {len(df)}, subjects {df.Subject_ID.nunique()}")
print(f"wave-pair x hemisphere observations {len(pairs)}, "
      f"subjects {pairs.Subject_ID.nunique()}")

XCOL = "d_theta_SCR"
fits = {}
for name in METRICS:
    fits[name] = cluster_boot(pairs, XCOL, f"rel_{name}")

# ---- Figure R1 ------------------------------------------------------------

fig = plt.figure(figsize=(6.9, 2.9))
gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0], wspace=0.34)
ax = fig.add_subplot(gs[0, 0])

xg = np.linspace(0, np.percentile(pairs[XCOL], 99), 100)
for name, colour in (("Classic", C_CLASSIC), ("Refined", C_REFINED)):
    sl, ic, r, lo, hi, boots = fits[name]
    ax.scatter(pairs[XCOL], pairs[f"rel_{name}"], s=7, alpha=0.25,
               color=colour, edgecolors="none", zorder=2)
    band = boots[:, 0][:, None] * xg[None, :] + boots[:, 1][:, None]
    b_lo, b_hi = np.percentile(band, [2.5, 97.5], axis=0)
    ax.fill_between(xg, b_lo, b_hi, color=colour, alpha=0.18, lw=0, zorder=3)
    ax.plot(xg, sl * xg + ic, color=colour, lw=2, zorder=4,
            label=f"{name}  {sl:+.2f} %/deg")

ax.set_xlim(0, xg[-1])
ax.set_ylim(0, np.percentile(pairs[["rel_Classic", "rel_Refined"]].values, 98))
ax.set_xlabel(r"between-wave $|\Delta\theta_{\mathrm{SCR}}|$ (degrees)")
ax.set_ylabel("between-wave |change| in ALPS (%)")
ax.grid(True, color=C_GRID, lw=0.5, zorder=0)
ax.set_axisbelow(True)
leg = ax.legend(frameon=False, loc="upper left", handlelength=1.6, borderaxespad=0.2)
for t, c in zip(leg.get_texts(), (C_CLASSIC, C_REFINED)):
    t.set_color(C_INK)
ax.set_title("(a) real repositioning between waves", loc="left", color=C_INK)

ax2 = fig.add_subplot(gs[0, 1])
names = list(METRICS)
ys = np.arange(len(names))[::-1]
for y, name in zip(ys, names):
    sl, ic, r, lo, hi, _ = fits[name]
    colour = C_CLASSIC if name == "Classic" else (
        C_REFINED if name == "Refined" else C_MUTED)
    ax2.plot([lo, hi], [y, y], color=colour, lw=2, solid_capstyle="round", zorder=3)
    ax2.plot([sl], [y], "o", ms=6, color=colour, mec="white", mew=1.0, zorder=4)
ax2.axvline(0, color=C_INK, lw=0.9, ls="--", zorder=2)
ax2.set_yticks(ys)
ax2.set_yticklabels(names, color=C_INK)
ax2.set_ylim(-0.6, len(names) - 0.4)
ax2.set_xlabel("slope (% per degree)")
ax2.grid(True, axis="x", color=C_GRID, lw=0.5, zorder=0)
ax2.set_axisbelow(True)
ax2.set_title("(b) slope, 95% CI", loc="left", color=C_INK)

fig.savefig(HERE.parent / "fig_repositioning.png", bbox_inches="tight", dpi=300)
plt.close(fig)
print("wrote fig_repositioning.png")
for name in names:
    sl, ic, r, lo, hi, _ = fits[name]
    print(f"  {name:<9s} slope {sl:+.4f} [{lo:+.4f},{hi:+.4f}]  r={r:+.3f}")

# ---- Figure R2 ------------------------------------------------------------

for ang in ("theta_PVS", "theta_SCR", "theta_SLF"):
    df[f"{ang}_avg"] = df[[f"{ang}_L", f"{ang}_R"]].mean(axis=1)
d2 = df.dropna(subset=["theta_PVS_avg", "theta_SCR_avg", "theta_SLF_avg"]).copy()

import statsmodels.formula.api as smf

raw_b, adj_b = {}, {}
for name, (_, _, acol) in METRICS.items():
    d = d2[["Subject_ID", "Age", acol, "theta_PVS_avg", "theta_SCR_avg",
            "theta_SLF_avg"]].dropna().rename(columns={acol: "y"}).copy()
    for c in ("y", "Age", "theta_PVS_avg", "theta_SCR_avg", "theta_SLF_avg"):
        d[c + "_z"] = (d[c] - d[c].mean()) / d[c].std(ddof=1)
    raw = smf.ols("y_z ~ Age_z", d).fit(
        cov_type="cluster", cov_kwds={"groups": d["Subject_ID"]})
    adj = smf.ols("y_z ~ Age_z + theta_PVS_avg_z + theta_SCR_avg_z + theta_SLF_avg_z",
                  d).fit(cov_type="cluster", cov_kwds={"groups": d["Subject_ID"]})
    raw_b[name] = float(raw.params["Age_z"])
    adj_b[name] = float(adj.params["Age_z"])

fig = plt.figure(figsize=(6.9, 2.9))
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.36)

axA = fig.add_subplot(gs[0, 0])
sl, ic, r, p, se = stats.linregress(d2["Age"], d2["theta_SCR_avg"])
axA.scatter(d2["Age"], d2["theta_SCR_avg"], s=8, alpha=0.3,
            color=C_CLASSIC, edgecolors="none", zorder=2)
xa = np.linspace(d2["Age"].min(), d2["Age"].max(), 100)
axA.plot(xa, sl * xa + ic, color=C_CLASSIC, lw=2, zorder=4)
axA.set_xlabel("age (years)")
axA.set_ylabel(r"$\theta_{\mathrm{SCR}}$, SCR axis vs scanner $z$ (deg)")
axA.grid(True, color=C_GRID, lw=0.5, zorder=0)
axA.set_axisbelow(True)
axA.set_title("(a) scanner-anatomy angle tracks age", loc="left", color=C_INK)
axA.text(0.03, 0.95, f"$r$ = {r:.3f}\n{sl:.3f} deg/year",
         transform=axA.transAxes, va="top", ha="left", color=C_INK)

axB = fig.add_subplot(gs[0, 1])
names = list(METRICS)
ys = np.arange(len(names))[::-1]
for y, name in zip(ys, names):
    axB.plot([raw_b[name], adj_b[name]], [y, y], color=C_MUTED, lw=1.2,
             zorder=2, solid_capstyle="round")
    axB.plot([raw_b[name]], [y], "o", ms=6, color=C_CLASSIC, mec="white",
             mew=1.0, zorder=4)
    axB.plot([adj_b[name]], [y], "o", ms=6, color=C_REFINED, mec="white",
             mew=1.0, zorder=4)
    pct = 100 * (adj_b[name] - raw_b[name]) / abs(raw_b[name])
    axB.annotate(f"{pct:+.0f}%", (adj_b[name], y), textcoords="offset points",
                 xytext=(8, 0), va="center", ha="left", fontsize=8, color=C_MUTED)

axB.plot([], [], "o", ms=6, color=C_CLASSIC, label="unadjusted")
axB.plot([], [], "o", ms=6, color=C_REFINED, label="orientation-adjusted")
axB.set_yticks(ys)
axB.set_yticklabels(names, color=C_INK)
axB.set_ylim(-0.6, len(names) - 0.4)
lo_x = min(min(raw_b.values()), min(adj_b.values()))
hi_x = max(max(raw_b.values()), max(adj_b.values()))
pad = 0.12 * (hi_x - lo_x)
axB.set_xlim(lo_x - pad, hi_x + 3.2 * pad)
axB.set_xlabel("standardised age coefficient")
axB.grid(True, axis="x", color=C_GRID, lw=0.5, zorder=0)
axB.set_axisbelow(True)
axB.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.06),
           ncol=2, handlelength=1.0, columnspacing=1.4, borderaxespad=0.0)
axB.set_title("(b) adjusting for orientation", loc="left", color=C_INK, pad=22)

fig.savefig(HERE.parent / "fig_orientation_confound.png", bbox_inches="tight", dpi=300)
plt.close(fig)
print("wrote fig_orientation_confound.png")
for name in names:
    print(f"  {name:<9s} raw {raw_b[name]:+.4f} -> adj {adj_b[name]:+.4f} "
          f"({100*(adj_b[name]-raw_b[name])/abs(raw_b[name]):+.1f}%)")
