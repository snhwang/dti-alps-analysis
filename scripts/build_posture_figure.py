"""
Figure: head position covaries with age, and a fixed-axis index inherits it.

This is the paper's principal finding and had no figure.

  (a) Head pitch against age in both cohorts. In obliquely acquired DLBS pitch
      rises with age; in anatomically aligned HCP-A, where preprocessing has
      removed head position, it does not. Same index, comparable age range,
      identical analysis. The confound appears only where the confounding
      variable survived into the analysed data.
  (b) Axis profile in the oblique cohort. Pitch shows the strongest association
      with age, roll none, yaw an intermediate one. A general tendency to move
      would raise all three about equally, which is not what is seen, but the
      elevated yaw means the pattern is not purely sagittal.
  (c) Standardised age coefficient before and after adjusting for head pose, by
      variant and cohort. The classic index loses 45% in DLBS and nothing in
      HCP-A, because in HCP-A there is nothing to lose.

Output: fig_head_position.png
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
C_CLASSIC = "#1f77b4"
C_REFINED = "#ff7f0e"
C_MUTED = "#6b6b6b"
C_GRID = "#dddddd"
C_OBL, C_ALN = "#4a4a4a", "#b0b0b0"

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "axes.edgecolor": "#999999", "axes.linewidth": 0.8, "figure.dpi": 300,
})


def one_per_subject(head, alps):
    h, a = pd.read_csv(head), pd.read_csv(alps)
    for d in (h, a):
        d["Subject_ID"] = d.Subject_ID.astype(str)
        d["Visit"] = d.Visit.astype(str)
    m = h.merge(a, on=["Subject_ID", "Visit"], how="inner")
    return (m.sort_values(["Subject_ID", "Visit"])
              .groupby("Subject_ID").first().reset_index()
              .dropna(subset=["Age", "pitch", "classic"]))


D = one_per_subject(HERE / "head_rotation_dlbs.csv", HERE / "measured_pvs_axis_dlbs.csv")
H = one_per_subject(HERE / "head_rotation_hcpa.csv", HERE / "measured_pvs_axis_hcpa_b1500_all.csv")

fig = plt.figure(figsize=(7.0, 2.9))
gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 0.9, 1.15], wspace=0.42,
                      left=0.07, right=0.985, bottom=0.19, top=0.88)

# (a) pitch against age, both cohorts
ax = fig.add_subplot(gs[0, 0])
for d, c, lab in ((D, C_OBL, "DLBS (oblique)"), (H, C_ALN, "HCP-A (aligned)")):
    x, y = d.Age.to_numpy(float), d.pitch.abs().to_numpy(float)
    ax.scatter(x, y, s=5, alpha=0.35, color=c, linewidths=0)
    lr = stats.linregress(x, y)
    xx = np.linspace(x.min(), x.max(), 50)
    ax.plot(xx, lr.intercept + lr.slope * xx, color=c, linewidth=2.0)
    r = np.corrcoef(x, y)[0, 1]
    ax.text(0.03, 0.97 if c == C_OBL else 0.88, f"{lab}  r = {r:+.3f}",
            transform=ax.transAxes, fontsize=7.5,
            color=c if c == C_OBL else "#7a7a7a", va="top")
ax.set_xlabel("Age (years)")
ax.set_ylabel("Absolute head pitch (degrees)")
ax.set_title("(a) Pitch rises with age", loc="left", pad=6)
ax.set_ylim(0, 26)
ax.grid(color=C_GRID, linewidth=0.6); ax.set_axisbelow(True)

# (b) axis specificity in the oblique cohort
ax = fig.add_subplot(gs[0, 1])
axes = ["pitch", "roll", "yaw"]
rs = [np.corrcoef(D.Age, D[c].abs())[0, 1] for c in axes]
# pitch highlighted because it is the axis the index is most sensitive to;
# yaw is elevated too and is not claimed to be negligible
cols = [C_CLASSIC, C_MUTED, C_MUTED]
ax.bar(range(3), rs, 0.6, color=cols, alpha=0.85)
for i, r in enumerate(rs):
    ax.annotate(f"{r:+.3f}", xy=(i, r), xytext=(0, 3 if r > 0 else -11),
                textcoords="offset points", ha="center", fontsize=7.5,
                color=cols[i])
ax.axhline(0, color="#888888", linewidth=0.8)
ax.set_xticks(range(3)); ax.set_xticklabels(["Pitch", "Roll", "Yaw"])
ax.set_ylabel("Correlation with age")
ax.set_title("(b) Pitch strongest, roll absent", loc="left", pad=6)
ax.set_ylim(-0.05, 0.45)
ax.grid(axis="y", color=C_GRID, linewidth=0.6); ax.set_axisbelow(True)

# (c) age coefficient before and after adjusting for head pose
ax = fig.add_subplot(gs[0, 2])


def beta(d, col, adjust):
    z = lambda v: (np.asarray(v, float) - np.mean(v)) / np.std(v, ddof=1)
    cols = [np.ones(len(d)), z(d.Age)]
    if adjust:
        cols += [z(d.pitch.abs()), z(d.total)]
    X = np.column_stack(cols)
    return float(np.linalg.lstsq(X, z(d[col]), rcond=None)[0][1])


groups = [("DLBS\nclassic", D, "classic", C_CLASSIC),
          ("DLBS\nrefined", D, "cross", C_REFINED),
          ("HCP-A\nclassic", H, "classic", C_CLASSIC),
          ("HCP-A\nrefined", H, "cross", C_REFINED)]
xi = np.arange(4); w = 0.36
for k, (lab, d, col, c) in enumerate(groups):
    b0, b1 = abs(beta(d, col, False)), abs(beta(d, col, True))
    ax.bar(k - w / 2, b0, w, color=c, alpha=0.9)
    ax.bar(k + w / 2, b1, w, color=c, alpha=0.35)
    drop = 100 * (1 - b1 / b0)
    ax.annotate(f"-{drop:.0f}%" if drop > 1 else "0%",
                xy=(k, max(b0, b1)), xytext=(0, 3), textcoords="offset points",
                ha="center", fontsize=7, color=c)
ax.set_xticks(xi); ax.set_xticklabels([g[0] for g in groups], fontsize=7)
ax.set_ylabel("|standardised age coefficient|")
ax.set_title("(c) Solid: raw. Faded: pose-adjusted", loc="left", pad=6)
ax.set_ylim(0, 0.78)
ax.grid(axis="y", color=C_GRID, linewidth=0.6); ax.set_axisbelow(True)

out = HERE.parent / "fig_head_position.png"
fig.savefig(out, bbox_inches="tight", dpi=300)
fig.savefig(str(out).replace(".png", ".pdf"), bbox_inches="tight")
print(f"wrote {out}")
