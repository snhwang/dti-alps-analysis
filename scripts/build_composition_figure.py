"""
Figure: what is inside the region, and why it matters.

  (a) Distribution of the off-tract voxel fraction for the projection and
      association regions, both cohorts. The projection region is clean; the
      association region carries about a fifth off-tract in every session.
  (b) That fraction against the index value. Left-right oriented voxels carry
      high x-diffusivity, and x-diffusivity is the numerator of the ratio, so
      the association is expected and is present for both variants.
  (c) Three confounds side by side, as percentage of the standardised age
      coefficient absorbed. Region volume and composition describe what is
      inside the region and are absorbed about equally by both indices, because
      no choice of measurement axis can make an index invariant to its own
      contents. Head position is a property of the measurement frame, and is
      the only one the correction addresses.

Matches the house style of build_revision_figures.py: same colour pair, same
rcParams, same 6.9 inch print width, every series directly labelled so identity
never depends on colour alone.

Output: fig_roi_composition.png
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
C_CLASSIC = "#1f77b4"
C_REFINED = "#ff7f0e"
C_INK = "#222222"
C_MUTED = "#6b6b6b"
C_GRID = "#dddddd"

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.edgecolor": "#999999", "axes.linewidth": 0.8, "figure.dpi": 300,
})

H = pd.read_csv(HERE / "roi_placement_quality_hcpa_b1500.csv")
D = pd.read_csv(HERE / "roi_placement_quality_dlbs_all.csv")

fig = plt.figure(figsize=(6.9, 3.1))
gs = fig.add_gridspec(1, 3, wspace=0.52, left=0.07, right=0.985, bottom=0.22, top=0.86)

# (a) distribution of off-tract fraction, by region and cohort
ax = fig.add_subplot(gs[0, 0])
data = [H.scr_off_tract * 100, D.scr_off_tract * 100,
        H.slf_off_tract * 100, D.slf_off_tract * 100]
pos = [0.9, 1.5, 2.9, 3.5]
bp = ax.boxplot(data, positions=pos, widths=0.44, showfliers=False,
                patch_artist=True, medianprops=dict(color=C_INK, linewidth=1.2))
# Neutral greys here on purpose: in panels (b) and (c) blue and orange mean
# Classic and Refined, so reusing them for cohort would give one colour two
# meanings in a single figure.
C_H, C_D = "#4a4a4a", "#b0b0b0"
for patch, col in zip(bp["boxes"], [C_H, C_D, C_H, C_D]):
    patch.set_facecolor(col); patch.set_alpha(0.75); patch.set_edgecolor("#888888")
ax.set_xticks([1.2, 3.2]); ax.set_xticklabels(["Projection\n(SCR)", "Association\n(SLF)"])
ax.set_ylabel("Off-tract voxels in region (%)")
ax.set_title("(a) Region composition", loc="left", pad=6)
ax.set_ylim(-1, 46)
ax.grid(axis="y", color=C_GRID, linewidth=0.6)
ax.set_axisbelow(True)
ax.legend(handles=[Patch(facecolor=C_H, alpha=0.75, label="HCP-A"),
                   Patch(facecolor=C_D, alpha=0.75, label="DLBS")],
          frameon=False, loc="upper left", handlelength=1.1, borderaxespad=0.2)

# (b) off-tract fraction against the index, HCP-A
ax = fig.add_subplot(gs[0, 1])
for k, (col, c, lab) in enumerate((("classic", C_CLASSIC, "Classic"),
                                   ("refined_slab", C_REFINED, "Refined"))):
    s = H[["slf_off_tract", col]].dropna()
    x, y = s.slf_off_tract * 100, s[col]
    ax.scatter(x, y, s=2.0, alpha=0.13, color=c, linewidths=0, rasterized=True)
    lr = stats.linregress(x, y)
    xx = np.linspace(x.quantile(.005), x.quantile(.995), 60)
    ax.plot(xx, lr.intercept + lr.slope * xx, color=c, linewidth=1.7)
    ax.text(0.03, 0.97 - 0.09 * k, f"{lab}  r = {np.corrcoef(x, y)[0,1]:+.2f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=7.5, color=c)
ax.set_xlabel("Off-tract voxels in\nassociation region (%)")
ax.set_ylabel("ALPS index")
ax.set_title("(b) Composition predicts value", loc="left", pad=6)
ax.set_xlim(2, 52)
ax.grid(color=C_GRID, linewidth=0.6); ax.set_axisbelow(True)

# (c) the three geometric confounds
ax = fig.add_subplot(gs[0, 2])
labels = ["Head\nposition", "Region\nvolume", "Compo-\nsition"]
classic_v = [45.0, 34.0, 18.3]
refined_v = [20.3, 33.4, 18.1]
xi = np.arange(3); w = 0.36
ax.bar(xi - w / 2, classic_v, w, color=C_CLASSIC, alpha=0.85, label="Classic")
ax.bar(xi + w / 2, refined_v, w, color=C_REFINED, alpha=0.85, label="Refined")
for i, (a, b) in enumerate(zip(classic_v, refined_v)):
    ax.annotate(f"{a:.0f}", xy=(i - w / 2, a), xytext=(0, 2), textcoords="offset points",
                ha="center", fontsize=7.5, color=C_CLASSIC)
    ax.annotate(f"{b:.0f}", xy=(i + w / 2, b), xytext=(0, 2), textcoords="offset points",
                ha="center", fontsize=7.5, color=C_REFINED)
ax.set_xticks(xi); ax.set_xticklabels(labels, fontsize=7.5)
ax.set_ylabel("Age coefficient absorbed (%)")
ax.set_title("(c) Only pose is asymmetric", loc="left", pad=6)
ax.set_ylim(0, 58)
ax.legend(frameon=False, loc="upper right", handlelength=1.1, borderaxespad=0.2)
ax.grid(axis="y", color=C_GRID, linewidth=0.6); ax.set_axisbelow(True)

out = HERE.parent / "fig_roi_composition.png"
fig.savefig(out, bbox_inches="tight", dpi=300)
fig.savefig(str(out).replace(".png", ".pdf"), bbox_inches="tight")
print(f"wrote {out}")
