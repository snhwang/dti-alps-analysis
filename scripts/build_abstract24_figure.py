"""
Figure for abstract 24: the tolerance curve and the rotation that actually occurs.

  (a) Mean absolute error against imposed pitch, for the conventional index and
      each orientation-corrected variant, with the 1/2/5/10% tolerance levels
      marked. The observed DLBS pitch distribution is drawn along the axis, so
      the two halves of the argument appear in one place: the cohort sits where
      the classic curve is already climbing.
  (b) Observed head rotation by axis, DLBS against HCP-A. Rotation is almost
      entirely pitch, which is the axis panel (a) shows is worst.

Output: abstract_compendium/figures/abstract24_alps_rotation_tolerance.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FIGDIR = HERE.parent.parent / "abstract_compendium" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

C_CLASSIC = "#1f77b4"
C_REF = "#ff7f0e"
C_MUTED = "#6b6b6b"
C_GRID = "#dddddd"
plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "axes.edgecolor": "#999999", "axes.linewidth": 0.8, "figure.dpi": 300,
})

cur = pd.read_csv(HERE / "rotation_tolerance_curves.csv")
p = cur[cur["mode"] == "pitch (x)"].sort_values("sigma")
grp = pd.read_csv(HERE / "rotation_slab_group.csv")
dl = pd.read_csv(HERE / "head_rotation_dlbs.csv")
hc = pd.read_csv(HERE / "head_rotation_hcpa.csv")

fig = plt.figure(figsize=(7.2, 2.9))
gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 1.05, 1.05], wspace=0.40,
                      left=0.065, right=0.985, bottom=0.19, top=0.89)

# (a) tolerance curve
ax = fig.add_subplot(gs[0, 0])
series = [("classic", C_CLASSIC, "-", 2.0, "Classic"),
          ("refined", C_REF, "-", 1.5, "Refined"),
          ("refined+", C_REF, "--", 1.2, "Refined+"),
          ("per-voxel", "#2ca02c", ":", 1.4, "Per-voxel"),
          ("ALPS-PAS", "#9467bd", "-.", 1.3, "ALPS-PAS")]
for col, c, ls, lw, lab in series:
    ax.plot(p.sigma, p[col], color=c, linestyle=ls, linewidth=lw, label=lab)
# Mark where the classic curve crosses each tolerance level, since the
# threshold is the quantity the abstract reports.
def cross(y, lvl):
    x = p.sigma.to_numpy(); y = np.asarray(y)
    for a in range(len(x) - 1):
        if y[a] <= lvl < y[a + 1]:
            f = (lvl - y[a]) / (y[a + 1] - y[a])
            return x[a] + f * (x[a + 1] - x[a])
    return None

for lvl in (2, 5, 10):
    xc = cross(p["classic"], lvl)
    if xc is None:
        continue
    ax.plot([xc, xc], [0, lvl], color=C_CLASSIC, linewidth=0.7, alpha=0.5, zorder=1)
    ax.plot([xc], [lvl], "o", color=C_CLASSIC, markersize=4, zorder=3)
    ax.annotate(f"{lvl}% at {xc:.1f}°", xy=(xc, lvl), xytext=(4, -1),
                textcoords="offset points", fontsize=7, color=C_CLASSIC, va="center")

# observed DLBS pitch as a density strip along the bottom
pit = dl.pitch.abs().dropna()
hist, edges = np.histogram(pit, bins=np.arange(0, 31, 1.5), density=True)
scale = 3.2 / max(hist.max(), 1e-9)
ax.bar(edges[:-1], hist * scale, width=np.diff(edges), align="edge",
       color=C_MUTED, alpha=0.22, zorder=0, linewidth=0)
ax.axvline(pit.median(), color=C_MUTED, linewidth=1.1, linestyle="--")
ax.annotate(f"DLBS median\npitch {pit.median():.1f}°", xy=(pit.median(), 17.4),
            xytext=(3, 0), textcoords="offset points", fontsize=7,
            color=C_MUTED, va="top")

ax.set_xlim(0, 30); ax.set_ylim(0, 18)
ax.set_xlabel("Imposed head pitch (degrees)")
ax.set_ylabel("Mean absolute error (%)")
ax.set_title("(a) Error vs imposed pitch", loc="left", pad=6)
ax.legend(frameon=False, loc="upper left", handlelength=1.6)
ax.grid(axis="y", color=C_GRID, linewidth=0.5); ax.set_axisbelow(True)

# (b) registration removes head pose but not anatomy
ax = fig.add_subplot(gs[0, 1])
dev = pd.read_csv(HERE / "roi_placement_quality_hcpa_b1500.csv")
sets = [(dl.total.dropna(), "#4a4a4a", "Head pose\n(DLBS)"),
        (hc.total.dropna(), "#b8b8b8", "Head pose\n(HCP-A)"),
        (pd.concat([dev.theta_scr, dev.theta_slf, dev.theta_pvs]).dropna(),
         C_REF, "Anatomy\n(HCP-A)")]
bp = ax.boxplot([s for s, _, _ in sets], positions=[1, 2, 3], widths=0.55,
                showfliers=False, patch_artist=True,
                medianprops=dict(color="#222222", linewidth=1.1))
for patch, (_, c, _) in zip(bp["boxes"], sets):
    patch.set_facecolor(c); patch.set_alpha(0.75); patch.set_edgecolor("#888888")
ax.set_xticks([1, 2, 3]); ax.set_xticklabels([l for _, _, l in sets], fontsize=6.5)
ax.set_ylabel("Angle (degrees)")
ax.set_title("(b) What registration leaves", loc="left", pad=6)
ax.set_ylim(-0.5, 28)
ax.annotate("removed by\nregistration", xy=(1.5, 27.5), fontsize=6.5,
            color=C_MUTED, ha="center", va="top")
ax.annotate("not removed", xy=(3, 27.5), fontsize=6.5, color=C_REF,
            ha="center", va="top")
ax.grid(axis="y", color=C_GRID, linewidth=0.5); ax.set_axisbelow(True)

# (c) the artefact this produces in a group comparison, where truth is zero
ax = fig.add_subplot(gs[0, 2])
ax.axhline(0, color="#888888", linewidth=0.8)
ax.plot(grp.tilt_deg, grp.classic, "-o", color=C_CLASSIC, markersize=3.5,
        linewidth=1.8, label="Classic")
ax.plot(grp.tilt_deg, grp.refined, "-s", color=C_REF, markersize=3.5,
        linewidth=1.6, label="Corrected (all)")
ax.fill_between([0, 21], -20, -5, color="#d62728", alpha=0.07, zorder=0)
ax.annotate("reported disease\neffect range", xy=(20.3, -16.5), fontsize=6.5,
            color="#a03030", ha="right", va="center")
ax.annotate(f"{grp.classic.iloc[-1]:.1f}%", xy=(20, grp.classic.iloc[-1]),
            xytext=(-4, 6), textcoords="offset points", fontsize=7,
            color=C_CLASSIC, ha="right")
ax.annotate("exactly 0", xy=(20, 0), xytext=(-4, 4), textcoords="offset points",
            fontsize=7, color=C_REF, ha="right")
ax.set_xlim(0, 21); ax.set_ylim(-20, 3)
ax.set_xlabel("Positioning offset between groups (°)")
ax.set_ylabel("Apparent group difference (%)")
ax.set_title("(c) Artefact, true difference 0", loc="left", pad=6)
ax.legend(frameon=False, loc="lower left", handlelength=1.4, bbox_to_anchor=(-0.02, -0.02))
ax.grid(axis="y", color=C_GRID, linewidth=0.5); ax.set_axisbelow(True)

out = FIGDIR / "abstract24_alps_rotation_tolerance.png"
fig.savefig(out, bbox_inches="tight", dpi=300)
fig.savefig(str(out).replace(".png", ".pdf"), bbox_inches="tight")
print(f"wrote {out}")
