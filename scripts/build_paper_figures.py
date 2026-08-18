"""
Figures for the revised manuscript, from the committed result tables.

Figure A  Rotation, accuracy and its cost
  (a) absolute error against the unrotated reference, by rotation magnitude,
      for every ALPS variant, with the crossover marked
  (b) error by rotation axis at 15 degrees, showing pitch dominates

Figure B  Consequences for inference
  (a) apparent group difference produced purely by a positioning offset, with
      the range of reported disease effects shaded
  (b) share of single-patient reads crossing a normative threshold

Sized at final print width so \\includegraphics[width=\\linewidth] is 1:1.
Classic-blue against refined-orange, validated as colourblind-safe (worst
adjacent dE 24.6 protan); all series are directly labelled so identity never
rests on colour alone.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE.parent

C_CLASSIC = "#1f77b4"
C_REFINED = "#ff7f0e"
C_INK = "#222222"
C_MUTED = "#6b6b6b"
C_GRID = "#dddddd"
C_BAND = "#c8c8c8"

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.edgecolor": "#999999", "axes.linewidth": 0.8, "figure.dpi": 300,
})

STYLE = {
    "classic":  (C_CLASSIC, "-", 2.2),
    "refined":  (C_REFINED, "-", 2.2),
    "ALPS-PAS": (C_MUTED,   ":", 1.6),
}

# ---- Figure A --------------------------------------------------------------

acc = pd.read_csv(HERE / "rotation_slab_accuracy.csv")
per = pd.read_csv(HERE / "rotation_slab_peraxis.csv")

fig = plt.figure(figsize=(6.9, 2.9))
gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.30)

ax = fig.add_subplot(gs[0, 0])
for m, (c, ls, lw) in STYLE.items():
    if m in acc.columns:
        ax.plot(acc["sigma"], acc[m], ls, color=c, lw=lw, label=m)
cross = np.interp(0.0, (acc["classic"] - acc["refined"]).values, acc["sigma"].values)
ax.axvline(cross, color=C_INK, lw=0.9, ls="--", zorder=1)
ax.annotate(f"crossover {cross:.1f}$\\degree$", (cross, ax.get_ylim()[1] * 0.94),
            xytext=(4, 0), textcoords="offset points", fontsize=8, color=C_INK)
ax.set_xlabel("imposed rotation, per-axis SD (degrees)")
ax.set_ylabel("absolute error vs unrotated reference (%)")
ax.grid(True, color=C_GRID, lw=0.5); ax.set_axisbelow(True)
ax.legend(frameon=False, loc="upper left", handlelength=2.0, borderaxespad=0.2)
ax.set_title("(a) accuracy against known rotation", loc="left", color=C_INK)

ax2 = fig.add_subplot(gs[0, 1])
axes_lbl = per["axis"].tolist()
xs = np.arange(len(axes_lbl))
w = 0.36
ax2.bar(xs - w / 2, per["classic"], w, color=C_CLASSIC, label="classic")
ax2.bar(xs + w / 2, per["refined"], w, color=C_REFINED, label="refined")
ax2.set_xticks(xs); ax2.set_xticklabels([a.split()[0] for a in axes_lbl])
ax2.set_ylabel("absolute error (%)")
ax2.set_xlabel("rotation axis, 15$\\degree$")
ax2.grid(True, axis="y", color=C_GRID, lw=0.5); ax2.set_axisbelow(True)
ax2.legend(frameon=False, loc="upper right", borderaxespad=0.2)
ax2.set_title("(b) which axis matters", loc="left", color=C_INK)

fig.savefig(OUT / "fig_rotation_accuracy.png", bbox_inches="tight", dpi=300)
plt.close(fig)
print("wrote fig_rotation_accuracy.png  (crossover %.1f deg)" % cross)

# ---- Figure B --------------------------------------------------------------

grp = pd.read_csv(HERE / "rotation_slab_group.csv")
sp = pd.read_csv(HERE / "single_patient_impact.csv")

fig = plt.figure(figsize=(6.9, 2.9))
gs = fig.add_gridspec(1, 2, wspace=0.32)

axA = fig.add_subplot(gs[0, 0])
axA.axhspan(-20, -5, color=C_BAND, alpha=0.45, lw=0, zorder=0)
axA.annotate("reported disease effects, 5-20%", (grp["tilt_deg"].min(), -12.5),
             xytext=(2, 0), textcoords="offset points", fontsize=7.5, color=C_INK,
             va="center")
axA.plot(grp["tilt_deg"], grp["classic"], "-o", color=C_CLASSIC, lw=2.2, ms=4,
         label="classic")
axA.plot(grp["tilt_deg"], grp["refined"], "-o", color=C_REFINED, lw=2.2, ms=4,
         label="refined")
axA.axhline(0, color=C_INK, lw=0.9, ls="--", zorder=1)
axA.set_xlabel("positioning difference between groups (degrees pitch)")
axA.set_ylabel("apparent group difference (%)")
axA.grid(True, color=C_GRID, lw=0.5); axA.set_axisbelow(True)
axA.legend(frameon=False, loc="lower left", borderaxespad=0.2)
axA.set_title("(a) a difference that is not there", loc="left", color=C_INK)

axB = fig.add_subplot(gs[0, 1])
axB.plot(sp["tilt_deg"], sp["classic_flip_pct"], "-o", color=C_CLASSIC, lw=2.2,
         ms=4, label="classic")
axB.plot(sp["tilt_deg"], sp["refined_flip_pct"], "-o", color=C_REFINED, lw=2.2,
         ms=4, label="refined")
axB.set_xlabel("head tilt (degrees pitch)")
axB.set_ylabel("reads crossing the 10th percentile (%)")
axB.grid(True, color=C_GRID, lw=0.5); axB.set_axisbelow(True)
axB.legend(frameon=False, loc="upper left", borderaxespad=0.2)
axB.set_title("(b) single-patient misclassification", loc="left", color=C_INK)

fig.savefig(OUT / "fig_inference_impact.png", bbox_inches="tight", dpi=300)
plt.close(fig)
print("wrote fig_inference_impact.png")
