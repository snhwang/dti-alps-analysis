"""How much do tract directions vary between individuals after gross alignment?

Registering everyone to a common space aligns gross morphology, but it is driven
by a T1 that carries no directional information about white matter. If the
individual tract direction varied only trivially between people, a template axis
would serve for everyone and the case for measuring the direction per participant
would be weak.

HCP-A answers this, because its diffusion data is anatomically aligned during
preprocessing: residual head rotation is small, so what remains of the angle
between a participant's tract direction and the scanner axis is that
participant's anatomy rather than their posture. Two things are asked of it.
How wide is the between-participant spread, and is it a stable trait of the
person or session noise? The second is settled by the repeat visits: a direction
that reproduces across separately acquired and separately processed sessions is
anatomy.

Writes tract_direction_variation.csv and fig_tract_direction.pdf/.png.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import atomic_io  # noqa: F401  writes become atomic on import
from scipy import stats

HERE = Path(__file__).resolve().parent
FIGS = HERE.parent   # figures live beside the manuscript

ANGLES = [("theta_scr", "Projection (SCR) vs scanner $z$"),
          ("theta_slf", "Association (SLF) vs scanner $y$"),
          ("theta_pvs", "Perivascular axis vs scanner $x$")]


def icc11(d, col):
    """ICC(1,1) from an unbalanced one-way random-effects ANOVA."""
    d = d.dropna(subset=[col])
    ni = d.groupby("Subject_ID")[col].size()
    d = d[d.Subject_ID.isin(ni[ni >= 2].index)]
    ni = d.groupby("Subject_ID")[col].size()
    mi = d.groupby("Subject_ID")[col].mean()
    a, N, grand = len(ni), int(ni.sum()), d[col].mean()
    msb = float((ni * (mi - grand) ** 2).sum() / (a - 1))
    msw = float(sum(((d[d.Subject_ID == s][col] - mi[s]) ** 2).sum()
                    for s in ni.index) / (N - a))
    n0 = (N - (ni ** 2).sum() / N) / (a - 1)
    return (msb - msw) / (msb + (n0 - 1) * msw), a


rows = []
data = {}
for tag, fn in (("HCP-A", "roi_placement_quality_hcpa_b1500.csv"),
                ("DLBS", "roi_placement_quality_dlbs_all.csv")):
    d = pd.read_csv(HERE / fn)
    data[tag] = d
    for col, _ in ANGLES:
        v = d[col].dropna()
        icc, nsub = icc11(d, col)
        rows.append(dict(cohort=tag, angle=col, n=len(v),
                         median=v.median(),
                         q1=np.percentile(v, 25), q3=np.percentile(v, 75),
                         p5=np.percentile(v, 5), p95=np.percentile(v, 95),
                         icc=icc, n_repeat=nsub))

out = pd.DataFrame(rows)
out.to_csv(HERE / "tract_direction_variation.csv", index=False)
print(out.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# Is the spread posture that alignment missed, or anatomy? Correlate the angle
# with each session's residual head rotation.
try:
    hr = pd.read_csv(HERE / "head_rotation_hcpa.csv")
    key = [c for c in hr.columns if c.lower() in ("total", "angle", "rotation",
                                                  "total_deg", "rot_deg")]
    d = data["HCP-A"].merge(hr, on=["Subject_ID", "Visit"], how="inner")
    print(f"\nresidual-rotation merge: {len(d)} sessions, "
          f"rotation columns {list(hr.columns)}")
    if key:
        for col, _ in ANGLES:
            m = d.dropna(subset=[col, key[0]])
            r, p = stats.pearsonr(m[col], m[key[0]])
            print(f"  {col} vs residual rotation: r={r:+.3f} p={p:.3g} n={len(m)}")
except FileNotFoundError:
    print("\nno head_rotation_hcpa.csv")

# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=(7.2, 4.6))
gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.95], hspace=0.55, wspace=0.30)
COL = {"HCP-A": "#1f4e79", "DLBS": "#c0504d"}

for j, (col, title) in enumerate(ANGLES):
    ax = fig.add_subplot(gs[0, j])
    bins = np.arange(0, 45, 1.5)
    for tag in ("HCP-A", "DLBS"):
        v = data[tag][col].dropna()
        ax.hist(v, bins=bins, density=True, histtype="step", lw=1.4,
                color=COL[tag], label=f"{tag} ($n={len(v)}$)")
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("deviation (deg)", fontsize=8)
    ax.set_xlim(0, 42)
    ax.tick_params(labelsize=7)
    if j == 0:
        ax.set_ylabel("density", fontsize=8)
        ax.legend(fontsize=6.5, frameon=False)

# repeat visits: the same angle measured at two separate sessions
d = data["HCP-A"]
for j, (col, title) in enumerate(ANGLES):
    ax = fig.add_subplot(gs[1, j])
    g = d.dropna(subset=[col]).sort_values(["Subject_ID", "Visit"])
    pairs = g.groupby("Subject_ID")[col].agg(list)
    pairs = pairs[pairs.apply(len) >= 2]
    x = np.array([p[0] for p in pairs])
    y = np.array([p[1] for p in pairs])
    ax.scatter(x, y, s=4, alpha=0.30, color=COL["HCP-A"], edgecolors="none")
    lim = (0, max(x.max(), y.max()) * 1.05)
    ax.plot(lim, lim, color="0.4", lw=0.8, ls="--")
    icc = float(out[(out.cohort == "HCP-A") & (out.angle == col)].icc.iloc[0])
    ax.set_title(f"repeat visits, ICC $={icc:.3f}$", fontsize=8)
    ax.set_xlabel("visit 1 (deg)", fontsize=8)
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)
    if j == 0:
        ax.set_ylabel("visit 2 (deg)", fontsize=8)

# Is the departure an age effect, or a trait?
#
# It reproduces within a participant at ICC ~0.92, so it is stable. Stability
# alone does not settle whether it also drifts with age. If it did, template
# reorientation would leave an age-correlated confound behind, since it
# evaluates fixed axes in a common space and so cannot remove an anatomical
# term at all. It does not. The perivascular axis's departure from scanner x is
# uncorrelated with age in both cohorts, one session per participant.
#
# That null is what makes the head-to-head of Section 3.8 interpretable. The
# anatomical term inflates between-participant variance without biasing a group
# age slope, which is why reorientation and the closed-form correction come out
# indistinguishable on the age association despite removing different things.
rows = []
for name, d in data.items():
    g = (d.dropna(subset=["theta_pvs", "Age"]).sort_values(["Subject_ID", "Visit"])
          .groupby("Subject_ID").first().reset_index())
    r, pv = stats.pearsonr(g.theta_pvs, g.Age)
    rows.append(dict(cohort=name, n=len(g), median=float(g.theta_pvs.median()),
                     r_age=float(r), p_age=float(pv)))
    print(f"  {name:8s} n={len(g):4d}  median {g.theta_pvs.median():5.2f} deg"
          f"   vs age r = {r:+.3f}  p = {pv:.3g}")
pd.DataFrame(rows).to_csv(HERE / "tract_direction_age.csv", index=False)

for ext in ("pdf", "png"):
    fig.savefig(FIGS / f"fig_tract_direction.{ext}", dpi=300, bbox_inches="tight")
print(f"\nwrote {FIGS / 'fig_tract_direction.pdf'}")
