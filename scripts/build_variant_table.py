"""
Generate the variant-comparison table directly from the data files.

The manuscript describes several ways of evaluating the index and reports their
performance in scattered sentences. This consolidates them into one table, and
generates it from source so the numbers cannot drift from the analyses.

Columns are the endpoints on which the variants can actually be separated:
departure from the unrotated reference, and test-retest reliability and age
association in both cohorts. Writes LaTeX for direct inclusion.

Two conventions run through this table and they are not the same as the ones in
tbl:beyond. Reliability is ICC(1,1) over participants with two or more visits.
The age association is a Pearson correlation over every session, not one per
participant, which is why its values differ from the raw column of tbl:beyond.

The trigeminal column was dropped with that cohort. ALPS-PAS and the per-voxel
form used to be dashes here because they were computed only in that cohort;
they now come from comparators_*.csv, measured in the same regions as the
variants beside them. LD-ALPS comes from its own output, since it places its
own regions, and keeps a dash for departure because it was never put through
the rotation study.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from estimator_variants import variance_components          # noqa: E402

# manuscript name -> index column, rotation_slab_accuracy column
VARIANTS = [("Classic", "classic", "classic"),
            ("Refined (cross product)", "cross", "refined"),
            ("Refined+", None, "refined+"),
            ("Measured axis", "v2_slab", None),
            ("Voxelwise measured axis", "pv_perp", None),
            ("Anatomical axis", "anat_x", None),
            ("LD-ALPS", "LD-ALPS", None),
            ("ALPS-PAS", "ALPS-PAS", "ALPS-PAS"),
            ("Per-voxel", "per-voxel", "per-voxel")]

COHORTS = {"hcpa": "measured_pvs_axis_hcpa_b1500_all.csv",
           "dlbs": "measured_pvs_axis_dlbs.csv"}


def table(cohort: str) -> pd.DataFrame:
    """The index table with the comparators merged onto it."""
    d = pd.read_csv(HERE / COHORTS[cohort])
    d["Subject_ID"] = d.Subject_ID.astype(str)
    d["Visit"] = d.Visit.astype(str)
    c = pd.read_csv(HERE / f"comparators_{cohort}.csv")
    c["Subject_ID"] = c.Subject_ID.astype(str)
    c["Visit"] = c.Visit.astype(str)
    d = d.merge(c[["Subject_ID", "Visit", "ALPS-PAS", "per-voxel"]],
                on=["Subject_ID", "Visit"], how="left")
    ld = HERE / f"ld_alps_{cohort}.csv"
    if ld.exists():
        L = pd.read_csv(ld)[["Subject_ID", "Visit", "ALPS_overall"]].rename(
            columns={"ALPS_overall": "LD-ALPS"})
        L["Subject_ID"] = L.Subject_ID.astype(str)
        L["Visit"] = L.Visit.astype(str)
        d = d.merge(L, on=["Subject_ID", "Visit"], how="left")
    return d


def icc_and_age(d: pd.DataFrame, col: str | None):
    if col is None or col not in d.columns:
        return np.nan, np.nan
    lon = d[d.Subject_ID.isin(d.Subject_ID.value_counts()[lambda s: s >= 2].index)]
    s = lon.dropna(subset=[col])
    icc = variance_components(s, col)["icc"] if len(s) > 20 else np.nan
    ds = d.dropna(subset=[col, "Age"])
    r = stats.pearsonr(ds.Age, ds[col])[0] if len(ds) > 20 else np.nan
    return icc, r


rot = pd.read_csv(HERE / "rotation_slab_accuracy.csv")
H, D = table("hcpa"), table("dlbs")

rows = []
for name, col, rcol in VARIANTS:
    dep = f"{rot[rcol].iloc[0]:.2f}" if rcol and rcol in rot and name != "Classic" else "--"
    if name == "Classic":
        dep = "0 to 20.3"
    if name == "ALPS-PAS":
        dep = f"{rot[rcol].iloc[0]:.2f} to {rot[rcol].iloc[-1]:.2f}"
    # anat_x is exactly invariant, verified to machine precision in
    # anat_x_invariance.py, provided the registration is recomputed for the
    # rotated head so that R'x rotates with it.
    if name in ("Measured axis", "Voxelwise measured axis", "Anatomical axis"):
        dep = "flat"
    hi, ha = icc_and_age(H, col)
    di, da = icc_and_age(D, col)
    f = lambda v, d=3: "--" if (v is None or (isinstance(v, float) and np.isnan(v))) else f"{v:.{d}f}"
    rows.append((name, dep, f(hi), f(ha), f(di), f(da)))

print(f"{'variant':<24s} {'rot dep':>14s} {'HCP ICC':>8s} {'HCP age':>8s} "
      f"{'DLBS ICC':>9s} {'DLBS age':>9s}")
for r in rows:
    print(f"{r[0]:<24s} {r[1]:>14s} {r[2]:>8s} {r[3]:>8s} {r[4]:>9s} {r[5]:>9s}")

BS = chr(92)
tex = [BS + r"begin{table}[tb]",
       BS + r"caption{The ways of evaluating the index compared on the endpoints that "
       r"separate them. Departure is the constant offset from the unrotated classic value, "
       r"which is flat for every corrected variant and grows with rotation for classic. "
       r"Reliability is ICC(1,1) across visits, over participants with repeat sessions. The "
       r"age association is a Pearson correlation over every session, which is why it differs "
       r"from the one-session-per-participant values in Table~" + BS + r"ref{tbl:beyond}. The "
       r"voxelwise measured axis is the eigenvalue ratio $" + BS + r"lambda_2/" + BS +
       r"lambda_3$, for the reason given in Section~" + BS + r"ref{sec:measured-axis}. LD-ALPS "
       r"places its own regions rather than the ones used for every other row, so it is "
       r"comparable here, where each variant is measured against age within its own regions, "
       r"and not where a shared region is required. Dashes mark combinations not computed: "
       r"Refined+ was carried through the rotation analysis only, and LD-ALPS was never put "
       r"through it.}",
       BS + r"label{tbl:variants}",
       BS + r"begin{tabular*}{" + BS + r"tblwidth}{@{}LCCCCC@{}}",
       BS + r"toprule",
       (BS + r"textbf{Variant} & " + BS + r"textbf{Depart.} & " + BS + r"textbf{ICC} & "
        + BS + r"textbf{$r$ age} & " + BS + r"textbf{ICC} & " + BS + r"textbf{$r$ age} "
        + BS + BS),
       r" & (\%) & \multicolumn{2}{c}{HCP-A} & \multicolumn{2}{c}{DLBS} " + BS + BS,
       BS + r"midrule"]
for r in rows:
    tex.append(" & ".join(r) + " " + BS + BS)
tex += [BS + r"bottomrule", BS + r"end{tabular*}", BS + r"end{table}"]
(HERE / "variant_table.tex").write_text("\n".join(tex), encoding="utf-8")
print(f"\nwrote {HERE / 'variant_table.tex'}")
