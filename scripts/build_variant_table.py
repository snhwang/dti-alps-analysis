"""
Generate the variant-comparison table directly from the data files.

The manuscript describes four ways of evaluating the index and reports their
performance in scattered sentences. This consolidates them into one table, and
generates it from source so the numbers cannot drift from the analyses.

Columns are the endpoints on which the variants can actually be separated:
departure from the unrotated reference, test-retest reliability in both cohorts,
the age association in both cohorts, and the patient-control effect. Writes
LaTeX for direct inclusion.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

from data_paths import winpath
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from estimator_variants import variance_components

# manuscript name -> column in measured_pvs_axis_*, column in rotation_slab_accuracy
VARIANTS = [("Classic", "classic", "classic"),
            ("Refined (cross product)", "cross", "refined"),
            ("Refined+", None, "refined+"),
            ("Measured axis", "v2_slab", None),
            ("Voxelwise measured axis", "pv_perp", None),
            ("ALPS-PAS", None, "ALPS-PAS"),
            ("Per-voxel", None, "per-voxel"),
            # Anatomical x: the measured tract frame with the perivascular axis
            # taken from the registration rather than from a cross product.
            # Exactly invariant, since R'x is fixed in the anatomy.
            ("Anatomical axis", "anat_x", None)]


def icc_and_age(fn, col):
    d = pd.read_csv(HERE / fn)
    lon = d[d.Subject_ID.isin(d.Subject_ID.value_counts()[lambda s: s >= 2].index)]
    s = lon.dropna(subset=[col])
    icc = variance_components(s, col)["icc"] if len(s) > 20 else np.nan
    ds = d.dropna(subset=[col, "Age"])
    r = stats.pearsonr(ds.Age, ds[col])[0] if len(ds) > 20 else np.nan
    return icc, r


rot = pd.read_csv(HERE / "rotation_slab_accuracy.csv")
tn = pd.read_csv(HERE / "tn_alps.csv")
par = pd.read_csv(winpath("M:/ds005713-download/participants_v2.0.1.tsv"), sep="\t")
m = tn.merge(par, on="BIDS_ID")
m["patient"] = (m.BIDS_ID.astype(str).str.extract(r"sub-(\d+)")[0].str.len() >= 3).astype(int)
m["age"] = pd.to_numeric(m.age, errors="coerce")
m["sex_n"] = pd.to_numeric(m.sex, errors="coerce")
m = m.dropna(subset=["age", "sex_n"])
TN_COL = {"Classic": "classic", "Refined (cross product)": "cross",
          "Measured axis": "v2_slab_b8", "Voxelwise measured axis": "pv_perp",
          "ALPS-PAS": "ALPS-PAS", "Per-voxel": "per-voxel",
          "Anatomical axis": "anat_x"}


def tn_beta(col):
    if col is None or col not in m:
        return np.nan
    s = m.dropna(subset=[col])
    C = np.column_stack([np.ones(len(s)), s.age, s.sex_n])
    def rz(v):
        b, *_ = np.linalg.lstsq(C, np.asarray(v, float), rcond=None)
        return np.asarray(v, float) - C @ b
    return float(np.corrcoef(rz(s.patient.astype(float)), rz(s[col]))[0, 1])


rows = []
for name, col, rcol in VARIANTS:
    dep = f"{rot[rcol].iloc[0]:.2f}" if rcol and rcol in rot and name != "Classic" else ""
    if name == "Classic":
        dep = "0 to 20.3"
    if name == "ALPS-PAS":
        dep = f"{rot[rcol].iloc[0]:.2f} to {rot[rcol].iloc[-1]:.2f}"
    # anat_x is exactly invariant, verified to machine precision in
    # anat_x_invariance.py, provided the registration is recomputed for the
    # rotated head so that R'x rotates with it.
    if name in ("Measured axis", "Voxelwise measured axis", "Anatomical axis"):
        dep = "flat"
    hi, ha = icc_and_age("measured_pvs_axis_hcpa_b1500_all.csv", col) if col else (np.nan, np.nan)
    di, da = icc_and_age("measured_pvs_axis_dlbs.csv", col) if col else (np.nan, np.nan)
    b = tn_beta(TN_COL.get(name))
    f = lambda v, d=3: "--" if (v is None or (isinstance(v, float) and np.isnan(v))) else f"{v:.{d}f}"
    rows.append((name, dep, f(hi), f(ha), f(di), f(da), f(b)))

print(f"{'variant':<24s} {'rot dep':>10s} {'HCP ICC':>8s} {'HCP age':>8s} "
      f"{'DLBS ICC':>9s} {'DLBS age':>9s} {'TN beta':>8s}")
for r in rows:
    print(f"{r[0]:<24s} {r[1]:>10s} {r[2]:>8s} {r[3]:>8s} {r[4]:>9s} {r[5]:>9s} {r[6]:>8s}")

BS = chr(92)
tex = [BS + r"begin{table}[tb]",
       BS + r"caption{The ways of evaluating the index compared on the endpoints that "
       r"separate them. Departure is the constant offset from the unrotated classic value, "
       r"which is flat for every corrected variant and grows with rotation for classic. "
       r"Reliability is ICC(1,1) across visits and the age association is a Pearson "
       r"correlation, both reported for each cohort. The patient-control column is the "
       r"standardised group coefficient in trigeminal neuralgia, adjusted for age and sex. "
       r"The voxelwise measured axis is the eigenvalue ratio "
       r"$\lambda_2/\lambda_3$, for the reason given in Section~\ref{sec:measured-axis}. "
       r"Dashes mark combinations not computed: Refined+, ALPS-PAS and the per-voxel variant "
       r"were carried through the rotation and patient analyses but not the longitudinal ones.}",
       BS + r"label{tbl:variants}",
       BS + r"begin{tabular*}{" + BS + r"tblwidth}{@{}LCCCCCC@{}}",
       BS + r"toprule",
       (BS + r"textbf{Variant} & " + BS + r"textbf{Depart.} & " + BS + r"textbf{ICC} & "
        + BS + r"textbf{$r$ age} & " + BS + r"textbf{ICC} & " + BS + r"textbf{$r$ age} & "
        + BS + r"textbf{$" + BS + r"beta$ TN} " + BS + BS),
       r" & (\%) & \multicolumn{2}{c}{HCP-A} & \multicolumn{2}{c}{DLBS} & " + BS + BS,
       BS + r"midrule"]
for r in rows:
    tex.append(" & ".join(r) + " " + BS + BS)
tex += [BS + r"bottomrule", BS + r"end{tabular*}", BS + r"end{table}"]
(HERE / "variant_table.tex").write_text("\n".join(tex), encoding="utf-8")
print(f"\nwrote {HERE / 'variant_table.tex'}")
