"""
Cardiometabolic parameters, including the ones stored as text.

The phenotype sweep took only numeric columns, which silently dropped four
cardiometabolic variables held as strings: sitting and standing blood pressure
as "systolic/diastolic", and HbA1c and insulin as quoted numbers. Blood pressure
is the most consequential omission, since hypertension is the vascular exposure
most consistently linked to white matter damage and to perivascular space
burden, so it is the cardiometabolic variable a reader would look for first.

Parsed here and tested alongside the numeric panel: BMI, glucose, HbA1c,
insulin, HDL, LDL, total cholesterol, triglycerides, pulse pressure and mean
arterial pressure.

Pulse pressure and MAP are derived rather than raw because they carry different
vascular meaning: pulse pressure indexes arterial stiffness, MAP indexes
perfusion pressure, and the two dissociate with age.

Adjusted for age and sex. Antihypertensive use is available as BP_MEDS and is
reported as a sensitivity analysis, since treatment lowers measured pressure in
exactly the people whose vasculature is worst, which biases the raw association
toward zero.

Usage:
    python cardiometabolic.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import atomic_io  # noqa: F401  writes become atomic on import
from scipy import stats

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from hemisphere_age import williams

HERE = Path(__file__).resolve().parent
DIFF = HERE.parent.parent / "diffusion"
AABC = DIFF / "HCP" / "AABC2_subjects_2026_02_05_14_29_11.csv"
VARIANTS = ["classic", "cross", "v2_sphere", "v2_slab", "pv_perp", "anat_x"]


def num(s):
    """Coerce a text column to float, tolerating '<5.7' and stray whitespace."""
    return pd.to_numeric(s.astype(str).str.replace(r"[<>~\s]", "", regex=True),
                         errors="coerce")


def bh(p):
    p = np.asarray(p, float); ok = ~np.isnan(p)
    q = np.full_like(p, np.nan); n = int(ok.sum())
    if n == 0:
        return q
    idx = np.argsort(p[ok])
    r = p[ok][idx] * n / (np.arange(n) + 1)
    r = np.minimum.accumulate(r[::-1])[::-1]
    o = np.empty(n); o[idx] = np.clip(r, 0, 1); q[ok] = o
    return q


def main() -> None:
    d = pd.read_csv(HERE / "measured_pvs_axis_hcpa_b1500_all.csv")
    d = d.sort_values(["Subject_ID", "Visit"]).groupby("Subject_ID").first().reset_index()

    a = pd.read_csv(AABC, low_memory=False)
    a["Subject_ID"] = a.id_event.astype(str).str.split("_").str[0]

    bp = a.bp_sitting.astype(str).str.extract(r"(\d+)\s*/\s*(\d+)")
    a["systolic"] = pd.to_numeric(bp[0], errors="coerce")
    a["diastolic"] = pd.to_numeric(bp[1], errors="coerce")
    a["pulse_pressure"] = a.systolic - a.diastolic
    a["map_mmhg"] = a.diastolic + (a.systolic - a.diastolic) / 3
    a["hba1c_n"] = num(a.hba1c)
    a["insulin_n"] = num(a.insulin)

    COLS = ["systolic", "diastolic", "pulse_pressure", "map_mmhg", "hba1c_n",
            "insulin_n", "bmi", "glucose", "hdl", "friedewald_ldl",
            "cholesterol", "triglyceride"]
    LABEL = {"hba1c_n": "HbA1c", "insulin_n": "insulin", "map_mmhg": "MAP",
             "friedewald_ldl": "LDL"}

    ph = a.groupby("Subject_ID")[COLS + ["BP_MEDS"]].first().reset_index()
    sx = a.groupby("Subject_ID")["sex"].first().reset_index()
    m = d.merge(ph, on="Subject_ID", how="inner").merge(sx, on="Subject_ID", how="left")
    m["sex_n"] = (m.sex.astype(str).str.upper().str[0] == "M").astype(float)
    print(f"{len(m)} participants merged\n")

    def run(sub, tag, extra_cov=False):
        rows = []
        for c in COLS:
            s = sub[[c, "Age", "sex_n"] + VARIANTS + (["BP_MEDS"] if extra_cov else [])].copy()
            s[c] = pd.to_numeric(s[c], errors="coerce")
            s = s.replace([np.inf, -np.inf], np.nan).dropna()
            if len(s) < 60:
                continue
            cov = [s.Age.to_numpy(float), s.sex_n.to_numpy(float)]
            if extra_cov:
                cov.append(s.BP_MEDS.to_numpy(float))
            C = np.column_stack([np.ones(len(s))] + cov)
            def rz(v):
                b, *_ = np.linalg.lstsq(C, np.asarray(v, float), rcond=None)
                return np.asarray(v, float) - C @ b
            yy = rz(s[c])
            rec = {"phenotype": LABEL.get(c, c), "n": len(s)}
            for k in VARIANTS:
                xx = rz(s[k])
                r = float(np.corrcoef(xx, yy)[0, 1])
                dof = len(s) - 2 - len(cov)
                t = r * np.sqrt(dof / max(1 - r * r, 1e-12))
                rec[k] = r
                rec[f"p_{k}"] = float(2 * (1 - stats.t.cdf(abs(t), dof)))
            c1, c2 = rz(s["classic"]), rz(s["v2_slab"])
            rec["williams_p"] = williams(rec["classic"], rec["v2_slab"],
                                         float(np.corrcoef(c1, c2)[0, 1]), len(s))[1]
            rows.append(rec)
        out = pd.DataFrame(rows)
        for k in VARIANTS:
            out[f"q_{k}"] = bh(out[f"p_{k}"].values)
        print(f"--- {tag} ---")
        print(f"{'phenotype':<16s} {'n':>5s} " + " ".join(f"{k:>10s}" for k in VARIANTS)
              + f" {'q(classic)':>11s} {'q(v2_slab)':>11s}")
        for r in out.sort_values("p_classic").itertuples():
            cells = " ".join(f"{getattr(r, k):10.3f}" for k in VARIANTS)
            star = "  *" if min(r.q_classic, r.q_v2_slab) < 0.05 else ""
            print(f"{r.phenotype:<16s} {r.n:5d} {cells} {r.q_classic:11.3f} "
                  f"{r.q_v2_slab:11.3f}{star}")
        print()
        return out

    out = run(m, "age + sex adjusted")
    out.to_csv(HERE / "cardiometabolic.csv", index=False)

    med = m.dropna(subset=["BP_MEDS"])
    if len(med) > 100:
        run(med, "additionally adjusted for antihypertensive use", extra_cov=True)

    untreated = m[pd.to_numeric(m.BP_MEDS, errors="coerce") == 0]
    if len(untreated) > 100:
        run(untreated, f"untreated participants only (n={len(untreated)})")


if __name__ == "__main__":
    main()
