"""
Does the measured-axis index track clinical phenotypes better than classic ALPS?

The case for DTI-ALPS is empirical: it correlates with a long list of
pathologies. So the test that matters for an improved version is not whether it
measures perivascular flow, which is out of scope, but whether it correlates
more strongly with the outcomes the index is actually used against.

Age is only a proxy for that. This uses the HCP-Aging phenotypes directly:

  plasma      NfL, GFAP, total tau. Log-transformed, since concentrations are
              right-skewed.
  cognition   MoCA total, Trail Making B (executive), and the NIH Toolbox
              memory, fluid and crystallised composites.
  sleep       PSQI global. Included because the ALPS literature leans heavily
              on sleep, so it is the phenotype a reader will look for.

Partial correlations adjust for age and sex, which is the convention in this
literature and also the conservative choice here: the measured-axis index has
the stronger age association, so leaving age in would hand it an advantage that
has nothing to do with the phenotype.

One session per participant, earliest visit, so repeat visits are not treated as
independent. Differences against classic are tested with Williams' test for
dependent overlapping correlations, and FDR-corrected across phenotypes.

Usage:
    python phenotype_validation.py
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

OUTCOMES = [
    ("NfL_Conc_pg_ml", "plasma NfL", True),
    ("GFAP_Conc_pg_ml", "plasma GFAP", True),
    ("tTau_Conc_pg_ml", "plasma total tau", True),
    ("moca_sum", "MoCA total", False),
    ("trail2", "Trail Making B", True),
    ("Memory_Tr35_60y", "NIHTB memory", False),
    ("FluidIQ_Tr35_60y", "NIHTB fluid", False),
    ("CrystIQ_Tr35_60y", "NIHTB crystallised", False),
    ("psqi_global", "PSQI sleep", False),
]


def partial_r(y, x, covs):
    """Correlation between x and y after regressing both on covs."""
    C = np.column_stack([np.ones(len(y))] + covs)
    def resid(v):
        b, *_ = np.linalg.lstsq(C, v, rcond=None)
        return v - C @ b
    rx, ry = resid(np.asarray(x, float)), resid(np.asarray(y, float))
    if rx.std() < 1e-12 or ry.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def bh(p):
    p = np.asarray(p, float)
    ok = ~np.isnan(p)
    q = np.full_like(p, np.nan)
    idx = np.argsort(p[ok])
    n = ok.sum()
    ranked = p[ok][idx] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n); out[idx] = np.clip(ranked, 0, 1)
    q[ok] = out
    return q


ADJUST_AGE = "--no-age" not in sys.argv


def main() -> None:
    d = pd.read_csv(HERE / "measured_pvs_axis_hcpa_b1500.csv")
    d["id_event"] = d.Subject_ID.astype(str) + "_" + d.Visit.astype(str)
    d = d.sort_values(["Subject_ID", "Visit"]).groupby("Subject_ID").first().reset_index()
    d["id_event"] = d.Subject_ID.astype(str) + "_" + d.Visit.astype(str)

    a = pd.read_csv(AABC, low_memory=False)
    keep = ["id_event", "sex"] + [c for c, _, _ in OUTCOMES if c in a.columns]
    m = d.merge(a[keep], on="id_event", how="inner")
    m["sex_n"] = (m.sex.astype(str).str.upper().str[0] == "M").astype(float)
    print(f"merged {len(m)} participants (one session each)\n")

    rows = []
    for col, label, logit in OUTCOMES:
        if col not in m.columns:
            print(f"  {label:<20s} column absent, skipped")
            continue
        s = m[[col, "Age", "sex_n"] + VARIANTS].copy()
        s[col] = pd.to_numeric(s[col], errors="coerce")
        s = s.dropna()
        if logit:
            s = s[s[col] > 0]
            s[col] = np.log(s[col])
        if len(s) < 30:
            print(f"  {label:<20s} n={len(s)} too few, skipped")
            continue
        covs = ([s.Age.to_numpy(float), s.sex_n.to_numpy(float)]
                if ADJUST_AGE else [s.sex_n.to_numpy(float)])
        rec = {"outcome": label, "n": len(s)}
        for k in VARIANTS:
            rec[k] = partial_r(s[col].to_numpy(float), s[k].to_numpy(float), covs)
        # Williams test, v2_slab against classic, on the residualised variables
        C = np.column_stack([np.ones(len(s))] + covs)
        def rz(v):
            b, *_ = np.linalg.lstsq(C, np.asarray(v, float), rcond=None)
            return np.asarray(v, float) - C @ b
        yy, c1, c2 = rz(s[col]), rz(s["classic"]), rz(s["v2_slab"])
        r1 = np.corrcoef(yy, c1)[0, 1]; r2 = np.corrcoef(yy, c2)[0, 1]
        r23 = np.corrcoef(c1, c2)[0, 1]
        t, p = williams(r1, r2, r23, len(s))
        rec["williams_t"], rec["williams_p"] = t, p
        rows.append(rec)

    if not rows:
        print("no usable outcomes")
        return
    out = pd.DataFrame(rows)
    out["q_fdr"] = bh(out.williams_p.values)
    out.to_csv(HERE / f"phenotype_validation{'' if ADJUST_AGE else '_noage'}.csv", index=False)

    print(f"{'outcome':<22s} {'n':>5s} " + " ".join(f"{k:>10s}" for k in VARIANTS)
          + f" {'v2 vs cls':>10s} {'q':>8s}")
    for r in out.itertuples():
        cells = " ".join(f"{getattr(r, k):10.3f}" for k in VARIANTS)
        better = "v2" if abs(r.v2_slab) > abs(r.classic) else "classic"
        print(f"{r.outcome:<22s} {r.n:5d} {cells} {better:>10s} {r.q_fdr:8.3f}")

    n_better = sum(abs(r.v2_slab) > abs(r.classic) for r in out.itertuples())
    print(f"\nv2_slab has the larger |partial r| in {n_better} of {len(out)} phenotypes")
    sig = out[(out.q_fdr < 0.05)]
    print(f"differences surviving FDR: {len(sig)}")
    for r in sig.itertuples():
        who = "v2_slab" if abs(r.v2_slab) > abs(r.classic) else "classic"
        print(f"  {r.outcome}: {who} stronger (q={r.q_fdr:.3f})")


if __name__ == "__main__":
    main()
