"""
Do any ALPS variants track amyloid or tau PET, and does the measured axis do it better?

The HCP-Aging phenotype test was null once age was removed, but HCP-Aging is a
healthy cohort and its outcomes are proxies. DLBS carries amyloid and tau PET,
which is actual pathology and is the closest available analogue to the
conditions DTI-ALPS is used for.

This also avoids the artefact that made the unadjusted HCP-Aging comparison
misleading. There, every variant's apparent phenotype association was inherited
from its age slope, so the variant with the steepest age correlation won
automatically. Amyloid and tau are age-related too, so age is adjusted for here
as the primary analysis, and the unadjusted version is reported alongside so the
difference is visible rather than hidden.

Amyloid: GlobalSUVR, wave 1, n up to 276.
Tau: TemporalMetaSUVR, the standard meta-ROI, waves 2 and 3 pooled by taking the
earliest available per participant.

Diffusion is matched to the earliest session per participant. PET and MRI are not
same-day, which adds noise but does not bias the comparison between variants,
since all four are computed on identical sessions.

Usage:
    python pet_validation.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from hemisphere_age import williams

HERE = Path(__file__).resolve().parent
DIFF = HERE.parent.parent / "diffusion"
VARIANTS = ["classic", "cross", "v2_sphere", "v2_slab"]


def partial_r(y, x, covs):
    C = np.column_stack([np.ones(len(y))] + covs) if covs else np.ones((len(y), 1))
    def resid(v):
        b, *_ = np.linalg.lstsq(C, np.asarray(v, float), rcond=None)
        return np.asarray(v, float) - C @ b
    rx, ry = resid(x), resid(y)
    if rx.std() < 1e-12 or ry.std() < 1e-12:
        return np.nan, np.nan, None, None
    r = float(np.corrcoef(rx, ry)[0, 1])
    n = len(y)
    dof = n - 2 - (len(covs) if covs else 0)
    t = r * np.sqrt(dof / max(1 - r * r, 1e-12))
    return r, float(2 * (1 - stats.t.cdf(abs(t), dof))), rx, ry


def main() -> None:
    d = pd.read_csv(HERE / "measured_pvs_axis_dlbs.csv")
    d = d.sort_values(["Subject_ID", "Visit"]).groupby("Subject_ID").first().reset_index()

    amy = pd.read_csv(DIFF / "DLBS" / "dlbs_amyloid_w1.csv")[["participant_id", "GlobalSUVR"]]
    amy = amy.rename(columns={"participant_id": "Subject_ID", "GlobalSUVR": "amyloid"})
    tau = pd.concat([pd.read_csv(DIFF / "DLBS" / f"dlbs_tau_w{w}.csv")[
        ["participant_id", "TemporalMetaSUVR"]] for w in (2, 3)])
    tau = (tau.rename(columns={"participant_id": "Subject_ID",
                               "TemporalMetaSUVR": "tau"})
             .dropna().groupby("Subject_ID").first().reset_index())

    for name, pet in (("amyloid GlobalSUVR", amy), ("tau TemporalMeta", tau)):
        col = pet.columns[1]
        m = d.merge(pet, on="Subject_ID", how="inner").dropna(subset=[col, "Age"] + VARIANTS)
        if len(m) < 25:
            print(f"{name}: only {len(m)} matched, skipped\n")
            continue
        print(f"=== {name}: {len(m)} participants matched to diffusion ===")
        for label, covs in (("age + sex adjusted", [m.Age.to_numpy(float)]),
                            ("unadjusted", [])):
            print(f"  {label}")
            res = {}
            for k in VARIANTS:
                r, p, _, _ = partial_r(m[col].to_numpy(float), m[k].to_numpy(float), covs)
                res[k] = (r, p)
                print(f"    {k:<12s} r {r:+.3f}  p {p:.3f}")
            C = (np.column_stack([np.ones(len(m))] + covs) if covs
                 else np.ones((len(m), 1)))
            def rz(v):
                b, *_ = np.linalg.lstsq(C, np.asarray(v, float), rcond=None)
                return np.asarray(v, float) - C @ b
            yy, c1, c2 = rz(m[col]), rz(m["classic"]), rz(m["v2_slab"])
            r1 = np.corrcoef(yy, c1)[0, 1]; r2 = np.corrcoef(yy, c2)[0, 1]
            r23 = np.corrcoef(c1, c2)[0, 1]
            t, p = williams(r1, r2, r23, len(m))
            who = "v2_slab" if abs(r2) > abs(r1) else "classic"
            print(f"    Williams v2_slab vs classic: t={t:+.2f} p={p:.3f} "
                  f"(larger |r|: {who})")
        print()


if __name__ == "__main__":
    main()
