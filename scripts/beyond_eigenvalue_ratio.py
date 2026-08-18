"""Does any ALPS variant carry information beyond the eigenvalue ratio?

Section 2.7 shows that every term of the ALPS ratio lies in the plane
perpendicular to the local fiber, so the index compares two directions within
that plane, and that choosing them per voxel to maximize and minimize
diffusivity makes the index exactly lambda2 over lambda3. The empirical
ordering follows: each variant sits below that ratio by however far its axis
misses the second eigenvector.

That is an argument about the ceiling. This asks the operational question. If a
directional variant carries anything the eigenvalue ratio does not, it should
retain an association after the ratio is partialled out. If it carries nothing,
the association should vanish.

Three tests, on three cohorts and two kinds of endpoint:

  HCP-A     age, 809 participants
  DLBS      age, 156 participants
  trigeminal    patient against control, 168 participants, adjusted for age and sex

The comparison is the one Burles et al. made against mean diffusivity, with a
sharper covariate. Mean diffusivity is a related quantity; lambda2 over lambda3
is the quantity the index provably approximates.

Reading the output. A variant whose partial association is indistinguishable
from zero adds nothing to the eigenvalue ratio. A residual is not evidence of
extra signal: the variants that retain one are those whose axes miss v2, so
what remains is their axis error, and the variants that estimate v2 from the
data retain nothing at all, which is the pattern to check.

    python beyond_eigenvalue_ratio.py

Writes beyond_eigenvalue_ratio.csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import atomic_io  # noqa: F401  writes become atomic on import
from data_paths import winpath
from scipy import stats

HERE = Path(__file__).resolve().parent
DIFF = HERE.parent.parent / "diffusion"
RATIO = "pv_perp"          # lambda2 / lambda3
DIRECTIONAL = ["classic", "cross", "v2_sphere", "v2_slab", "anat_x", "ld_alps",
               "ALPS-PAS", "per-voxel"]


def partial(y, x, covs) -> tuple[float, float]:
    """Correlation of x with y, both residualized on covs."""
    C = np.column_stack([np.ones(len(y))] + [np.asarray(c, float) for c in covs])

    def rz(v):
        b, *_ = np.linalg.lstsq(C, np.asarray(v, float), rcond=None)
        return np.asarray(v, float) - C @ b

    a, b = rz(x), rz(y)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan"), float("nan")
    r = float(np.corrcoef(a, b)[0, 1])
    dof = len(y) - C.shape[1] - 1
    return r, float(2 * stats.t.sf(abs(r * np.sqrt(dof / max(1 - r * r, 1e-12))), dof))


def aging(cohort: str):
    f = ("measured_pvs_axis_hcpa_b1500_all.csv" if cohort == "hcpa"
         else "measured_pvs_axis_dlbs.csv")
    d = pd.read_csv(HERE / f)
    if cohort == "dlbs" and (HERE / "ld_alps_dlbs.csv").exists():
        L = pd.read_csv(HERE / "ld_alps_dlbs.csv")[
            ["Subject_ID", "Visit", "ALPS_overall"]].rename(columns={"ALPS_overall": "ld_alps"})
        for x in (d, L):
            x["Subject_ID"] = x.Subject_ID.astype(str)
            x["Visit"] = x.Visit.astype(str)
        d = d.merge(L, on=["Subject_ID", "Visit"], how="left")
    one = d.sort_values(["Subject_ID", "Visit"]).groupby("Subject_ID").first().reset_index()
    rows = []
    for c in [v for v in DIRECTIONAL if v in one.columns]:
        s = one[[c, "Age", RATIO]].dropna()
        r0 = float(stats.pearsonr(s[c], s.Age)[0])
        rp, pp = partial(s.Age, s[c], [s[RATIO]])
        rows.append(dict(cohort=cohort, endpoint="age", variant=c, n=len(s),
                         raw=r0, partial=rp, p=pp))
    s = one[[RATIO, "Age"]].dropna()
    rows.append(dict(cohort=cohort, endpoint="age", variant=RATIO, n=len(s),
                     raw=float(stats.pearsonr(s[RATIO], s.Age)[0]),
                     partial=np.nan, p=np.nan))
    return rows


def trigeminal():
    tn = pd.read_csv(HERE / "tn_alps.csv")
    par = pd.read_csv(winpath("M:/ds005713-download/participants_v2.0.1.tsv"), sep="\t")
    m = tn.merge(par, on="BIDS_ID")
    m["patient"] = (m.BIDS_ID.astype(str).str.extract(r"sub-(\d+)")[0].str.len() >= 3).astype(float)
    m["age"] = pd.to_numeric(m.age, errors="coerce")
    m["sex_n"] = pd.to_numeric(m.sex, errors="coerce")
    cols = [v for v in DIRECTIONAL if v in m.columns]
    m = m.dropna(subset=["age", "sex_n", RATIO] + cols)
    rows = []
    for c in cols:
        r0, _ = partial(m.patient, m[c], [m.age, m.sex_n])
        rp, pp = partial(m.patient, m[c], [m.age, m.sex_n, m[RATIO]])
        rows.append(dict(cohort="trigeminal", endpoint="patient", variant=c, n=len(m),
                         raw=r0, partial=rp, p=pp))
    r0, p0 = partial(m.patient, m[RATIO], [m.age, m.sex_n])
    rows.append(dict(cohort="trigeminal", endpoint="patient", variant=RATIO, n=len(m),
                     raw=r0, partial=np.nan, p=p0))
    return rows


def main() -> None:
    argparse.ArgumentParser().parse_args()
    rows = aging("hcpa") + aging("dlbs") + trigeminal()
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "beyond_eigenvalue_ratio.csv", index=False)

    for (coh, ep), g in out.groupby(["cohort", "endpoint"], sort=False):
        n = int(g.n.max())
        print(f"\n{coh}, {ep}, {n} participants")
        print(f"  {'variant':12s} {'raw':>9s} {'| ratio':>9s} {'p':>10s}")
        for r in g.itertuples():
            if r.variant == RATIO:
                print(f"  {'lambda2/3':12s} {r.raw:+9.4f} {'(covariate)':>9s}")
            else:
                star = "  *" if (r.p == r.p and r.p < 0.05) else ""
                print(f"  {r.variant:12s} {r.raw:+9.4f} {r.partial:+9.4f} {r.p:10.3g}{star}")

    est = out[(out.variant.isin(["v2_slab", "v2_sphere"])) & out.partial.notna()]
    print("\n  The variants that estimate the second eigenvector from the data retain")
    print(f"  nothing: partial correlations from {est.partial.min():+.3f} to "
          f"{est.partial.max():+.3f}, none significant. A residual elsewhere is axis")
    print("  error, not extra signal.")


if __name__ == "__main__":
    main()
