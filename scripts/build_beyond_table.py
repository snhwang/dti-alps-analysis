"""Emit tbl:beyond, the age association before and after partialling the ratio.

The table was maintained by hand while its numbers came from three different
scripts, which is how it came to hold a mixture of placements: re-sphered
variants beside warped-mask comparators. It is generated here instead, from
the canonical index tables, so the whole table moves together whenever the
placement rule or the sample does.

Every cell is one session per participant. The raw column is the plain Pearson
correlation with age. The |ratio column partials out pv_perp, the eigenvalue
ratio measured in the same voxels as the variant beside it.

Two rows are not like the others and the caption has to say so.

pv_perp is the ratio, so its |ratio cell is empty by construction rather than
by measurement. Reporting a number there would be a variable regressed against
itself, which returns whatever the rounding gives.

LD-ALPS places its own regions. Its |ratio cell partials out a ratio measured
in our voxels, not its own, so the adjustment necessarily under-corrects and a
residual is expected on those grounds alone. It is reported for completeness
and is not evidence that LD-ALPS carries anything the ratio does not.

    python build_beyond_table.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent

# manuscript label -> column in the index table
ROWS = [
    ("Classic", "classic"),
    ("Refined (cross product)", "cross"),
    ("Measured axis", "v2_slab"),
    ("Anatomical axis", "anat_x"),
    ("LD-ALPS", "LD-ALPS"),
    ("ALPS-PAS", "ALPS-PAS"),
    ("Per-voxel", "per-voxel"),
]
RATIO_ROW = ("$\\lambda_2/\\lambda_3$", "pv_perp")


def partial(y, x, z):
    ok = ~(np.isnan(y) | np.isnan(x) | np.isnan(z))
    y, x, z = y[ok], x[ok], z[ok]
    if len(y) < 30:
        return np.nan, np.nan
    A = np.column_stack([np.ones(len(z)), z])

    def rz(v):
        b, *_ = np.linalg.lstsq(A, v, rcond=None)
        return v - A @ b
    ry, rx = rz(y), rz(x)
    # pv_perp against itself leaves only rounding. Report it as undefined.
    if np.std(ry) <= 1e-8 * max(np.std(y), 1e-30) or np.std(rx) == 0:
        return np.nan, np.nan
    r = float(np.corrcoef(ry, rx)[0, 1])
    dof = len(y) - A.shape[1] - 1
    t = r * np.sqrt(dof / max(1 - r ** 2, 1e-12))
    return r, float(2 * stats.t.sf(abs(t), dof))


def load(cohort: str) -> pd.DataFrame:
    stem = ("measured_pvs_axis_hcpa_b1500_all" if cohort == "hcpa"
            else "measured_pvs_axis_dlbs")
    d = pd.read_csv(HERE / f"{stem}.csv")
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
    return (d.sort_values(["Subject_ID", "Visit"])
             .groupby("Subject_ID").first().reset_index())


def cells(d: pd.DataFrame, col: str) -> tuple[str, str, float, float]:
    if col not in d.columns:
        return "--", "--", np.nan, np.nan
    y = d[col].to_numpy(float)
    age = d.Age.to_numpy(float)
    ok = ~(np.isnan(y) | np.isnan(age))
    if ok.sum() < 30:
        return "--", "--", np.nan, np.nan
    raw = float(stats.pearsonr(y[ok], age[ok])[0])
    r, p = partial(y, age, d.pv_perp.to_numpy(float))
    star = "" if np.isnan(r) or p >= 0.05 else "*"
    return (f"{raw:+.3f}".replace("+", ""),
            "--" if np.isnan(r) else f"{r:+.3f}{star}".replace("+", "+"),
            raw, r)


def main() -> None:
    argparse.ArgumentParser().parse_args()
    H, D = load("hcpa"), load("dlbs")
    print(f"HCP-A {len(H)} participants, DLBS {len(D)} participants\n")

    lines, rec = [], []
    for label, col in ROWS + [RATIO_ROW]:
        if label == RATIO_ROW[0]:
            lines.append("\\midrule")
        hr, hp, hrv, hpv = cells(H, col)
        dr, dp, drv, dpv = cells(D, col)
        lines.append(f"{label} & {hr} & {hp} & {dr} & {dp} \\\\")
        rec.append({"variant": label, "hcpa_raw": hrv, "hcpa_given_ratio": hpv,
                    "dlbs_raw": drv, "dlbs_given_ratio": dpv})
        print(f"   {label:<26s} HCP-A {hr:>7s} {hp:>8s}    DLBS {dr:>7s} {dp:>8s}")

    body = "\n".join(lines)
    (HERE / "beyond_table_rows.tex").write_text(body + "\n", encoding="utf-8")
    pd.DataFrame(rec).to_csv(HERE / "beyond_table.csv", index=False)

    corrected = [r for r in rec if r["variant"] in
                 ("Refined (cross product)", "Anatomical axis",
                  "ALPS-PAS", "Per-voxel")]
    worst = max(corrected, key=lambda r: abs(r["hcpa_given_ratio"]))
    print(f"\n   largest HCP-A residual among corrected variants: "
          f"{worst['variant']} at {worst['hcpa_given_ratio']:+.3f}")
    print("   (LD-ALPS excluded: its regions are its own, so the ratio it is")
    print("    adjusted for was not measured in the voxels it used)")
    print("\n   wrote beyond_table_rows.tex and beyond_table.csv")


if __name__ == "__main__":
    main()
