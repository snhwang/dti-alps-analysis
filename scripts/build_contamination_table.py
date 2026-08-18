"""Emit the denominator-contamination table for the manuscript.

Kept separate from Table 4 for two reasons. Template reorientation is not a row
there, and it is the comparison that matters most here. And the quantity is of a
different kind: Table 4 reports what each variant associates with, this reports
whether each variant computes the quantity the index is defined to be.

Reads denominator_contamination_{dlbs,hcpa}.csv, writes contamination_table.tex.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent

ROWS = [("Classic", "classic"),
        ("Template reorientation", "vecreg"),
        ("Refined (cross product)", "refined"),
        ("Anatomical axis", "anat_x")]

CAPTION = (
    "Fiber contamination of the two denominators. DTI-ALPS compares diffusivities in "
    "the plane perpendicular to the local fiber, so a denominator is meant to carry "
    "diffusion across the tract and not along it. Each entry is the share of that "
    "denominator contributed by the fiber's own $\\lambda_1$, "
    "$\\lambda_1(\\hat{\\mathbf u}\\cdot\\hat{\\mathbf v}_1)^2 / "
    "\\sum_i \\lambda_i(\\hat{\\mathbf u}\\cdot\\hat{\\mathbf v}_i)^2$, "
    "median over region-hemispheres. Lower is closer to the defined quantity, and "
    "$\\lambda_1$ is two to three times the perpendicular eigenvalues, so a few "
    "degrees of frame error costs a large share. Template reorientation is evaluated "
    "as $\\mathbf R^{\\top}\\hat{\\mathbf y}$ and $\\mathbf R^{\\top}\\hat{\\mathbf z}$ "
    "in native space, which is identical to warping the tensors and using fixed "
    "template axes, and uses the affine alone: that is its favorable case, since the "
    "nonlinear warp departs from the affine by $5$ to $6^{\\circ}$ and removes less of "
    "the direction spread. The corrected variants are perpendicular to the measured "
    "tract direction by construction, so what remains is per-voxel dispersion about "
    "the regional mean rather than frame error, and any other axis in that same plane "
    "gives the same figures.")


def main() -> None:
    d = {c: pd.read_csv(HERE / f"denominator_contamination_{c}.csv") for c in ("hcpa", "dlbs")}
    n = {c: (v.Subject_ID.nunique(), len(v)) for c, v in d.items()}

    out = [r"\begin{table}[tb]",
           r"\caption{" + CAPTION +
           f" HCP-A, {n['hcpa'][0]} participants and {n['hcpa'][1]} region-hemispheres; "
           f"DLBS, {n['dlbs'][0]} and {n['dlbs'][1]}." + "}",
           r"\label{tbl:contamination}",
           r"\begin{tabular*}{\tblwidth}{@{}LCCCC@{}}",
           r"\toprule",
           r"\textbf{Variant} & \textbf{Proj.} & \textbf{Assoc.} & "
           r"\textbf{Proj.} & \textbf{Assoc.} \\",
           r" & \multicolumn{2}{c}{HCP-A} & \multicolumn{2}{c}{DLBS} \\",
           r"\midrule"]
    for name, key in ROWS:
        cells = [f"{100 * d[c][f'{key}_{reg}'].median():.1f}"
                 for c in ("hcpa", "dlbs") for reg in ("proj", "assoc")]
        out.append(f"{name} & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular*}", r"\end{table}", ""]

    (HERE / "contamination_table.tex").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
