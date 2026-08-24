"""
Age associations adjusted for the covariates Reviewer 1 asked about.

The submitted manuscript reported Pearson correlations with age and nothing
else. Reviewer 1 asked for multivariable models controlling for sex, motion,
brain atrophy, white matter hyperintensity burden, ROI volume and scanning
batch.

What is available here: sex, head motion (mean absolute eddy-corrected RMS
displacement), ROI volume (surviving voxel count), and, in HCP-A only,
acquisition site and scanner. Atrophy and white matter hyperintensity burden
were not derived for either cohort and are noted as a limitation rather than
silently omitted. DLBS is single-site, so batch does not apply there.

Standard errors are clustered by participant throughout, since both cohorts
contain repeat visits and the original analysis treated sessions as
independent.

Reported for the classic index and for the refined index with directions taken
from the tract slab, which is the recommended form.

Usage:
    python covariate_models.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import os

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from alps_common import parse_age

HERE = Path(__file__).resolve().parent
DIFF = HERE.parent.parent / "diffusion"

lines: list[str] = []


def say(t: str = "") -> None:
    print(t)
    lines.append(t)


def build(cohort: str) -> pd.DataFrame:
    # Honour the tensor-shell suffix, as decoupled_roi.py does. Without this
    # the HCP-A models silently read the multishell file rather than the
    # b=1500 file the rest of the paper uses.
    # DLBS is single-shell b=1000 and has no suffixed file; only HCP-A does.
    shell = os.environ.get("ALPS_TENSOR_SUFFIX", "") if cohort == "hcpa" else ""
    res = pd.read_csv(HERE / f"decoupled_roi_{cohort}{shell}.csv")
    if cohort == "hcpa":
        src = pd.read_csv(DIFF / "HCP" / "hcpa_alps_spheres_5mm.csv")
        mot = pd.read_csv(DIFF / "HCP" / "hcpa_motion.csv")
        src = src.merge(mot[["Subject_ID", "Visit", "Eddy_Mean_RMS"]],
                        on=["Subject_ID", "Visit"], how="left")
        keep = ["Subject_ID", "Visit", "Sex", "site", "scanner",
                "n_proj", "n_assoc", "Eddy_Mean_RMS"]
    else:
        src = pd.read_csv(DIFF / "DLBS" / "dlbs_alps_spheres_5mm.csv")
        mot = pd.read_csv(DIFF / "DLBS" / "dlbs_motion.csv")
        src = src.merge(mot[["DTI_Session_ID", "Eddy_Mean_RMS"]],
                        on="DTI_Session_ID", how="left")
        src["Visit"] = src["Session"]
        src["site"] = "single"
        src["scanner"] = "single"
        keep = ["Subject_ID", "Visit", "Sex", "site", "scanner",
                "n_proj", "n_assoc", "Eddy_Mean_RMS"]
    d = res.merge(src[keep], on=["Subject_ID", "Visit"], how="left")
    d["Age"] = parse_age(d["Age"])
    d["nvox"] = pd.to_numeric(d.n_proj, errors="coerce") + pd.to_numeric(d.n_assoc, errors="coerce")
    d["motion"] = pd.to_numeric(d.Eddy_Mean_RMS, errors="coerce")
    d["Sex"] = d["Sex"].astype(str).str.strip().str.upper().str[0]
    d = d[d.Sex.isin(["M", "F"])]

    # Body habitus. pose_phenotype.py finds that head position tracks BMI in
    # both cohorts, r=+0.28 in DLBS and +0.19 in HCP-A given age and sex, so a
    # body-size term belongs in a model that claims to control what else could
    # drive an age association. It was not in the reviewer's list, which is why
    # it was missing.
    if cohort == "hcpa":
        a = pd.read_csv(DIFF / "HCP" / "AABC2_subjects_2026_02_05_14_29_11.csv",
                        low_memory=False)
        a["Subject_ID"] = a.id_event.astype(str).str.split("_").str[0]
        if "bmi" in a.columns:
            b = a.groupby("Subject_ID")["bmi"].first().reset_index()
            b["Subject_ID"] = b.Subject_ID.astype(str)
            d["Subject_ID"] = d.Subject_ID.astype(str)
            d = d.merge(b, on="Subject_ID", how="left")
    else:
        t = pd.read_csv(DIFF / "DLBS" / "ds004856_participants.tsv", sep="	",
                        low_memory=False)
        t["Subject_ID"] = t.participant_id.astype(str)
        if "BMI_W1" in t.columns:
            d["Subject_ID"] = d.Subject_ID.astype(str)
            d = d.merge(t[["Subject_ID", "BMI_W1"]].rename(
                columns={"BMI_W1": "bmi"}), on="Subject_ID", how="left")
    if "bmi" not in d.columns:
        d["bmi"] = float("nan")
    return d.dropna(subset=["Age", "classic", "refined_slab", "nvox", "motion"])


def run(d: pd.DataFrame, col: str, formula: str) -> tuple[float, float, float]:
    cols = ["Subject_ID", "Age", "Sex", "site", "scanner", "nvox", "motion", col]
    if "bmi" in formula:
        cols.append("bmi")
    x = d[cols].dropna()
    x = x.rename(columns={col: "y"})
    for c in ("y", "Age", "nvox", "motion") + (("bmi",) if "bmi" in formula else ()):
        x[c + "_z"] = (x[c] - x[c].mean()) / x[c].std(ddof=1)
    f = smf.ols(formula, x).fit(cov_type="cluster",
                                cov_kwds={"groups": x["Subject_ID"]})
    return (float(f.params["Age_z"]), float(f.bse["Age_z"]),
            float(f.pvalues["Age_z"]))


def main() -> None:
    for cohort, label in (("hcpa", "HCP-A"), ("dlbs", "DLBS")):
        d = build(cohort)
        nsite = d.site.nunique()
        say(f"\n{'='*74}\n{label}: {len(d)} sessions, {d.Subject_ID.nunique()} "
            f"participants, {nsite} site(s)\n{'='*74}")
        say(f"sex: {d.Sex.value_counts().to_dict()}")

        models = [("age only", "y_z ~ Age_z"),
                  ("+ sex", "y_z ~ Age_z + C(Sex)"),
                  ("+ sex, motion", "y_z ~ Age_z + C(Sex) + motion_z"),
                  ("+ sex, motion, ROI volume",
                   "y_z ~ Age_z + C(Sex) + motion_z + nvox_z")]
        # BMI last, on its own matched sample, since it is missing for some
        # participants and a smaller sample moves the coefficient by itself.
        models.append(("+ sex, motion, ROI volume, BMI",
                       "y_z ~ Age_z + C(Sex) + motion_z + nvox_z + bmi_z"))
        if nsite > 1:
            models.append(("+ sex, motion, ROI volume, site",
                           "y_z ~ Age_z + C(Sex) + motion_z + nvox_z + C(site)"))
            models.append(("+ all, and scanner",
                           "y_z ~ Age_z + C(Sex) + motion_z + nvox_z + C(site) + C(scanner)"))

        for col, nm in (("classic", "classic"), ("refined_slab", "refined")):
            say(f"\n{nm}:")
            say(f"  {'model':<34s} {'beta_age':>9s} {'SE':>8s} {'p':>11s}")
            for mlabel, formula in models:
                try:
                    b, se, p = run(d, col, formula)
                    say(f"  {mlabel:<34s} {b:9.4f} {se:8.4f} {p:11.3e}")
                except Exception as e:
                    say(f"  {mlabel:<34s}  failed: {type(e).__name__}")

    say("\nAtrophy and white matter hyperintensity burden were not derived for")
    say("either cohort and remain unadjusted; this is stated as a limitation.")
    (HERE / "covariate_models.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {HERE/'covariate_models.txt'}")


if __name__ == "__main__":
    main()
