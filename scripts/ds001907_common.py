"""Group assignment and demographics for OpenNeuro ds001907.

THE README IS WRONG ABOUT WHICH PREFIX IS WHICH, so this module exists to make
sure the mapping is stated once and checked, rather than copied from the README
into every script. The README says

    This project contains 46 healthy aging (n = 25, RC41*) participants and
    participants with Parkinson's disease (n = 21, RC42*) at two sessions each.

The data paper (Day et al., Data in Brief 2020, PMC7217223) says the opposite:

    The sample of subjects includes 25 participants with PD and 21 healthy
    controls (HC) who participated in two scanning sessions.

with Table 1 giving PD as age 66.1 (SD 10.0), 18 M / 7 F, Hoehn and Yahr 2.0,
and HC as age 62.1 (SD 9.9), 9 M / 12 F, no Hoehn and Yahr.

The dataset's own demographics.csv settles it. RC41 has n=25, age 66.2 (SD
10.0), 18 M / 7 F, and a Hoehn and Yahr stage for every subject. RC42 has n=21,
age 62.1 (SD 9.9), 9 M / 12 F, and no Hoehn and Yahr at all. Every other
Parkinson-specific field, the UK Brain Bank criteria, Schwab and England,
diagnosis accuracy, tremor, rigidity, bradykinesia, is filled for RC41 and empty
for RC42.

    RC41 = Parkinson's disease.  RC42 = healthy control.

Getting this backwards inverts the sign of every group contrast, so
`assert_group_mapping()` re-derives it from demographics.csv rather than trusting
this docstring, and any script here should call it before reporting a result.

demographics.csv is a ragged export: some rows carry more fields than the header
because late free-text columns contain unquoted commas. The columns used here
all sit at or before index 317, ahead of the first ragged field, and
`assert_group_mapping` checks their values are still sane rather than assuming.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DEMOG = HERE / "ds001907_demographics.csv"
DEST = Path(r"M:\ds001907-download")

# Column positions in demographics.csv, all ahead of the ragged tail.
COL = {"id": 0, "hoehn_yahr": 292, "sex": 301, "education_years": 316, "age": 317}
PD_PREFIX, HC_PREFIX = "RC41", "RC42"


def group(subject: str) -> str:
    """'patient' or 'control' from a subject label, with or without 'sub-'."""
    s = subject.replace("sub-", "")
    if s.startswith(PD_PREFIX):
        return "patient"
    if s.startswith(HC_PREFIX):
        return "control"
    raise ValueError(f"unrecognized subject id {subject!r}")


def demographics() -> pd.DataFrame:
    """One row per subject: group, age, sex, education, Hoehn and Yahr."""
    rows = list(csv.reader(DEMOG.open(newline="", encoding="utf-8-sig")))
    out = []
    for r in rows[1:]:
        def val(k):
            j = COL[k]
            v = r[j].strip() if j < len(r) else ""
            return None if v in ("", "NA") else v
        out.append({
            "subject": r[0].strip(),
            "group": group(r[0]),
            "age": pd.to_numeric(val("age"), errors="coerce"),
            "sex": val("sex"),
            "education_years": pd.to_numeric(val("education_years"), errors="coerce"),
            "hoehn_yahr": pd.to_numeric(val("hoehn_yahr"), errors="coerce"),
        })
    return pd.DataFrame(out)


def assert_group_mapping() -> pd.DataFrame:
    """Re-derive the mapping from the data and fail loudly if it has flipped.

    The test is Hoehn and Yahr, which only a Parkinson's cohort can have. If the
    arm this module calls 'patient' is not the staged one, every downstream
    contrast would carry the wrong sign, so this raises rather than warns.
    """
    d = demographics()
    staged = d.dropna(subset=["hoehn_yahr"]).group.unique()
    if list(staged) != ["patient"]:
        raise AssertionError(
            f"Hoehn and Yahr is present in {sorted(staged)}, so the prefix-to-group "
            f"mapping in ds001907_common.py is wrong and every contrast built on "
            f"it would be inverted.")
    n_pd = int((d.group == "patient").sum())
    n_hc = int((d.group == "control").sum())
    if (n_pd, n_hc) != (25, 21):
        raise AssertionError(f"expected 25 PD and 21 HC per the data paper, "
                             f"got {n_pd} and {n_hc}")
    return d


if __name__ == "__main__":
    d = assert_group_mapping()
    print("group mapping verified against Hoehn and Yahr and the data paper\n")
    for g, s in d.groupby("group"):
        hy = s.hoehn_yahr.dropna()
        print(f"  {g:8s} n={len(s):3d}  age {s.age.mean():.1f} (SD {s.age.std():.1f})  "
              f"{(s.sex=='Male').sum()} M / {(s.sex=='Female').sum()} F  "
              f"education {s.education_years.mean():.1f} y"
              + (f"  H&Y {hy.mean():.1f} (SD {hy.std():.1f})" if len(hy) else "  H&Y none"))
