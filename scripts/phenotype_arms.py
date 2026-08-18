"""Phenotype associations under four nested adjustments.

The two-arm sweep showed that DTI-ALPS associates with about half of everything
measured in an aging cohort until age is adjusted, and with almost nothing
afterwards. That leaves two questions it cannot answer. How much of the
collapse is age alone rather than age and sex together, and does head
orientation account for anything beyond them, which is what this paper claims
about the index in general.

Four nested arms, each adding one block of covariates:

    none        raw correlation
    age         age only
    age+sex     age and sex
    age+sex+pose    and head pose, absolute pitch and total rotation

Reading them in order shows where an association goes. One that survives age and
sex but falls when pose enters is an orientation artifact, which is the claim
this paper makes for the age effect itself. One that falls at the age step was
never about the index.

Benjamini-Hochberg is applied within each arm and variant separately, so the
counts are comparable down a column but a hit in one variant alone is about as
likely as the number of variants would suggest.

    python phenotype_arms.py --cohort hcpa
    python phenotype_arms.py --cohort dlbs

Writes phenotype_arms_<cohort>.csv.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

import atomic_io  # noqa: F401  writes become atomic on import
from scipy import stats

HERE = Path(__file__).resolve().parent
DIFF = HERE.parent.parent / "diffusion"
AABC = DIFF / "HCP" / "AABC2_subjects_2026_02_05_14_29_11.csv"

VARIANTS = ["classic", "cross", "v2_sphere", "v2_slab", "pv_perp", "anat_x", "ld_alps"]
MIN_N = 60
ARMS = (("none", []),
        ("age", ["Age"]),
        ("age+sex", ["Age", "sex_n"]),
        ("age+sex+pose", ["Age", "sex_n", "abs_pitch", "total"]))
DROP = re.compile(r"(age|_id$|^id|date|visit|interview|src_|subjectkey|_dt|wave|s#|"
                  r"mritotau|mritoamy)", re.I)


def fdr(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = np.empty_like(p)
    n = len(p)
    prev = 1.0
    for rank, i in enumerate(o[::-1], 1):
        prev = min(prev, p[i] * n / (n - rank + 1))
        q[i] = prev
    return q


def load(cohort: str) -> pd.DataFrame:
    """Index variants, phenotypes and head pose, one session per participant."""
    if cohort == "hcpa":
        d = pd.read_csv(HERE / "measured_pvs_axis_hcpa_b1500_all.csv")
        hr = pd.read_csv(HERE / "head_rotation_hcpa.csv")
    else:
        d = pd.read_csv(HERE / "measured_pvs_axis_dlbs.csv")
        hr = pd.read_csv(HERE / "head_rotation_dlbs.csv")
        ld = HERE / "ld_alps_dlbs.csv"
        if ld.exists():
            L = pd.read_csv(ld)[["Subject_ID", "Visit", "ALPS_overall"]].rename(
                columns={"ALPS_overall": "ld_alps"})
            for f in (d, L):
                f["Subject_ID"] = f.Subject_ID.astype(str)
                f["Visit"] = f.Visit.astype(str)
            d = d.merge(L, on=["Subject_ID", "Visit"], how="left")
    for f in (d, hr):
        f["Subject_ID"] = f.Subject_ID.astype(str)
        f["Visit"] = f.Visit.astype(str)
    # pose has to join before collapsing to one session, since it is per session
    d = d.merge(hr[["Subject_ID", "Visit", "pitch", "total"]], on=["Subject_ID", "Visit"],
                how="left")
    d["abs_pitch"] = d.pitch.abs()
    d = d.sort_values(["Subject_ID", "Visit"]).groupby("Subject_ID").first().reset_index()

    if cohort == "hcpa":
        a = pd.read_csv(AABC, low_memory=False)
        a["Subject_ID"] = a.id_event.astype(str).str.split("_").str[0]
        num = [c for c in a.columns
               if pd.api.types.is_numeric_dtype(a[c]) and not DROP.search(c)
               and a[c].notna().sum() >= 100 and a[c].nunique() > 4]
        ph = a.groupby("Subject_ID")[num].first().reset_index()
        sx = a.groupby("Subject_ID")["sex"].first().reset_index()
        m = d.merge(ph, on="Subject_ID").merge(sx, on="Subject_ID", how="left")
        m["sex_n"] = (m.sex.astype(str).str.upper().str[0] == "M").astype(float)
    else:
        a = pd.read_csv(DIFF / "DLBS" / "ds004856_participants.tsv", sep="\t", low_memory=False)
        a["Subject_ID"] = a.participant_id.astype(str)
        m = d.merge(a, on="Subject_ID")
        sexcol = next(c for c in m.columns if c.lower() == "sex")
        m["sex_n"] = (m[sexcol].astype(str).str.lower().str[0] == "m").astype(float)
        num = [c for c in m.columns
               if pd.api.types.is_numeric_dtype(m[c]) and not DROP.search(c)]
        m.attrs["num"] = num
    if cohort == "hcpa":
        m.attrs["num"] = num
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=["hcpa", "dlbs"], default="hcpa")
    args = ap.parse_args()

    m = load(args.cohort)
    variants = [v for v in VARIANTS if v in m.columns]
    exclude = set(variants) | {"sex_n", "Age", "pitch", "total", "abs_pitch",
                               "v2_to_x", "v2_to_cross", "cross_to_x"}
    cand = [c for c in m.attrs["num"] if c not in exclude]
    have_pose = m.abs_pitch.notna().sum()
    print(f"{args.cohort}: {len(m)} participants, {len(cand)} candidate phenotypes, "
          f"{have_pose} with head pose\n")

    rows = []
    for arm, covs in ARMS:
        for c in cand:
            need = [c] + covs + variants
            s = m[[x for x in need if x in m.columns]].replace(
                [np.inf, -np.inf], np.nan).dropna()
            if len(s) < MIN_N or s[c].nunique() < 5:
                continue
            C = np.column_stack([np.ones(len(s))] + [s[k].to_numpy(float) for k in covs])

            def rz(v):
                b, *_ = np.linalg.lstsq(C, np.asarray(v, float), rcond=None)
                return np.asarray(v, float) - C @ b

            y = rz(s[c])
            if y.std() < 1e-12:
                continue
            rec = {"arm": arm, "phenotype": c, "n": len(s)}
            for k in variants:
                x = rz(s[k])
                if x.std() < 1e-12:
                    rec[k] = np.nan
                    rec[f"p_{k}"] = np.nan
                    continue
                r = float(np.corrcoef(x, y)[0, 1])
                dof = len(s) - C.shape[1] - 1
                rec[k] = r
                rec[f"p_{k}"] = float(2 * stats.t.sf(abs(r * np.sqrt(dof / max(1 - r * r, 1e-12))),
                                                     dof))
            rows.append(rec)

    out = pd.DataFrame(rows)
    for arm in out.arm.unique():
        sel = out.arm == arm
        for k in variants:
            q = np.full(sel.sum(), np.nan)
            ok = out.loc[sel, f"p_{k}"].notna().to_numpy()
            if ok.any():
                q[ok] = fdr(out.loc[sel, f"p_{k}"].to_numpy()[ok])
            out.loc[sel, f"q_{k}"] = q
    out.to_csv(HERE / f"phenotype_arms_{args.cohort}.csv", index=False)

    width = max(len(v) for v in variants) + 1
    print("  phenotypes surviving FDR, by adjustment")
    print("  " + " " * width + "".join(f"{a:>16s}" for a, _ in ARMS))
    for k in variants:
        cells = []
        for arm, _ in ARMS:
            sel = (out.arm == arm) & out[f"q_{k}"].notna()
            cells.append(f"{int((out.loc[sel, f'q_{k}'] < 0.05).sum()):>16d}")
        print(f"  {k:<{width}s}" + "".join(cells))
    n_arm = {a: int(((out.arm == a)).sum()) for a, _ in ARMS}
    print(f"\n  phenotypes tested per arm: {n_arm}")
    print("  A count that falls between age+sex and age+sex+pose is an association")
    print("  the index carried through head orientation rather than through tissue.")


if __name__ == "__main__":
    main()
