r"""Generate the covariate-adjustment table from the models that produced it.

The table was hand-entered and had drifted from every analysis it reports. Its
region-volume rows read "$-0.448$ to $-0.444$ ($0.9\%$)" in HCP-A and "none" in
DLBS, which says adjusting for region volume does nothing. The regression chain
in covariate_models.txt says it takes the classic coefficient from $-0.446$ to
$-0.302$, and the sentence directly above the table says the same thing, "the
one term that changes the reading, absorbing about a third". So the table
contradicted its own text, and it contradicted it in the direction that makes
the argument look unnecessary.

The numbers here come from two places, and the caption says which is which.

  Sex, motion, region volume, site and scanner come from covariate_models.txt,
  the nested chain of models. The percentage is the reduction in the age
  coefficient from the age-only model, so it accumulates down a cohort block.

  Composition comes from the placement-quality set, where the two off-tract
  fractions enter as separate covariates, which is a different convention and
  a different sample. It is reported as absorption, not as a coefficient pair.

    python build_covariate_table.py            # rewrite the table in place
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BS = chr(92)
MODELS = HERE / "covariate_models.txt"


def chain(text: str) -> dict[str, dict[str, list[tuple[str, float]]]]:
    """cohort -> variant -> [(model, beta_age)], in file order."""
    out: dict[str, dict[str, list[tuple[str, float]]]] = {}
    cohort = None
    for block in re.split(r"={20,}", text):
        m = re.search(r"(HCP-A|DLBS): (\d+) sessions", block)
        if m:
            cohort = m.group(1)
            continue
        if cohort is None or "beta_age" not in block:
            continue
        out[cohort] = {}
        for part in re.split(r"\n(?=\w+:)", block):
            name = part.strip().split(":")[0]
            rows = [(a.strip(), float(b)) for a, b in
                    re.findall(r"^\s{2}(\S.*?)\s{2,}(-?\d\.\d+)\s", part, re.M)
                    if "beta_age" not in a]
            if rows:
                out[cohort][name] = rows
        cohort = None
    return out


def absorbed(d: pd.DataFrame, col: str, extra: list) -> float:
    """Percent of the age coefficient taken by the extra covariates."""
    def z(v):
        v = np.asarray(v, float)
        return (v - v.mean()) / v.std(ddof=1)
    y, age = z(d[col]), z(d.Age)
    b0 = np.linalg.lstsq(np.column_stack([np.ones(len(d)), age]), y,
                         rcond=None)[0][1]
    X = np.column_stack([np.ones(len(d)), age] + [z(e) for e in extra])
    b1 = np.linalg.lstsq(X, y, rcond=None)[0][1]
    return 100 * (1 - abs(b1) / abs(b0))


WANT = [("age only", "age only"),
        ("+ sex, motion", "sex and motion"),
        ("+ sex, motion, ROI volume", "region volume"),
        ("+ all, and scanner", "site and scanner")]

ch = chain(MODELS.read_text(encoding="utf-8"))
rows: list[tuple[str, str, str]] = []
for cohort in ("HCP-A", "DLBS"):
    base = {v: abs(ch[cohort][v][0][1]) for v in ("classic", "refined")}
    for key, label in WANT:
        cells = []
        for v in ("classic", "refined"):
            hit = [b for k, b in ch[cohort][v] if k == key]
            cells.append(f"${hit[0]:.3f}$"
                         + ("" if key == "age only" else
                            f" (${100 * (1 - abs(hit[0]) / base[v]):.1f}" + BS + "%$)")
                         if hit else "--")
        if cells == ["--", "--"]:
            continue          # DLBS is one site, so that model was never fitted
        name = (f"{cohort}, age only" if key == "age only"
                else BS + "quad + " + label)
        rows.append((name, cells[0], cells[1]))

# Composition, the other convention, HCP-A only.
d = pd.read_csv(HERE / "roi_placement_quality_hcpa_b1500.csv").dropna(
    subset=["Age", "classic", "refined_slab", "n_scr", "n_slf",
            "slf_off_tract", "scr_off_tract"])
comp = [absorbed(d, c, [d.slf_off_tract, d.scr_off_tract])
        for c in ("classic", "refined_slab")]
rows.append(("HCP-A composition, two measures",
             f"${comp[0]:.1f}" + BS + "%$ absorbed",
             f"${comp[1]:.1f}" + BS + "%$ absorbed"))

body = [BS + "toprule",
        (BS + "textbf{Model} & " + BS + "textbf{Classic} & " + BS
         + "textbf{Refined} " + BS + BS),
        BS + "midrule"]
body += [" & ".join(r) + " " + BS + BS for r in rows[:-1]]
body += [BS + "midrule", " & ".join(rows[-1]) + " " + BS + BS, BS + "bottomrule"]
frag = "\n".join(body)
(HERE / "covariate_table.tex").write_text(frag, encoding="utf-8")

tex = HERE.parent / "mri_revision.tex"
s = tex.read_text(encoding="utf-8")
i = s.index(BS + "toprule", s.index("label{tbl:covariate-adjustment}"))
j = s.index(BS + "end{tabular*}", i)
tex.write_text(s[:i] + frag + "\n" + s[j:], encoding="utf-8")
for r in rows:
    print("  " + " | ".join(x.replace(BS, "") for x in r))
print(f"\n  rewrote tbl:covariate-adjustment ({len(rows)} rows)")
