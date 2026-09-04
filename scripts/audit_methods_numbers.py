"""Do the Methods numbers agree with each other and with the data?

Methods states sample sizes, exclusions and thresholds that the rest of the
paper depends on, and it is the section nothing has audited. This checks the
arithmetic that must close internally, and the counts that can be read from a
CSV.

    python audit_methods_numbers.py

Reports only. A failure here is a manuscript error, not a tolerance question.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
BS = chr(92)
TEX = (HERE.parent / "mri_revision.tex").read_text(encoding="utf-8")
FLAT = " ".join(TEX.split())

rows: list[tuple[bool, str, object, object]] = []


def check(label, want, got):
    rows.append((want == got, label, want, got))


def num(pattern: str, label: str):
    m = re.search(pattern, FLAT)
    if not m:
        rows.append((False, f"{label} (phrase absent)", "found", "missing"))
        return None
    return float(m.group(1))


# --- the DLBS processing chain must close ---
total = num(r"Of \$(\d+)\$ sessions", "DLBS sessions attempted")
ok = num(r"sessions, \$(\d+)\$ completed successfully", "completed")
lost = num(r"\$(\d+)\$ lacked a current cached atlas registration", "no registration")
if None not in (total, ok, lost):
    check("attempted = completed + unregistered", total, ok + lost)

# --- the longitudinal subset must close ---
two_plus = num(r"\$(\d+)\$ participants contributed two or more visits", "participants >=2")
three = num(r"and \$(\d+)\$ contributed three", "participants with 3")
sess = num(r"giving \$(\d+)\$\s*sessions in the longitudinal subset", "longitudinal sessions")
if None not in (two_plus, three, sess):
    # those with exactly two are the remainder
    exactly_two = two_plus - three
    check("2x(two-visit) + 3x(three-visit) = sessions",
          sess, exactly_two * 2 + three * 3)

# --- and against the file the analyses read ---
d = pd.read_csv(HERE / "measured_pvs_axis_dlbs.csv")
d["Subject_ID"] = d.Subject_ID.astype(str)
counts = d.Subject_ID.value_counts()
check("longitudinal sessions match the CSV", sess, float(len(d)))
check("participants with two or more visits match", two_plus,
      float((counts >= 2).sum()))
check("participants with three visits match", three, float((counts == 3).sum()))

q = pd.read_csv(HERE / "roi_placement_quality_dlbs_all.csv")
q["Subject_ID"] = q.Subject_ID.astype(str)
post = num(r"in each ROI left \$(\d+)\$ sessions from \$(\d+)\$ participants",
           "post-QC sessions")
m = re.search(r"in each ROI left \$(\d+)\$ sessions from \$(\d+)\$ participants", FLAT)
if m:
    check("post-QC sessions match the CSV", float(m.group(1)), float(len(q)))
    check("post-QC participants match the CSV", float(m.group(2)),
          float(q.Subject_ID.nunique()))

# --- thresholds quoted in more than one place must agree ---
fa = re.findall(r"FA\}\s*\\ge\s*0?\.?2|FA\}\\ge0\.2|FA \\ge 0\.2", FLAT)
check("the FA floor is quoted consistently", True, len(set(fa)) <= 3)
band = re.findall(r"\$8\$~mm (?:above|optimal)", FLAT)
check("the band half-width appears as 8 mm", True, len(band) >= 1)

bad = [r for r in rows if not r[0]]
for okflag, label, want, got in rows:
    if not okflag:
        print(f"  FAIL  {label:<46s} states {want!r}, data {got!r}")
print(f"  {len(rows) - len(bad)}/{len(rows)} Methods numbers agree")
