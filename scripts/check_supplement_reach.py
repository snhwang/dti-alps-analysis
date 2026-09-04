r"""Can a reader get to every supplement section, and does anything ask for it?

Twenty-seven supplement sections is a lot to justify, and nothing checked that
any of them is reachable. A section the article never points at is one a reader
meets only by scrolling, and an editor weighing length will ask what it is for.

A section counts as reachable if the article references it directly or
references the part (\section) it sits under. That is the honest test: pointing
at "The Rotation Study" reaches its three subsections.

Separately, a section is a reviewer deliverable if a phrase from
check_reviewer_coverage lands inside it, which is a stronger justification than
a cross-reference.

    python check_supplement_reach.py

Reports only. A section that is neither reachable nor a deliverable is one to
either point at, or cut.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BS = chr(92)
TEX = (ROOT / "mri_revision.tex").read_text(encoding="utf-8")
CUT = TEX.index("{" + BS + "Large" + BS + "bfseries Supplementary material}")
ART, SUP = TEX[:CUT], TEX[CUT:]

sys.path.insert(0, str(HERE))
from check_reviewer_coverage import COVERAGE  # noqa: E402

# part (\section) -> its label, so a pointer at the part counts for its children
parts: list[tuple[int, str, str | None]] = []
for m in re.finditer(BS + BS + r"section\{([^}]*)\}", SUP):
    lab = re.search(BS + BS + r"label\{([^}]+)\}", SUP[m.end():m.end() + 120])
    parts.append((m.start(), m.group(1), lab.group(1) if lab else None))

subs = [(m.start(), m.group(1)) for m in
        re.finditer(BS + BS + r"subsection\{([^}]*)\}", SUP)]

rows, unreached = [], []
for k, (pos, name) in enumerate(subs):
    end = subs[k + 1][0] if k + 1 < len(subs) else len(SUP)
    body = SUP[pos:end]
    m = re.search(BS + BS + r"label\{(sec:[a-z0-9-]+)\}", body)
    lab = m.group(1) if m else None
    direct = ART.count("ref{" + lab + "}") if lab else 0
    owner = [p for p in parts if p[0] < pos]
    via = ART.count("ref{" + owner[-1][2] + "}") if owner and owner[-1][2] else 0
    asked = [pt for pt, (_, phrases) in COVERAGE.items()
             if any(p in body or p == name for p in phrases)]
    rows.append((name, direct, via, asked))
    if not direct and not via and not asked:
        unreached.append(name)

print(f"  {'section':<50s} {'direct':>7s} {'via part':>9s}  reviewer")
for name, direct, via, asked in rows:
    tag = asked[0].split()[0] if asked else ""
    print(f"  {name[:50]:<50s} {direct:>7d} {via:>9d}  {tag}")

print(f"\n  {len(rows)} sections, {len(unreached)} neither reachable nor asked for")
for n in unreached:
    print(f"    unjustified: {n}")
