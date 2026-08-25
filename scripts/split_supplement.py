"""Move named appendix subsections into a supplementary file.

Most of the appendix answers a specific reviewer request and belongs in the
article, where the reviewer will see it. A few subsections answer requests that
the revision has since made irrelevant: they defend the construction of the
refined index, and the paper no longer argues that the refined index is better,
only that every variant collapses to the same ratio. Those are provided in
supplementary material and the reply says so, rather than being dropped.

Cross-references are handled in both directions. The supplement loads xr and
points at the article's aux file, so references out of it still resolve. Any
reference running the other way, from the article into a moved subsection, is
reported so it can be rewritten before the move is applied.

    python split_supplement.py --report
    python split_supplement.py --apply
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEX = HERE.parent / "mri_revision.tex"
SUPP = HERE.parent / "mri_supplement.tex"

# Subsections whose question the revision retired, with the reviewer point each
# answers so the reply can say where the answer went.
MOVE = [("Rotation Simulation Protocol", "apparatus for a demonstration that is not needed"),
        ("The Size of the Orientation Error", "its headline is true by construction"),
        ("Consequences for Group Comparisons and Single-Patient Reads",
         "its conclusion now follows from the closed form")]

PREAMBLE = r"""\documentclass[3p]{elsarticle}
\usepackage{amsmath,amssymb,graphicx,booktabs,array,xr}
\externaldocument{mri_revision}
\newcommand{\tblwidth}{\linewidth}
\newcolumntype{L}{l}
\newcolumntype{C}{c}
\begin{document}

\begin{center}
{\Large\bfseries Supplementary material}\\[0.4em]
{\large Head Position, DTI-ALPS, and Radial Anisotropy}
\end{center}

\vspace{1em}

These analyses are reported here rather than in the article because the revision
made them unnecessary to its argument, and because two reviewers identified the
weakness in the central one.

The rotation experiment imposes known rotations on already-aligned data and
shows that the corrected variants do not change. That is true by construction:
they are evaluated along axes that rotate with the tensor, and the invariance
was verified to machine precision rather than needing a simulation to reveal it.
Reviewers 1 and 4 both made this point, and we agree with it.

The quantities that do matter are now obtained without it. That pitch lowers the
index and cannot raise it, and that roll and yaw are an order smaller and of
undetermined sign, follow in closed form from the two-region geometry. The size
of the effect in real heads is measured directly, from the change in the index
between visits at which a participant was genuinely repositioned, and from the
share of the age association that pose adjustment removes. That groups differ in
head position is measured in the cohort rather than imposed, by splitting on
body habitus.

The experiments are unchanged and complete, and a reader who wants the
dose-response curves will find them here. We are glad to return them to the
article if the reviewers prefer.

\renewcommand{\thesection}{S\arabic{section}}
\setcounter{section}{0}
\section{Supporting analyses retained from review}

"""


def blocks(t: str):
    """(title, start, end) for every subsection, in document order."""
    ms = list(re.finditer(r"\\subsection\{([^}]*)\}", t))
    out = []
    for i, m in enumerate(ms):
        end = ms[i + 1].start() if i + 1 < len(ms) else len(t)
        out.append((m.group(1), m.start(), end))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    t = TEX.read_text(encoding="utf-8")
    found = {title: (s, e) for title, s, e in blocks(t)}

    chunks, spans = [], []
    for title, why in MOVE:
        if title not in found:
            print(f"  NOT FOUND: {title}")
            continue
        s, e = found[title]
        chunks.append((title, why, t[s:e]))
        spans.append((s, e))

    moved = "".join(c for _, _, c in chunks)
    labs = set(re.findall(r"\\label\{([^}]+)\}", moved))
    rest = t
    for s, e in sorted(spans, reverse=True):
        rest = rest[:s] + rest[e:]
    into = sorted({r for r in re.findall(r"\\ref\{([^}]+)\}", rest) if r in labs})

    print(f"moving {len(chunks)} subsection(s), {len(moved.split())} words\n")
    for title, why, c in chunks:
        print(f"   {title}  ({len(c.split())}w, answers {why})")
    print(f"\nlabels defined in the moved text: {sorted(labs)}")
    print(f"article references pointing INTO the moved text: {len(into)}")
    for r in into:
        print(f"   !! {r} must be rewritten before applying")

    if not args.apply:
        print("\n(report only; pass --apply to write the files)")
        return
    if into:
        raise SystemExit("refusing to apply while references point into the moved text")

    SUPP.write_text(PREAMBLE + moved + "\n\\end{document}\n", encoding="utf-8")
    TEX.write_text(rest, encoding="utf-8")
    print(f"\nwrote {SUPP.name}, article now {len(rest.split())} words of source")


if __name__ == "__main__":
    main()
