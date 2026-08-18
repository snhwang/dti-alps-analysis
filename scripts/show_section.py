"""Print a manuscript subsection as readable plain text.

For reviewing prose without the LaTeX. Strips markup, resolves cross-references
to bare labels, and replaces figures and tables with placeholders.

Usage:
    python show_section.py "Head Pose" "Head Position Covaries with Age"
"""

import re
import sys
import textwrap
from pathlib import Path

BS = chr(92)
TEX = Path(__file__).resolve().parent.parent / "mri_revision.tex"
t = TEX.read_text(encoding="utf-8")


def show(name):
    start = t.find(BS + "subsection{" + name + "}")
    if start < 0:
        return f"[{name}: not found]"
    rest = t[start + len(name) + 13:]
    ends = [i for i in (rest.find(BS + "subsection{"), rest.find(BS + "section{")) if i >= 0]
    body = rest[:min(ends)] if ends else rest

    body = re.sub(BS + BS + r"label\{[^}]*\}", "", body)
    body = re.sub(BS + BS + r"begin\{figure\}.*?" + BS + BS + r"end\{figure\}",
                  "\n[FIGURE]\n", body, flags=re.S)
    body = re.sub(BS + BS + r"begin\{table\}.*?" + BS + BS + r"end\{table\}",
                  "\n[TABLE]\n", body, flags=re.S)
    body = re.sub(BS + BS + r"(ref|cite)\{([^}]*)\}", r"[\2]", body)
    body = re.sub(BS + BS + r"[a-zA-Z]+\{([^}]*)\}", r"\1", body)
    for a, b in ((BS + "%", "%"), (BS + "circ", " deg"), (BS + "times", "x"),
                 (BS + "approx", "~"), (BS + "pm", "+/-"), (BS + ",", " "),
                 (BS + "emph", ""), (BS + "&", "&")):
        body = body.replace(a, b)
    body = re.sub(r"[${}]", "", body)
    body = re.sub(BS + BS + r"[a-zA-Z]+", "", body)

    out = []
    for para in body.strip().split("\n\n"):
        p = " ".join(para.split())
        if p:
            out.append(textwrap.fill(p, 94))
    return "\n\n".join(out)


for nm in sys.argv[1:]:
    print("=" * 94)
    print(nm.upper())
    print("=" * 94)
    print(show(nm))
    print()
