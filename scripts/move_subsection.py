"""Move a named subsection into the appendix, leaving a pointer behind.

The rule this serves: a section stays in the article if a reader who already
accepts the paper's two claims still needs it, and goes to the appendix if not.
The two claims are that head position confounds the index, and that the index is
bounded by and collapses to the eigenvalue ratio.

Applied to the template-reorientation comparison, most of that section asks
whether the closed-form correction matches an alternative correction. That was a
live question when the paper proposed a method. It is not one now, because the
paper argues every variant reduces to the same ratio and declines to recommend
any of them. One finding inside it does survive the test and is kept in Results:
reorientation removes posture but leaves the anatomical departure, so a
reoriented pipeline carries the same fiber contamination as an uncorrected one.

    python move_subsection.py --title "..." --anchor "..." --report
    python move_subsection.py --title "..." --anchor "..." --apply
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEX = HERE.parent / "mri_revision.tex"


def bounds(lines, title, which=1):
    """First line of the subsection and the line after its last.

    A title can legitimately appear twice, once in Methods describing how a
    thing was computed and once in Results reporting it. `which` selects
    between them, counting from the top of the document.
    """
    head = "\\subsection{" + title + "}"
    starts = [i for i, l in enumerate(lines) if l.strip() == head]
    if not starts:
        raise SystemExit(f"'{title}' not found")
    if which > len(starts):
        raise SystemExit(f"'{title}' occurs {len(starts)} time(s), asked for {which}")
    s = starts[which - 1]
    e = next((i for i in range(s + 1, len(lines))
              if lines[i].startswith("\\subsection{")
              or lines[i].startswith("\\section{")), len(lines))
    return s, e


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--title", required=True)
    ap.add_argument("--anchor", required=True,
                    help="appendix subsection to insert before")
    ap.add_argument("--pointer", default="",
                    help="text left in place of the moved section")
    ap.add_argument("--which", type=int, default=1,
                    help="which occurrence of the title, from the top")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    lines = TEX.read_text(encoding="utf-8").split("\n")
    s, e = bounds(lines, args.title, args.which)
    block = "\n".join(lines[s:e]).rstrip() + "\n"
    labs = set(re.findall(r"\\label\{([^}]+)\}", block))

    rest = lines[:s] + ([args.pointer, ""] if args.pointer else []) + lines[e:]
    rest_txt = "\n".join(rest)
    into = sorted({r for r in re.findall(r"\\ref\{([^}]+)\}", rest_txt)
                   if r in labs})

    print(f"'{args.title}': {len(block.split())} words, labels {sorted(labs)}")
    print(f"references from the rest of the article into it: {len(into)}")
    for r in into:
        print(f"   {r}  (kept resolvable, since the appendix is the same file)")
    if not args.apply:
        print("\n(report only; pass --apply)")
        return

    a_s, _ = bounds(rest, args.anchor)
    out = rest[:a_s] + block.split("\n") + [""] + rest[a_s:]
    TEX.write_text("\n".join(out), encoding="utf-8")
    print(f"\nmoved ahead of '{args.anchor}'")


if __name__ == "__main__":
    main()
