"""Which Discussion sentences restate Results rather than interpret them?

A Discussion earns its length by saying what the findings mean, what they do
not mean, and what a reader should do about them. It does not earn it by
repeating the findings. The habit is easy to fall into and hard to see while
writing, because each restatement feels like it is setting up the point that
follows.

Two signals find it mechanically. A number printed in the Discussion that was
already printed in Results is almost always a restatement, since interpretation
rarely needs the figure again. And a sentence sharing most of its distinctive
words with a Results sentence is the same sentence twice.

Neither is proof. A number may be repeated deliberately to anchor a comparison,
and some overlap is how prose connects. The output is a list to read, sorted so
the worst offenders come first.

    python find_restatement.py
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEX = HERE.parent / "mri_revision.tex"
STOP = set("""the a an and or of to in is are was were be been that this it its for with as by
on at from than then so not no but which where when what who whom whose have has had do does
did will would can could may might must shall should we our us they their them he she his her
one two both each every all any some more most less least other another same such only just
also very much many few first second third here there now already still yet".split()""".split())


def sentences(block: str):
    flat = " ".join(block.split())
    flat = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", flat)
    return [s.strip() for s in re.split(r"(?<=[.!?]) +", flat) if len(s.split()) > 6]


def keywords(s: str):
    w = re.findall(r"[a-zA-Z]{4,}", s.lower())
    return {x for x in w if x not in STOP}


def section(t: str, name: str) -> str:
    i = t.index("\\section{" + name + "}")
    j = min([t.index(m, i + 1) for m in ("\\section{", "\\appendix") if m in t[i + 1:]]
            or [len(t)])
    nxt = t.find("\\section{", i + 1)
    app = t.find("\\appendix", i + 1)
    ends = [x for x in (nxt, app) if x > 0]
    return t[i:min(ends) if ends else len(t)]


def main() -> None:
    argparse.ArgumentParser().parse_args()
    t = TEX.read_text(encoding="utf-8")
    res, dis = section(t, "Results"), section(t, "Discussion")
    rs, ds = sentences(res), sentences(dis)
    rnums = set(re.findall(r"[-+]?\d+\.\d+", res)) | set(re.findall(r"\d+\\%", res))
    rkeys = [keywords(s) for s in rs]

    print(f"Results {len(rs)} sentences, Discussion {len(ds)} sentences\n")

    hits = []
    for s in ds:
        dn = set(re.findall(r"[-+]?\d+\.\d+", s)) | set(re.findall(r"\d+\\%", s))
        shared_nums = dn & rnums
        k = keywords(s)
        best, bi = 0.0, -1
        for i, rk in enumerate(rkeys):
            if not rk or not k:
                continue
            j = len(k & rk) / len(k | rk)
            if j > best:
                best, bi = j, i
        score = best + 0.25 * len(shared_nums)
        if shared_nums or best > 0.30:
            hits.append((score, s, shared_nums, best, rs[bi] if bi >= 0 else ""))

    hits.sort(reverse=True, key=lambda h: h[0])
    print(f"{len(hits)} Discussion sentence(s) look like restatement, worst first:\n")
    for score, s, nums, best, match in hits[:14]:
        print(f"--- overlap {best:.2f}" + (f", repeats {sorted(nums)}" if nums else ""))
        print(f"    D: {s[:150]}")
        if best > 0.30:
            print(f"    R: {match[:150]}")
        print()


if __name__ == "__main__":
    main()
