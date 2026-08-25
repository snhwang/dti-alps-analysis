"""Structural audit of the manuscript, separate from the numerical verifier.

verify_manuscript.py checks that the numbers in the text match the numbers in
the analysis outputs. It cannot see the things that go wrong when a section is
cut or a cohort removed: a label nobody references any more, a figure defined
and never called, a sentence describing an analysis that left the paper, the
same quantity printed two different ways in two places.

This finds those. Every check reports what it found rather than passing or
failing, because most of them need a human to decide whether the finding
matters.

    python audit_manuscript.py
"""
from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEX = HERE.parent / "mri_revision.tex"

# Terms that should be gone entirely, with what replaced them.
RETIRED = {
    "trigeminal": "cohort removed from the revision",
    "TN ": "trigeminal abbreviation",
    "neuralgia": "cohort removed from the revision",
    "ds001907": "Parkinson's download, never used in the paper",
    "Refined+": "kept for its rotation departure only",
}


def sections(t: str):
    """(name, body) for each \\section, in order."""
    out, parts = [], re.split(r"\\section\*?\{([^}]*)\}", t)
    for i in range(1, len(parts), 2):
        out.append((parts[i], parts[i + 1]))
    return out


def main() -> None:
    argparse.ArgumentParser().parse_args()
    t = TEX.read_text(encoding="utf-8")
    flat = " ".join(t.split())
    findings = 0

    def head(s):
        print(f"\n{'=' * 66}\n{s}\n{'=' * 66}")

    head("1. Labels and cross-references")
    labs = re.findall(r"\\label\{([^}]+)\}", t)
    refs = set(re.findall(r"\\(?:ref|autoref|eqref|cref)\{([^}]+)\}", t))
    dup = [l for l, n in Counter(labs).items() if n > 1]
    unused = sorted(set(labs) - refs)
    dangling = sorted(refs - set(labs))
    print(f"   {len(set(labs))} labels, {len(set(labs) & refs)} referenced")
    for l in dup:
        print(f"   DUPLICATE label: {l}")
        findings += 1
    for l in unused:
        print(f"   never referenced: {l}")
        findings += 1
    for r in dangling:
        print(f"   REFERENCED BUT UNDEFINED: {r}")
        findings += 1

    head("2. Floats defined but never called out")
    for env in ("figure", "table"):
        for m in re.finditer(r"\\begin\{" + env + r"\*?\}(.*?)\\end\{" + env
                             + r"\*?\}", t, re.S):
            lab = re.search(r"\\label\{([^}]+)\}", m.group(1))
            if lab and lab.group(1) not in refs:
                print(f"   {env} {lab.group(1)} is never referenced in the text")
                findings += 1
            elif not lab:
                cap = re.search(r"\\caption\{(.{0,60})", m.group(1), re.S)
                print(f"   {env} with no label: {(cap.group(1) if cap else '?')[:58]}...")
                findings += 1

    head("3. Graphics files: included but missing, or present but unused")
    inc = set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", t))
    for g in sorted(inc):
        hits = list(TEX.parent.glob(g)) + list(TEX.parent.glob(g + ".*"))
        if not hits:
            print(f"   MISSING on disk: {g}")
            findings += 1
    onwd = {p.stem for p in TEX.parent.glob("fig*.p*g")} | \
           {p.stem for p in TEX.parent.glob("fig*.pdf")}
    used = {Path(g).stem for g in inc}
    for o in sorted(onwd - used):
        print(f"   present but not included: {o}")
        findings += 1

    head("4. Retired terms")
    for term, why in RETIRED.items():
        n = flat.count(term)
        if n:
            print(f"   {n:3d}x  '{term}'  ({why})")
            for m in list(re.finditer(re.escape(term), flat))[:3]:
                print(f"        ...{flat[max(0, m.start() - 90):m.end() + 90]}...")
            findings += 1

    head("5. Sample sizes, every distinct count in the text")
    counts = defaultdict(list)
    for m in re.finditer(r"\$(\d{2,4})\$ (participants|sessions|visits)", flat):
        counts[(m.group(1), m.group(2))].append(
            flat[max(0, m.start() - 70):m.end() + 30])
    for (n, kind), ctx in sorted(counts.items(), key=lambda kv: -len(kv[1])):
        print(f"   {n:>5s} {kind:<13s} x{len(ctx)}")

    head("6. The same coefficient printed more than once")
    vals = defaultdict(list)
    for m in re.finditer(r"[-+]?0\.\d{3}", flat):
        vals[m.group(0)].append(flat[max(0, m.start() - 80):m.end() + 40])
    rep = {v: c for v, c in vals.items() if len(c) > 2}
    print(f"   {len(rep)} values appear three or more times; check they mean the same thing")
    for v, c in sorted(rep.items(), key=lambda kv: -len(kv[1]))[:6]:
        print(f"\n   {v}  x{len(c)}")
        for s in c[:3]:
            print(f"      ...{s[-105:]}...")

    head("7. Section balance")
    for name, body in sections(t):
        w = len(body.split())
        print(f"   {w:6d} words  {name}")

    head("8. Subsection lengths, for judging what earns its place")
    pat = re.compile(r"\\(section|subsection)\*?\{([^}]*)\}")
    ms = list(pat.finditer(t))
    total = 0
    for i, m in enumerate(ms):
        lvl, name = m.group(1), m.group(2)
        end = ms[i + 1].start() if i + 1 < len(ms) else len(t)
        w = len(t[m.end():end].split())
        total += w
        if lvl == "section":
            print(f"\n   ## {name}")
        else:
            print(f"      {w:5d}w  {name}")
    print(f"\n   body total, excluding front and back matter: about {total} words")

    print(f"\n{'=' * 66}\n{findings} structural findings above need a decision.\n")


if __name__ == "__main__":
    main()
