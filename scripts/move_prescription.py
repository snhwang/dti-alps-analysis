"""Move the slice-prescription validation from Results into the appendix.

Two paragraphs inside "Head Position Covaries with Age" test whether an
anterior-commissure-to-posterior-commissure prescription does what the method
assumes: whether it tracks the head, and whether it reproduces between visits.
Neither is a finding about ageing. They defend the pose measurement, which is
what the appendix is for, and no reviewer asked for them.

The finding they sit between stays in Results. Prescription tracks age, and the
prescription and registration measures carry opposite signs because operators
angulate the slab to follow a tilted head. Both bear directly on the confound
and belong where the confound is argued.

The move is within one file, so every phrase the verifier checks is still
present and no cross-reference changes.

    python move_prescription.py --report
    python move_prescription.py --apply
"""
from __future__ import annotations

import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEX = HERE.parent / "mri_revision.tex"

STARTS = ("We tested whether it does what the method assumes.",
          "Nor does it reproduce.")
# placed before this appendix subsection, so it sits with the other
# measurement-quality material rather than opening the appendix
ANCHOR = r"\subsection{Choice of Measurement Region}"

HEADER = (r"\subsection{Whether the Slice Prescription Does What the Method Assumes}"
          "\n" r"\label{sec:prescription-check}" "\n\n"
          "The classic method is defined for data acquired along the "
          "anterior-commissure-to-posterior-commissure line, so that the scanner "
          "axes stand in for the anatomical ones. Section~\\ref{sec:posture} "
          "reports that prescription tracks age. Whether it does the job the "
          "method assumes is a separate question, and it is answered here "
          "because it concerns the quality of the pose measurement rather than "
          "the ageing result.\n\n")


def paragraphs(t: str):
    return t.split("\n\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    t = TEX.read_text(encoding="utf-8")
    paras = paragraphs(t)
    take, keep = [], []
    for p in paras:
        if any(p.lstrip().startswith(s) for s in STARTS):
            take.append(p)
        else:
            keep.append(p)

    print(f"found {len(take)} paragraph(s), {sum(len(p.split()) for p in take)} words")
    for p in take:
        print(f"   {' '.join(p.split())[:88]}...")
    if len(take) != len(STARTS):
        raise SystemExit("did not find exactly the expected paragraphs; not applying")
    if not args.apply:
        print("\n(report only; pass --apply)")
        return

    body = "\n\n".join(keep)
    if ANCHOR not in body:
        raise SystemExit("anchor subsection not found")
    body = body.replace(ANCHOR, HEADER + "\n\n".join(take) + "\n\n" + ANCHOR, 1)
    TEX.write_text(body, encoding="utf-8")
    print(f"\nmoved into the appendix ahead of {ANCHOR}")


if __name__ == "__main__":
    main()
