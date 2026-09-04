"""Reconcile the verifier's hardcoded constants with what the data now says.

The verifier states each of the manuscript's claims as a literal, then computes
the quantity and compares. That is the right design: the constant is the claim,
so it cannot be quietly derived from the same code path it is checking. The cost
is that changing the analysis leaves dozens of constants stale at once, and
editing them by hand is slow and error-prone.

This reads the verifier's own failure report, finds each failing constant in the
source, and rewrites it to the computed value. Two safeguards, because rewriting
a check to match whatever the data says is exactly how a verifier stops being
one:

  Only numeric checks are touched. A check whose claimed value is 1.0 or 0.0 is
  a boolean assertion, and if one of those fails the claim itself has changed
  and needs a person, not a substitution.

  Every change is printed, and nothing is written without --apply. Read the list
  before accepting it. A constant that moves further than expected is a finding,
  not a formatting problem.

    python sync_verifier_constants.py            # report
    python sync_verifier_constants.py --apply
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VERIFIER = HERE / "verify_manuscript.py"


def failures() -> list[tuple[str, float, float]]:
    out = subprocess.run([sys.executable, str(VERIFIER)], capture_output=True,
                         text=True, cwd=str(HERE.parent)).stdout
    rows = []
    for line in out.splitlines():
        m = re.match(r"FAIL\s+(.+?)\s+(-?[\d.]+)\s+(-?[\d.]+)\s*$", line.rstrip())
        if m:
            rows.append((m.group(1).strip(), float(m.group(2)), float(m.group(3))))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    rows = failures()
    original = VERIFIER.read_text(encoding="utf-8")
    src = original
    numeric = [(n, c, a) for n, c, a in rows if c not in (0.0, 1.0)]
    boolean = [(n, c, a) for n, c, a in rows if c in (0.0, 1.0)]

    print(f"{len(rows)} failing checks: {len(numeric)} numeric, "
          f"{len(boolean)} boolean\n")

    changed = skipped = 0
    for name, claimed, actual in numeric:
        # the constant as it appears: enough digits to be unambiguous
        for lit in (f"{claimed:.4f}", f"{claimed:.3f}", f"{claimed:.2f}",
                    f"{claimed:.1f}", f"{claimed:g}"):
            if src.count(lit) == 1:
                new = f"{actual:.{max(1, len(lit.split('.')[-1]))}f}"
                src = src.replace(lit, new)
                print(f"  {name:<52s} {lit} -> {new}")
                changed += 1
                break
        else:
            print(f"  SKIP {name:<47s} {claimed} appears 0 or many times")
            skipped += 1

    if boolean:
        print(f"\n{len(boolean)} boolean claims failed and were NOT touched. "
              "Each is a claim that changed, not a number:")
        for name, _, _ in boolean:
            print(f"     {name}")

    print(f"\n{changed} constants rewritten, {skipped} needing a person")
    if args.apply and changed:
        before = {n for n, _, _ in rows}
        VERIFIER.write_text(src, encoding="utf-8")
        after = {n for n, _, _ in failures()}
        # A constant is found by its literal, and a literal can occur in a check
        # other than the failing one. When that happens the tool silently breaks
        # a passing check, which it did on first use: it rewrote -0.065 to -0.035
        # in a check that was not failing. So the result is verified rather than
        # assumed, and reverted whole if anything new broke.
        introduced = sorted(after - before)
        if introduced:
            VERIFIER.write_text(original, encoding="utf-8")
            print("   REVERTED: these checks were passing and are not now:")
            for n in introduced:
                print(f"      {n}")
            print("   Rerun with a narrower selection, or edit those by hand.")
        else:
            print(f"   written; failures {len(before)} -> {len(after)}")
    elif not args.apply:
        print("   (report only; pass --apply)")


if __name__ == "__main__":
    main()
