"""Which analyses measure inside which region, and does the manuscript use them?

Making the redrawn sphere the primary placement changed measured_pvs_axis and
tn_alps. Every other script that builds its own ALPS region kept the warped
mask, and nothing announced that, so the manuscript quietly came to hold numbers
from two different placements.

Finding them by eye failed twice. A first scan grepped for `sph == 1` and missed
vecreg_comparison, which spells the same operation `mask == code`. A second
assumed denominator_contamination honored ALPS_SPHERE_MM because it had been
passed the variable, when the script never reads it.

So this asks the two questions that matter, mechanically:

  does the script load the ALPS region mask and select voxels from it
  does it honor ALPS_SPHERE_MM, or is it pinned to the warped mask

and crosses the answer with whether the verifier reads that script's output,
which is the test of whether the manuscript depends on it. A script pinned to
the warped mask is not a fault by itself. Several are about the warped mask's
own behavior. It is a fault when the manuscript uses its numbers without saying
which placement produced them.

    python check_placement_consistency.py
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Scripts whose subject is the warped mask itself, so keeping it is correct.
BY_DESIGN = {
    "roi_placement_quality.py": "measures how the warped mask lands",
    "roi_variants.py": "compares region definitions against each other",
    "radius_robustness.py": "varies the radius, which the sphere fixes",
    "shifted_roi.py": "displaces the region deliberately",
    "manual_centroid_shift.py": "hand versus warped-atlas centroid",
    "hemisphere_slice_agreement.py": "geometry of the hand-drawn regions",
    "manual_pvs_axis.py": "measures inside the hand-drawn regions",
    "why_offtract.py": "asks where the warped region lands",
    "build_roi_comparison_figure.py": "draws the regions being compared",
    "tn_placement_frame_check.py": "retired with the patient cohort",
    "ds001907_alps.py": "retired with the patient cohort",
}

# Selecting voxels from the region mask, however the variable is spelled.
SELECT = re.compile(r"(sph|mask|roi|lab)\s*==\s*[12]\b")


def main() -> None:
    argparse.ArgumentParser().parse_args()
    vt = (HERE / "verify_manuscript.py").read_text(encoding="utf-8")
    reads = set(re.findall(r"['\"]([a-z0-9_]+\.csv)['\"]", vt))

    rows = []
    for f in sorted(HERE.glob("*.py")):
        src = f.read_text(encoding="utf-8", errors="ignore")
        if "sphere_roi_combined" not in src or not SELECT.search(src):
            continue
        aware = "ALPS_SPHERE_MM" in src
        outs = set(re.findall(r"['\"]([a-z0-9_]+\.csv)['\"]", src))
        outs |= {f"{f.stem}.csv"} | {f"{f.stem}_{c}.csv"
                                     for c in ("hcpa", "dlbs", "hcpa_b1500",
                                               "dlbs_all", "hcpa_b1500_all")}
        used = sorted(outs & reads)
        rows.append((f.name, aware, used))

    print(f"{len(rows)} scripts build an ALPS region and select voxels from it\n")

    print("=== honor the placement variable ===")
    for n, aware, used in rows:
        if aware:
            print(f"   {n}")

    print("\n=== pinned to the warped mask, by design ===")
    for n, aware, used in rows:
        if not aware and n in BY_DESIGN:
            print(f"   {n:<34s} {BY_DESIGN[n]}")

    print("\n=== pinned to the warped mask, and the manuscript uses the output ===")
    bad = [(n, u) for n, aware, u in rows if not aware and n not in BY_DESIGN and u]
    for n, used in bad:
        print(f"   {n}")
        for u in used:
            print(f"        -> {u}")
    if not bad:
        print("   none")

    print("\n=== pinned to the warped mask, output not used by the manuscript ===")
    for n, aware, used in rows:
        if not aware and n not in BY_DESIGN and not used:
            print(f"   {n}")

    print(f"\n{len(bad)} script(s) need either the placement rule or a caption "
          "saying which placement produced their numbers.\n")


if __name__ == "__main__":
    main()
