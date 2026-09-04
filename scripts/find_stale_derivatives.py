"""Which outputs predate the source they were computed from?

The placement was changed to 5 mm spheres and then reverted to the warped
masks. The revert restored the warped source files by renaming backups into
place, which preserves their modification time. Anything computed from the
canonical filename while it held sphere data therefore looks current by
timestamp and is not.

phenotype_sweep.csv was found this way: regenerating it from the restored
source changed every one of its 219 rows, moved a survivor count, and turned
the ordering statistic from p=0.005 to p=0.32.

This lists every output whose script reads a placement-dependent source,
together with whether the output is older than the reverted source. Older is
not proof of staleness, since a rename can make the source look old too, so
the only reliable test is to regenerate and diff. What this gives is the list
worth regenerating, in dependency order.

    python find_stale_derivatives.py
"""
from __future__ import annotations

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent

# The files whose contents depend on where the measurement region is placed.
PLACEMENT_SOURCES = [
    "measured_pvs_axis_hcpa_b1500_all.csv",
    # measured_pvs_axis_hcpa_b1500.csv was here. It held 234 rows from 100
    # participants where the same selection rule yields 1525 from 628, and its
    # only reader was phenotype_validation.py, which phenotype_sweep.py
    # supersedes. Both are in archive/.
    "measured_pvs_axis_dlbs.csv",
    "comparators_hcpa.csv",
    "comparators_dlbs.csv",
]


def outputs_of(script: Path) -> list[str]:
    t = script.read_text(encoding="utf-8", errors="ignore")
    return sorted(set(re.findall(r"to_csv\(\s*HERE\s*/\s*[\"']([^\"']+\.csv)", t)
                      + re.findall(r"to_csv\(\s*[\"']([^\"']+\.csv)", t)))


def main() -> None:
    src_mtime = {}
    for s in PLACEMENT_SOURCES:
        p = HERE / s
        if p.exists():
            src_mtime[s] = p.stat().st_mtime

    rows = []
    for script in sorted(HERE.glob("*.py")):
        t = script.read_text(encoding="utf-8", errors="ignore")
        reads = [s for s in PLACEMENT_SOURCES if s in t]
        if not reads:
            continue
        newest_src = max(src_mtime[s] for s in reads if s in src_mtime) \
            if any(s in src_mtime for s in reads) else 0
        for out in outputs_of(script):
            p = HERE / out
            if not p.exists():
                rows.append((script.name, out, None, reads))
                continue
            rows.append((script.name, out, p.stat().st_mtime - newest_src, reads))

    print(f"{len(rows)} outputs derived from a placement-dependent source\n")
    print(f"  {'output':<44s} {'vs source':>11s}  script")
    for script, out, delta, reads in sorted(rows, key=lambda r: (r[2] is None, r[2] or 0)):
        if delta is None:
            tag = "MISSING"
        elif delta < 0:
            tag = f"{delta/3600:+.1f} h"
        else:
            tag = f"{delta/3600:+.1f} h"
        mark = "  <-- older than source" if (delta is not None and delta < 0) else ""
        print(f"  {out:<44s} {tag:>11s}  {script}{mark}")

    print("\nTimestamps cannot settle this, because the revert renamed backups")
    print("into place and a rename keeps the old time. Regenerate and diff.")


if __name__ == "__main__":
    main()
