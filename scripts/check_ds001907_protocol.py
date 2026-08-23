"""Were ds001907 patients and controls scanned with the same diffusion protocol?

This is the question that decides whether the dataset can test a group
difference at all. If patients and controls were acquired differently, then any
group difference in a diffusion metric is confounded by acquisition, and a head
position analysis on top of it would be measuring the scanner rather than the
patient.

The check reads the sidecar JSON and the gradient tables straight from the git
tree, so it needs no downloaded imaging and runs in seconds.

    python check_ds001907_protocol.py
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

TAG = "3.0.2"
CLONE = Path(r"C:\tmp\ds001907git")
KEYS = ["EchoTime", "RepetitionTime", "MagneticFieldStrength", "Manufacturer",
        "ManufacturersModelName", "PhaseEncodingDirection", "FlipAngle",
        "SliceThickness", "PixelBandwidth", "SoftwareVersions"]


def git(*a):
    return subprocess.run(["git", *a], cwd=CLONE, capture_output=True,
                          text=True).stdout


from ds001907_common import group, assert_group_mapping  # noqa: E402


def main() -> None:
    argparse.ArgumentParser().parse_args()
    assert_group_mapping()
    files = [f.strip() for f in git("ls-tree", "-r", "--name-only", TAG).split("\n")
             if "/dwi/" in f.strip()]

    # --- sidecar parameters -------------------------------------------------
    rows = []
    for f in [x for x in files if x.endswith(".json")]:
        try:
            j = json.loads(git("show", f"{TAG}:{f}"))
        except Exception:                                       # noqa: BLE001
            continue
        sid = f.split("/")[0]
        rows.append({"subject": sid, "group": group(sid),
                     **{k: j.get(k) for k in KEYS}})
    js = pd.DataFrame(rows)
    print(f"{len(js)} sidecars: "
          f"{(js.group=='patient').sum()} patient, "
          f"{(js.group=='control').sum()} control\n")

    # A bare set difference overstates the problem: one odd scan out of forty
    # reads the same as a wholesale protocol split. Report how many scans hold
    # each value, and separate a value only one arm has from a value both have
    # in different proportions.
    print("=== acquisition parameters by group (scans per value) ===")
    exclusive = []
    for k in KEYS:
        if k not in js or js[k].isna().all():
            continue
        cnt = {g: d[k].astype(str).value_counts().to_dict()
               for g, d in js.groupby("group")}
        pv, cv = set(cnt.get("patient", {})), set(cnt.get("control", {}))
        only_p, only_c = pv - cv, cv - pv
        if only_p or only_c:
            exclusive.append((k, {v: cnt["patient"][v] for v in only_p},
                              {v: cnt["control"][v] for v in only_c}))
        mark = "  <-- one arm only" if (only_p or only_c) else ""
        print(f"  {k}{mark}")
        for g in ("patient", "control"):
            s = ", ".join(f"{v}x{n}" for v, n in sorted(cnt.get(g, {}).items()))
            print(f"      {g:8s} {s}")

    js.to_csv(Path(__file__).with_name("ds001907_protocol.csv"), index=False)

    # --- gradient tables ----------------------------------------------------
    print("\n=== gradient scheme by group ===")
    scheme = defaultdict(list)
    for f in [x for x in files if x.endswith(".bval")]:
        sid = f.split("/")[0]
        try:
            b = np.array([float(x) for x in git("show", f"{TAG}:{f}").split()])
        except ValueError:
            continue
        shells = tuple(sorted({int(round(v / 50) * 50) for v in b}))
        scheme[group(sid)].append((len(b), shells))

    for g, v in scheme.items():
        n_dirs = sorted({x[0] for x in v})
        shells = sorted({x[1] for x in v})
        print(f"  {g:8s} n={len(v):3d}  volumes per scan {n_dirs}")
        for s in shells:
            cnt = sum(1 for x in v if x[1] == s)
            print(f"            b-values {s}  in {cnt} scans")

    pat = {x for x in scheme["patient"]}
    ctl = {x for x in scheme["control"]}
    print("\n" + "=" * 64)
    if pat == ctl:
        print("Patients and controls share one diffusion scheme. A group")
        print("comparison is not confounded by acquisition.")
    else:
        print("The two arms DO NOT share a scheme.")
        print(f"  patient-only : {sorted(pat - ctl)}")
        print(f"  control-only : {sorted(ctl - pat)}")
        print("Any group difference here is confounded by acquisition, so the")
        print("dataset cannot support a patient-control diffusion contrast")
        print("without restricting to a scheme both arms share.")


if __name__ == "__main__":
    main()
