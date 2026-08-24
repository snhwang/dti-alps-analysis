"""ALPS-PAS and the per-voxel variant in the two aging cohorts.

Both are published-method comparators, and Reviewer 4 asked for exactly this
comparison. Until now they existed only in the patient cohort, so their HCP-A
and DLBS cells in the variant and beyond-ratio tables were dashes. That was a
gap even before the patient cohort was removed, and afterwards it would leave
the two comparators with no association evidence at all.

Nothing new is computed here. The same alps_variants used for the patient
cohort is run over the aging sessions, whose derivative trees already carry
tensor eigenvalues, eigenvectors, JHU labels and the warped spheres.

The classic index is recomputed alongside as a check. It is already known from
the existing tables, so if the recomputed value does not reproduce it, the
comparators computed in the same pass should not be trusted either.

    python aging_cohort_comparators.py --cohort hcpa --jobs 8
    python aging_cohort_comparators.py --cohort dlbs --jobs 8
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DIFFREPO = HERE.parent.parent / "diffusion"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(DIFFREPO))
os.environ.setdefault("DTI_OUTPUT_DIR", str(DIFFREPO / "dti_output"))

from data_paths import winpath                                  # noqa: E402
from tn_alps import alps_variants                               # noqa: E402

OUT = Path(winpath("Q:/dti_output"))
WANT = ["ALPS-PAS", "per-voxel", "classic", "cross", "pv_perp", "anat_x"]


def one(rec):
    sd = OUT / str(rec["session"])
    try:
        v = alps_variants(sd)
    except Exception:                                           # noqa: BLE001
        return None
    if not v:
        return None
    return {"Subject_ID": rec["sid"], "Visit": rec["visit"],
            **{k: v[k] for k in WANT if k in v}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=["hcpa", "dlbs"], default="hcpa")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if args.cohort == "hcpa":
        # Two things must match the manuscript or the comparators are not
        # comparable with the variants already published. HCP-A has b=1500 and
        # b=3000 shells and the paper fits b=1500, whose tensors are written
        # with a suffix. And only the 1706 sessions that have those tensors are
        # in the published table, not all 2742 that passed sphere placement.
        # Getting either wrong shifts classic away from its printed value, which
        # is what the check at the end of this script exists to catch.
        os.environ["ALPS_TENSOR_SUFFIX"] = "_b1500"
        s = pd.read_csv(DIFFREPO / "HCP" / "hcpa_alps_spheres_5mm.csv")
        s["visit"] = s.Visit.astype(str)
        s["Subject_ID"] = s.Subject_ID.astype(str)
        pub = pd.read_csv(HERE / "measured_pvs_axis_hcpa_b1500_all.csv")
        keep = {(str(a), str(b)) for a, b in zip(pub.Subject_ID, pub.Visit)}
        s = s[[(a, b) in keep for a, b in zip(s.Subject_ID, s.visit)]]
    else:
        s = pd.read_csv(DIFFREPO / "DLBS" / "dlbs_alps_spheres_5mm.csv")
        s["visit"] = s.Session.astype(str)
    s = s[s.status == "ok"]
    recs = [{"sid": str(r.Subject_ID), "visit": r.visit,
             "session": r.DTI_Session_ID} for r in s.itertuples()]
    if args.limit:
        recs = recs[: args.limit]
    print(f"{args.cohort}: {len(recs)} sessions, {args.jobs} at a time\n")

    outp = HERE / f"comparators_{args.cohort}.csv"
    done = []
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        for i, r in enumerate(ex.map(one, recs), 1):
            if r:
                done.append(r)
            if i % 200 == 0:
                print(f"   {i}/{len(recs)} ({len(done)} ok)", flush=True)
                # Flushed as it goes, so an interrupted run keeps its work.
                pd.DataFrame(done).to_csv(outp, index=False)
    d = pd.DataFrame(done)
    d.to_csv(outp, index=False)
    print(f"\n{len(d)} sessions computed")

    # Cross-check against the classic values already in the manuscript's source
    # table. A mismatch means this pass is not comparable and the comparators
    # from it should not be used.
    f = ("measured_pvs_axis_hcpa_b1500_all.csv" if args.cohort == "hcpa"
         else "measured_pvs_axis_dlbs.csv")
    old = pd.read_csv(HERE / f)[["Subject_ID", "Visit", "classic", "pv_perp"]]
    old["Subject_ID"] = old.Subject_ID.astype(str)
    old["Visit"] = old.Visit.astype(str)
    d["Subject_ID"] = d.Subject_ID.astype(str)
    d["Visit"] = d.Visit.astype(str)
    j = d.merge(old, on=["Subject_ID", "Visit"], suffixes=("_new", "_old"))
    print(f"\n=== check against the existing table ({len(j)} matched) ===")
    for c in ("classic", "pv_perp"):
        a, b = j[f"{c}_new"], j[f"{c}_old"]
        ok = a.notna() & b.notna()
        r = float(np.corrcoef(a[ok], b[ok])[0, 1])
        print(f"   {c:<8s} r={r:.6f}  max|diff|={float((a[ok]-b[ok]).abs().max()):.6f}")
    print(f"\n   wrote comparators_{args.cohort}.csv")


if __name__ == "__main__":
    main()
