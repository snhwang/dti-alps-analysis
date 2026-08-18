"""
How much head rotation actually occurs, so the tolerance thresholds mean something.

rotation_tolerance.py says how far the classic index can be rotated before its
error exceeds a stated level. That is only useful next to the rotation real
cohorts contain.

Head pose is taken from the subject-to-template affine (FLIRT .mat), not from
the tract-to-scanner angles. Those two are different quantities and conflating
them is easy: the tract angles contain the participant's own anatomy as well as
their position in the scanner, while the affine's rotation is head pose relative
to the template and is the thing a differently positioned acquisition changes.

The affine is decomposed by polar decomposition, A = R S, taking R as the
nearest rotation to the linear part. Shear and scale go into S and are
discarded. Pitch, roll and yaw are then read off R as intrinsic x-y-z Euler
angles, and total rotation is the angle of R about its own axis,
theta = arccos((tr(R) - 1) / 2).

DLBS was acquired obliquely and retains genuine positioning variation. HCP-A is
anatomically aligned during preprocessing, so its residual should be small and
serves as a check that the measurement is doing what it claims.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from data_paths import winpath

import atomic_io  # noqa: F401  writes become atomic on import

HERE = Path(__file__).resolve().parent
DIFF = HERE.parent.parent / "diffusion"
OUT = winpath("Q:/dti_output")


def decompose(mat: np.ndarray):
    """Polar decomposition of the linear part, then Euler angles of the rotation."""
    A = mat[:3, :3]
    U, _, Vt = np.linalg.svd(A)
    R = U @ Vt
    if np.linalg.det(R) < 0:            # keep it a proper rotation
        U[:, -1] *= -1
        R = U @ Vt
    total = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
    # intrinsic x-y-z: pitch about x, roll about y, yaw about z
    roll = float(np.degrees(np.arcsin(np.clip(-R[2, 0], -1, 1))))
    if abs(R[2, 0]) < 0.9999:
        pitch = float(np.degrees(np.arctan2(R[2, 1], R[2, 2])))
        yaw = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    else:
        pitch = float(np.degrees(np.arctan2(-R[1, 2], R[1, 1])))
        yaw = 0.0
    return pitch, roll, yaw, total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=["dlbs", "hcpa", "tn"], default="dlbs")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if args.cohort == "tn":
        # The trigeminal derivatives are laid out by session directory with the
        # BIDS id in metadata.json, not by a spheres CSV, so the session list is
        # built by walking them. Follow-up sessions are excluded as elsewhere.
        import json
        root = winpath("M:/ds005713-derivatives/dti_output/ds005713_preproc")
        recs = []
        for sd in sorted(d for d in root.iterdir() if d.is_dir()):
            meta = sd / "metadata.json"
            if not meta.exists():
                continue
            name = json.load(open(meta)).get("name", "")
            if not name or name.endswith("fu"):
                continue
            recs.append({"Subject_ID": name, "Visit": "1",
                         "DTI_Session_ID": sd.name, "status": "ok"})
        src = pd.DataFrame(recs)
    elif args.cohort == "dlbs":
        src = pd.read_csv(DIFF / "DLBS" / "dlbs_alps_spheres_5mm.csv")
        src["Visit"] = src["Session"]
    else:
        src = pd.read_csv(DIFF / "HCP" / "hcpa_alps_spheres_5mm.csv")
    src = src[src.status == "ok"].copy()
    if args.limit:
        src = src.head(args.limit)

    rows = []
    for i, r in enumerate(src.itertuples(), 1):
        _root = (winpath("M:/ds005713-derivatives/dti_output/ds005713_preproc")
                 if args.cohort == "tn" else OUT)
        m = _root / r.DTI_Session_ID / "processed" / "atlas" / "subject_to_mni_affine.mat"
        if not m.exists():
            continue
        try:
            A = np.loadtxt(m)
            if A.shape != (4, 4):
                continue
        except Exception:
            continue
        pitch, roll, yaw, total = decompose(A)
        rows.append({"Subject_ID": r.Subject_ID, "Visit": r.Visit,
                     "pitch": pitch, "roll": roll, "yaw": yaw, "total": total})
        if i % 200 == 0:
            print(f"  {i}/{len(src)}", flush=True)

    d = pd.DataFrame(rows)
    # TN has one session per participant and is keyed by BIDS id downstream.
    # Rename only the written copy so the summary below still works.
    _w = (d.rename(columns={"Subject_ID": "BIDS_ID"}).drop(columns=["Visit"])
          if args.cohort == "tn" else d)
    _w.to_csv(HERE / f"head_rotation_{args.cohort}.csv", index=False)
    _key = "Subject_ID"   # d keeps the internal schema; only the written copy is renamed
    print(f"\n{args.cohort}: {len(d)} sessions, {d[_key].nunique()} participants\n")

    print(f"{'axis':<10s} {'|median|':>9s} {'SD':>7s} {'p90':>7s} {'p95':>7s} {'max':>7s}")
    for c in ("pitch", "roll", "yaw"):
        a = d[c].abs()
        print(f"{c:<10s} {a.median():9.1f} {d[c].std():7.2f} {a.quantile(.90):7.1f} "
              f"{a.quantile(.95):7.1f} {a.max():7.1f}")
    t = d["total"]
    print(f"{'total':<10s} {t.median():9.1f} {t.std():7.2f} {t.quantile(.90):7.1f} "
          f"{t.quantile(.95):7.1f} {t.max():7.1f}")

    print("\nfraction of sessions exceeding a given total rotation")
    for lvl in (5, 8, 10, 12, 15, 20):
        print(f"  > {lvl:2d} deg: {100 * (t > lvl).mean():5.1f}%")

    # Between-visit repositioning, which is what a longitudinal study sees.
    rep = []
    for sid, g in d.groupby("Subject_ID"):
        if len(g) < 2:
            continue
        g = g.sort_values("Visit")
        for a, b in zip(g.itertuples(), g.iloc[1:].itertuples()):
            rep.append(abs(b.total - a.total))
    if rep:
        rep = np.array(rep)
        print(f"\nbetween-visit change in total rotation ({len(rep)} visit pairs)")
        print(f"  median {np.median(rep):.1f}  p90 {np.percentile(rep, 90):.1f}  "
              f"max {rep.max():.1f} deg")


if __name__ == "__main__":
    main()
