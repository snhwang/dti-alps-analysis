"""
How often does the automated region land in tissue that does not match its tract?

The DEC quality-control images showed a real problem that the direction-estimation
fix does not address. Moving the axis estimate onto the tract-label band stopped
the two direction estimates contaminating each other, but the *measurement*
sphere still sits wherever the atlas warp puts it, and still averages whatever
tissue is inside it. In a subset of sessions the association sphere sits partly
on left-right oriented tissue, which is not what the SLF looks like and not
where a human rater would have put it.

The manuscript currently reports only a cohort-level figure, that about 19% of
association-region voxels are left-right dominant. That is an average, and an
average cannot distinguish "every session is slightly contaminated" from "most
sessions are clean and some are badly placed". Those two have different
consequences, so this measures it per session.

Two definitions of "wrong direction" are computed, because they answer different
questions and neither alone is sufficient:

  off_axis    the voxel's dominant SCANNER axis is not the expected one
              (x for red, y for green, z for blue). This is what the DEC images
              show, so it is the one that corresponds to the visual inspection.
              It is not orientation-invariant, so a tilted head inflates it.

  off_tract   the voxel's principal direction is more than THRESH degrees from
              the slab-derived direction for its own tract. This is invariant to
              head position, so it isolates genuine tissue heterogeneity from
              head tilt. This is the defensible screening criterion.

For each session the index is recomputed with the off-tract voxels removed, so
the cost of screening can be compared against the cost of leaving them in.

Usage:
    python roi_placement_quality.py --cohort hcpa --limit 40
    ALPS_TENSOR_SUFFIX=_b1500 python roi_placement_quality.py --cohort hcpa
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from data_paths import winpath

import atomic_io  # noqa: F401  writes become atomic on import

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from estimator_variants import directional_diffusivity
from direction_estimators import weights_for, principal, align, X, Y, Z
from alps_common import parse_age

HERE = Path(__file__).resolve().parent
DIFF = HERE.parent.parent / "diffusion"
OUT = winpath("Q:/dti_output")
SLAB_MM = 8.0
FA_MIN = 0.2
THRESH = 45.0          # degrees from the tract direction before a voxel is "off tract"
SHELL = os.environ.get("ALPS_TENSOR_SUFFIX", "")

COLS = ["Subject_ID", "Visit", "Age",
        "scr_off_axis", "slf_off_axis", "slf_red",
        "scr_off_tract", "slf_off_tract",
        "theta_scr", "theta_slf", "theta_pvs", "theta_interfiber",
        "classic", "classic_screened", "refined_slab", "refined_slab_screened",
        "n_scr", "n_slf"]


def main() -> None:
    import nibabel as nib

    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=["hcpa", "dlbs"], default="hcpa")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None, help="override output filename")
    ap.add_argument("--all-sessions", action="store_true",
                    help="keep single-visit participants; needed for the "
                         "cross-sectional orientation-confound cohort")
    args = ap.parse_args()

    if args.cohort == "hcpa":
        src = pd.read_csv(DIFF / "HCP" / "hcpa_alps_spheres_5mm.csv")
        mot = pd.read_csv(DIFF / "HCP" / "hcpa_motion.csv")
        src = src.merge(mot[["Subject_ID", "Visit", "Eddy_Mean_RMS"]],
                        on=["Subject_ID", "Visit"], how="left")
        rms = pd.to_numeric(src.Eddy_Mean_RMS, errors="coerce")
        thr = float(np.nanpercentile(rms.dropna(), 76.4))
    else:
        src = pd.read_csv(DIFF / "DLBS" / "dlbs_alps_spheres_5mm.csv")
        mot = pd.read_csv(DIFF / "DLBS" / "dlbs_motion.csv")
        src = src.merge(mot[["DTI_Session_ID", "Eddy_Mean_RMS"]],
                        on="DTI_Session_ID", how="left")
        rms = pd.to_numeric(src.Eddy_Mean_RMS, errors="coerce")
        thr = 0.5
        src["Visit"] = src["Session"]

    src = src[(src.status == "ok") & (rms <= thr)].copy()
    src["Age"] = parse_age(src["Age"])
    src = src.dropna(subset=["Age"])
    if not args.all_sessions:
        counts = src.Subject_ID.value_counts()
        src = src[src.Subject_ID.isin(counts[counts >= 2].index)]
    if args.limit:
        rng = np.random.default_rng(20260809)
        keep = rng.choice(sorted(src.Subject_ID.unique()),
                          size=min(args.limit, src.Subject_ID.nunique()), replace=False)
        src = src[src.Subject_ID.isin(keep)]
    src = src.sort_values(["Subject_ID", "Visit"])
    print(f"cohort {args.cohort}: {len(src)} sessions, {src.Subject_ID.nunique()} participants\n")

    rows = []
    # Every skip below is counted and reported. A bare "except: continue" makes a
    # transient read on a network drive indistinguishable from a session that was
    # never eligible, and the session count then changes between runs with no
    # trace. That is how this file came to hold 1524 rows when 1525 sessions pass
    # every filter.
    skipped = {"missing files": [], "load failed": [], "no usable hemisphere": []}
    for i, r in enumerate(src.itertuples(), 1):
        sd = OUT / r.DTI_Session_ID / "processed"
        lab_p = sd / "atlas" / "jhu_labels_registered.nii.gz"
        sph_p = sd / "atlas" / "sphere_roi" / "sphere_roi_combined.nii.gz"
        if not (lab_p.exists() and sph_p.exists()):
            skipped["missing files"].append(f"{r.Subject_ID} {r.Visit}")
            continue
        try:
            limg = nib.load(str(lab_p))
            lab = limg.get_fdata().astype(int)
            sph = nib.load(str(sph_p)).get_fdata().astype(int)
            evals = nib.load(str(sd / f"tensor_eigenvalues{SHELL}.nii.gz")).get_fdata()
            evecs = nib.load(str(sd / f"tensor_eigenvectors{SHELL}.nii.gz")).get_fdata()
        except Exception as exc:
            skipped["load failed"].append(f"{r.Subject_ID} {r.Visit}: {type(exc).__name__} {exc}")
            continue

        md = evals.mean(axis=-1)
        nu = np.sqrt(((evals - md[..., None]) ** 2).sum(axis=-1))
        de = np.sqrt((evals ** 2).sum(axis=-1))
        fa = np.clip(np.sqrt(1.5) * np.divide(nu, de, out=np.zeros_like(nu), where=de != 0), 0, 1)

        ii, jj, kk = np.indices(lab.shape)
        zc = (limg.affine[2, 0] * ii + limg.affine[2, 1] * jj
              + limg.affine[2, 2] * kk + limg.affine[2, 3])
        xw = (limg.affine[0, 0] * ii + limg.affine[0, 1] * jj
              + limg.affine[0, 2] * kk + limg.affine[0, 3])

        def pack(mask):
            v1 = evecs[mask][:, :, 0]
            n = np.linalg.norm(v1, axis=1, keepdims=True); n[n == 0] = 1
            return {"v1": v1 / n, "fa": fa[mask], "evals": evals[mask], "evecs": evecs[mask]}

        acc = {k: [] for k in ("scr_off_axis", "slf_off_axis", "slf_red",
                               "scr_off_tract", "slf_off_tract",
                               "theta_scr", "theta_slf", "theta_pvs",
                               "theta_interfiber",
                               "classic", "classic_screened",
                               "refined_slab", "refined_slab_screened",
                               "n_scr", "n_slf")}
        # Hemisphere from world x, not voxel index: these volumes have a
        # negative x scale, so index < mid is world x > 0, the RIGHT side.
        for hemi, side, scr, slf in (("L", xw < 0, 26, 42), ("R", xw > 0, 25, 41)):
            mp_s = (sph == 1) & side & (fa >= FA_MIN)
            ma_s = (sph == 2) & side & (fa >= FA_MIN)
            if mp_s.sum() < 4 or ma_s.sum() < 4:
                continue
            P, A = pack(mp_s), pack(ma_s)

            # dominant scanner axis, the quantity the DEC images display
            dom_p = np.argmax(np.abs(P["v1"]), axis=1)
            dom_a = np.argmax(np.abs(A["v1"]), axis=1)
            acc["scr_off_axis"].append(float((dom_p != 2).mean()))   # expect z
            acc["slf_off_axis"].append(float((dom_a != 1).mean()))   # expect y
            acc["slf_red"].append(float((dom_a == 0).mean()))        # x-dominant, "red"

            # slab-derived tract directions, then off-tract fraction, orientation-free
            z0 = float(np.median(zc[sph > 0])) if (sph > 0).any() else 0.0
            band = np.abs(zc - z0) <= SLAB_MM
            mp_l = (lab == scr) & (fa >= FA_MIN) & band
            ma_l = (lab == slf) & (fa >= FA_MIN) & band
            if mp_l.sum() < 10 or ma_l.sum() < 10:
                continue
            L_p, L_a = pack(mp_l), pack(ma_l)
            vp = align(principal(L_p["v1"], weights_for("cl", L_p)), Z)
            va = align(principal(L_a["v1"], weights_for("cl", L_a)), Y)

            ang_p = np.degrees(np.arccos(np.clip(np.abs(P["v1"] @ vp), 0, 1)))
            ang_a = np.degrees(np.arccos(np.clip(np.abs(A["v1"] @ va), 0, 1)))
            keep_p, keep_a = ang_p <= THRESH, ang_a <= THRESH
            acc["scr_off_tract"].append(float((~keep_p).mean()))
            acc["slf_off_tract"].append(float((~keep_a).mean()))
            acc["n_scr"].append(int(mp_s.sum())); acc["n_slf"].append(int(ma_s.sum()))

            p = np.cross(vp, va); p /= max(np.linalg.norm(p), 1e-12)
            op = np.cross(p, vp); op /= max(np.linalg.norm(op), 1e-12)
            oa = np.cross(p, va); oa /= max(np.linalg.norm(oa), 1e-12)

            # Scanner-to-anatomy deviation, from the SLAB directions. Estimating
            # these from the spheres is what produced the retracted 67 degree
            # inter-fibre angle, so they are computed here on the same basis the
            # method itself uses. Acute angle throughout, since eigenvectors are
            # antipodally symmetric.
            def acute(u, v):
                return float(np.degrees(np.arccos(np.clip(abs(np.dot(u, v)), 0, 1))))
            acc["theta_scr"].append(acute(vp, Z))
            acc["theta_slf"].append(acute(va, Y))
            acc["theta_pvs"].append(acute(p, X))
            acc["theta_interfiber"].append(acute(vp, va))

            def alps(Pd, Ad, u_num, u_dp, u_da):
                return ((directional_diffusivity(Pd["evals"], Pd["evecs"], u_num)
                         + directional_diffusivity(Ad["evals"], Ad["evecs"], u_num))
                        / (directional_diffusivity(Pd["evals"], Pd["evecs"], u_dp)
                           + directional_diffusivity(Ad["evals"], Ad["evecs"], u_da)))

            acc["classic"].append(alps(P, A, X, Y, Z))
            acc["refined_slab"].append(alps(P, A, p, op, oa))

            if keep_p.sum() >= 4 and keep_a.sum() >= 4:
                Ps = {k: v[keep_p] for k, v in P.items()}
                As = {k: v[keep_a] for k, v in A.items()}
                acc["classic_screened"].append(alps(Ps, As, X, Y, Z))
                acc["refined_slab_screened"].append(alps(Ps, As, p, op, oa))

        if not acc["classic"]:
            skipped["no usable hemisphere"].append(f"{r.Subject_ID} {r.Visit}")
            continue
        rec = {"Subject_ID": r.Subject_ID, "Visit": r.Visit, "Age": r.Age}
        for k, v in acc.items():
            rec[k] = float(np.mean(v)) if v else np.nan
        rows.append(rec)
        if i % 100 == 0:
            print(f"  {i}/{len(src)}", flush=True)

    n_skip = sum(len(v) for v in skipped.values())
    print()
    print(f"{len(rows)} sessions written, {n_skip} skipped of {len(src)} eligible")
    for reason, items in skipped.items():
        if not items:
            continue
        print(f"  {reason}: {len(items)}")
        for s in items[:10]:
            print(f"    {s}")
        if len(items) > 10:
            print(f"    ... and {len(items) - 10} more")
    if skipped["load failed"]:
        print("  A load failure is not an eligibility criterion. Re-run before "
              "trusting this output.")

    d = pd.DataFrame(rows)[COLS]
    name = args.out or f"roi_placement_quality_{args.cohort}{SHELL}.csv"
    d.to_csv(HERE / name, index=False)
    print(f"\n{len(d)} sessions -> {name}")

    print("\nfraction of measurement-ROI voxels not matching the tract")
    for c in ("scr_off_axis", "slf_off_axis", "slf_red", "scr_off_tract", "slf_off_tract"):
        q = d[c].quantile([.5, .75, .9, .95]).values
        print(f"  {c:<16s} median {q[0]:.3f}  p75 {q[1]:.3f}  p90 {q[2]:.3f}  p95 {q[3]:.3f}")

    for c, lab_ in (("slf_red", "SLF x-dominant"), ("slf_off_tract", "SLF off-tract")):
        for t in (0.25, 0.5):
            n = int((d[c] > t).sum())
            print(f"  sessions with {lab_} > {t:.0%}: {n} ({100*n/len(d):.1f}%)")


if __name__ == "__main__":
    main()
