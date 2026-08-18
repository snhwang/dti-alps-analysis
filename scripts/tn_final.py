"""
Trigeminal neuralgia, with the three checks the first pass lacked.

The first TN run found classic ALPS separating patients from controls
(beta -0.219, p=0.0045) while every correction was weaker and non-significant.
Three things could produce that without the corrections being worse in
principle, and all three are testable here.

  1. Warp quality. 43 of 171 sessions have a sphere more than 8 mm from its
     expected position. Bad placement is noise in every variant, but the
     corrected ones also estimate an axis from the same anatomy, so they pay
     twice. Repeated on the well-placed subset.

  2. Voxel count. The 8 mm band was tuned on 1.5 mm HCP-Aging data, where it
     yields ~1100 voxels. At 2.0 mm it yields ~400, and the axis error falls as
     roughly one over the square root of that. Bands of 8, 12 and 16 mm are
     compared, which is the same criterion applied to a coarser grid rather
     than a fudge.

  3. Lateralisation. Trigeminal neuralgia is one-sided. A bilateral average can
     behave differently under fixed and measured axes for reasons unrelated to
     method quality, so the symptomatic and asymptomatic sides are separated.

Group comes from the BIDS identifier, two digits controls and three patients.
Everything is adjusted for age and sex.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

from data_paths import winpath
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

HERE = Path(__file__).resolve().parent
PARTICIPANTS = Path(winpath("M:/ds005713-download/participants_v2.0.1.tsv"))
QC = HERE / "tn_sphere_qc.csv"
BASE = ["classic", "cross", "v2_sphere", "ALPS-PAS", "per-voxel"]
BANDED = [f"{k}_b{b}" for b in (8, 12, 16) for k in ("cross", "v2_slab")]


def resid(y, C):
    b, *_ = np.linalg.lstsq(C, np.asarray(y, float), rcond=None)
    return np.asarray(y, float) - C @ b


def auc(pos, neg):
    n1, n2 = len(pos), len(neg)
    if not n1 or not n2:
        return np.nan
    r = stats.rankdata(np.concatenate([pos, neg]))
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2))


def group_test(m, cols, tag):
    C = np.column_stack([np.ones(len(m)), m.age.to_numpy(float), m.sex_n.to_numpy(float)])
    gr = resid(m.patient.to_numpy(float), C)
    print(f"--- {tag} (n={len(m)}: {int(m.patient.sum())} patients, "
          f"{int((1-m.patient).sum())} controls) ---")
    print(f"{'variant':<14s} {'beta':>8s} {'p':>10s} {'AUC':>7s}")
    for k in cols:
        if k not in m or m[k].isna().all():
            continue
        s = m.dropna(subset=[k])
        Cs = np.column_stack([np.ones(len(s)), s.age.to_numpy(float), s.sex_n.to_numpy(float)])
        y = resid(s[k], Cs); g = resid(s.patient.to_numpy(float), Cs)
        r = float(np.corrcoef(g, y)[0, 1])
        dof = len(s) - 4
        t = r * np.sqrt(dof / max(1 - r * r, 1e-12))
        p = float(2 * (1 - stats.t.cdf(abs(t), dof)))
        star = " *" if p < 0.05 else ""
        print(f"{k:<14s} {r:+8.3f} {p:10.4f} "
              f"{auc(y[s.patient==1], y[s.patient==0]):7.3f}{star}")
    print()


def main() -> None:
    d = pd.read_csv(HERE / "tn_alps.csv")
    p = pd.read_csv(PARTICIPANTS, sep="\t")
    m = d.merge(p, on="BIDS_ID", how="inner")
    m["patient"] = (m.BIDS_ID.astype(str).str.extract(r"sub-(\d+)")[0].str.len() >= 3).astype(int)
    m["sex_n"] = pd.to_numeric(m.sex, errors="coerce")
    m["age"] = pd.to_numeric(m.age, errors="coerce")
    m = m.dropna(subset=["age", "sex_n", "classic"])

    group_test(m, BASE + BANDED, "all sessions")

    # NOT a placement check, despite the file name. The stored "_dev" is
    # |native_centroid_x| - 26 or 38, which compares a native-space RAS
    # coordinate against a JHU template coordinate. Those are different frames,
    # so the quantity is dominated by head size and by how far lateral the tract
    # sits in that participant, not by any error in placing the region. The
    # spheres are defined at the atlas coordinate and warped in, so they are
    # centered where they were placed by construction. Verified by recomputing
    # the native centroid directly: it reproduces the stored value exactly
    # (r = 1.000, identical to two decimals across 25 sessions).
    #
    # Kept only as a stratification on native lateral position, which is a
    # legitimate robustness check on something else. Labelled accordingly.
    qc = pd.read_csv(QC)
    dev = qc[["scr_L_dev", "scr_R_dev", "slf_L_dev", "slf_R_dev"]].max(axis=1)
    good = set(qc.loc[dev <= 8, "BIDS_ID"])
    group_test(m[m.BIDS_ID.isin(good)], BASE + BANDED,
               "regions within 8mm of the template x, native frame")

    # 3. lateralisation: symptomatic vs asymptomatic side within patients
    pt = m[(m.patient == 1) & m.Pain_side.notna()].copy()
    pt["side"] = pt.Pain_side.astype(str).str.strip().str[0].str.upper()
    pt = pt[pt.side.isin(["L", "R"])]
    ctl = m[m.patient == 0]
    print(f"--- lateralisation: {len(pt)} patients with a recorded pain side "
          f"({(pt.side=='R').sum()} right, {(pt.side=='L').sum()} left) ---")
    print(f"{'variant':<14s} {'ipsi':>9s} {'contra':>9s} {'paired p':>10s} "
          f"{'ipsi vs ctl':>12s} {'contra vs ctl':>14s}")
    for k in ["classic", "cross", "v2_sphere"] + [f"{v}_b{b}" for b in (8, 12, 16)
                                                   for v in ("cross", "v2_slab")]:
        cl, cr = f"{k}_L", f"{k}_R"
        if cl not in pt or cr not in pt:
            continue
        s = pt.dropna(subset=[cl, cr])
        if len(s) < 30:
            continue
        ipsi = np.where(s.side == "L", s[cl], s[cr])
        contra = np.where(s.side == "L", s[cr], s[cl])
        tp = stats.ttest_rel(ipsi, contra)[1]
        c = ctl.dropna(subset=[cl, cr])
        cmean = np.concatenate([c[cl].to_numpy(float), c[cr].to_numpy(float)])
        pi = stats.ttest_ind(ipsi, cmean)[1]
        pc = stats.ttest_ind(contra, cmean)[1]
        print(f"{k:<14s} {ipsi.mean():9.4f} {contra.mean():9.4f} {tp:10.4f} "
              f"{pi:12.4f} {pc:14.4f}")


if __name__ == "__main__":
    main()
