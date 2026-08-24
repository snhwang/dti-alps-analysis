"""Does head position still track total tau once everything else is adjusted?

pose_phenotype.py found that absolute pitch tracks total tau in HCP-A at
r=+0.181 given age, sex and BMI. That is the one result in this line of work
that would let the paper claim head position is non-random with respect to a
clinical variable rather than only to age, so it deserves the full set of
controls rather than the two it was found with.

Covariates are added in nested arms, each one a specific alternative
explanation rather than a generic adjustment:

    age, sex          the baseline. Pose rises with age and so does tau, so an
                      unadjusted association between them is guaranteed.
    + BMI             body habitus. A larger neck and shoulders change how a
                      head lies in a coil, and BMI is what pose tracks most
                      strongly in both cohorts.
    + site, scanner   HCP-A is a four-site study on six scanners. Sites differ
                      in how they position participants and in who they
                      recruit, so a site effect could produce both halves.
    + motion          tau tracks frailty and frailty tracks movement. A
                      participant who moves more may also lie differently, and
                      motion degrades the registration the pose is read from.
    + registration    the pose measure is the rotation of a registration, so a
                      registration that fits differently gives a different
                      rotation. Scale, anisotropy and shear are carried.

The last arm holds all of them at once. Anything surviving that is not age,
not sex, not body size, not site, not scanner, not motion and not registration
behaviour.

    python pose_tau_controls.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
DIFF = HERE.parent.parent / "diffusion"
AABC = DIFF / "HCP" / "AABC2_subjects_2026_02_05_14_29_11.csv"
TARGETS = ["tTau_Conc_pg_ml", "glucose", "bmi"]


def dummies(s: pd.Series) -> np.ndarray:
    """One-hot with the first level dropped, as a plain array."""
    d = pd.get_dummies(s.astype(str), drop_first=True)
    return d.to_numpy(float) if d.shape[1] else np.empty((len(s), 0))


def partial(x, y, C):
    ok = ~(np.isnan(x) | np.isnan(y) | np.isnan(C).any(axis=1))
    x, y, C = x[ok], y[ok], C[ok]
    if len(x) < 30:
        return np.nan, np.nan, len(x)
    A = np.column_stack([np.ones(len(C)), C])

    def rz(v):
        b, *_ = np.linalg.lstsq(A, v, rcond=None)
        return v - A @ b
    rx, ry = rz(x), rz(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return np.nan, np.nan, len(x)
    r = float(np.corrcoef(rx, ry)[0, 1])
    dof = len(x) - A.shape[1] - 1
    t = r * np.sqrt(dof / max(1 - r ** 2, 1e-12))
    return r, float(2 * stats.t.sf(abs(t), dof)), len(x)


def main() -> None:
    argparse.ArgumentParser().parse_args()

    hr = pd.read_csv(HERE / "head_rotation_hcpa.csv")
    hr["Subject_ID"] = hr.Subject_ID.astype(str)
    hr["Visit"] = hr.Visit.astype(str)
    hr["abs_pitch"] = hr.pitch.abs()

    d = pd.read_csv(HERE / "measured_pvs_axis_hcpa_b1500_all.csv")
    d["Subject_ID"] = d.Subject_ID.astype(str)
    d["Visit"] = d.Visit.astype(str)
    m = hr.merge(d[["Subject_ID", "Visit", "Age"]], on=["Subject_ID", "Visit"])

    sp = pd.read_csv(DIFF / "HCP" / "hcpa_alps_spheres_5mm.csv")
    sp["Subject_ID"] = sp.Subject_ID.astype(str)
    sp["Visit"] = sp.Visit.astype(str)
    m = m.merge(sp[["Subject_ID", "Visit", "site", "scanner"]],
                on=["Subject_ID", "Visit"], how="left")

    mo = pd.read_csv(DIFF / "HCP" / "motion_rms_n1379.csv").rename(
        columns={"subject_id": "Subject_ID", "visit": "Visit"})
    mo["Subject_ID"] = mo.Subject_ID.astype(str)
    mo["Visit"] = mo.Visit.astype(str)
    m = m.merge(mo[["Subject_ID", "Visit", "motion_rms", "pct_outliers"]],
                on=["Subject_ID", "Visit"], how="left")

    rq = pd.read_csv(HERE / "registration_quality_hcpa.csv")
    rq["Subject_ID"] = rq.Subject_ID.astype(str)
    rq["Visit"] = rq.Visit.astype(str)
    m = m.merge(rq[["Subject_ID", "Visit", "det", "aniso", "shear"]],
                on=["Subject_ID", "Visit"], how="left")

    m = m.sort_values(["Subject_ID", "Visit"]).groupby("Subject_ID").first().reset_index()

    a = pd.read_csv(AABC, low_memory=False)
    a["Subject_ID"] = a.id_event.astype(str).str.split("_").str[0]
    keep = [c for c in TARGETS if c in a.columns]
    ph = a.groupby("Subject_ID")[keep].first().reset_index()
    sx = a.groupby("Subject_ID")["sex"].first().reset_index()
    m = m.merge(ph, on="Subject_ID", how="inner").merge(sx, on="Subject_ID", how="left")
    m["sex_n"] = (m.sex.astype(str).str.upper().str[0] == "M").astype(float)
    print(f"{len(m)} participants\n")

    base = [m.Age.to_numpy(float), m.sex_n.to_numpy(float)]
    arms = [
        ("age+sex", np.column_stack(base)),
        ("+bmi", np.column_stack(base + [m.bmi.to_numpy(float)])),
        ("+site,scanner", np.column_stack(base + [m.bmi.to_numpy(float),
                                                  dummies(m.site),
                                                  dummies(m.scanner)])),
        ("+motion", np.column_stack(base + [m.bmi.to_numpy(float),
                                            dummies(m.site), dummies(m.scanner),
                                            m.motion_rms.to_numpy(float),
                                            m.pct_outliers.to_numpy(float)])),
        ("+registration", np.column_stack(base + [m.bmi.to_numpy(float),
                                                  dummies(m.site), dummies(m.scanner),
                                                  m.motion_rms.to_numpy(float),
                                                  m.pct_outliers.to_numpy(float),
                                                  m.det.to_numpy(float),
                                                  m.aniso.to_numpy(float),
                                                  m.shear.to_numpy(float)])),
    ]

    # The motion cache covers fewer sessions than the imaging, so the arms that
    # include it run on a smaller sample. Comparing them against the full-sample
    # baseline confounds adjustment with a change of sample, and the coefficient
    # can move either way for that reason alone. This arm is the baseline
    # restricted to the same participants, and it is the only fair comparator
    # for the ones below it.
    have_motion = m.motion_rms.notna() & m.pct_outliers.notna()
    arms.insert(3, ("age+sex|mot-sub", np.column_stack(base), have_motion))
    arms = [(a[0], a[1], a[2] if len(a) > 2 else None) for a in arms]

    rows = []
    for target in [t for t in TARGETS if t in m.columns]:
        print(f"=== absolute pitch against {target} ===")
        print(f"{'covariates':<18s} {'n':>5s} {'r':>8s} {'p':>9s}")
        for name, C, mask in arms:
            if target == "bmi" and name != "age+sex":
                continue
            x = m.abs_pitch.to_numpy(float)
            y = m[target].to_numpy(float)
            CC = C
            if mask is not None:
                x, y, CC = x[mask.to_numpy()], y[mask.to_numpy()], C[mask.to_numpy()]
            r, p, n = partial(x, y, CC)
            print(f"{name:<18s} {n:5d} {r:8.3f} {p:9.4f}")
            rows.append({"target": target, "arm": name, "n": n, "r": r, "p": p})
        print()

    pd.DataFrame(rows).to_csv(HERE / "pose_tau_controls.csv", index=False)
    print("   wrote pose_tau_controls.csv")


if __name__ == "__main__":
    main()
