"""
Final consistency check: manuscript claims against the data files that produce them.

Every number below is re-derived from source and compared with what the
manuscript states. This exists because several rounds of correction moved values
around, and eyeballing a 35-page document is not a check.

Run before submission. A FAIL means the manuscript and the data disagree.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np
from data_paths import winpath
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
TEX = (HERE.parent / "mri_revision.tex").read_text(encoding="utf-8")
sys.path.insert(0, str(HERE))
from estimator_variants import variance_components

results = []


def check(label, claimed, actual, tol=0.02):
    ok = abs(claimed - actual) <= tol * max(abs(actual), 1e-9) + 1e-9
    results.append((ok, label, claimed, actual))


def in_tex(pattern):
    """True if the literal string appears in the manuscript."""
    return pattern in TEX


# --- rotation, from the corrected slab cache ---
acc = pd.read_csv(HERE / "rotation_slab_accuracy.csv")
for sig, want in ((10, 5.2), (20, 13.5), (30, 20.3)):
    check(f"classic rotation error at {sig} deg", want,
          float(acc[acc.sigma == sig].classic.iloc[0]), tol=0.02)
check("refined flat", 3.67, float(acc.refined.iloc[0]))
check("refined+ flat", 3.50, float(acc["refined+"].iloc[0]))
check("per-voxel flat", 3.97, float(acc["per-voxel"].iloc[0]))

per = pd.read_csv(HERE / "rotation_slab_peraxis.csv")
for ax, want in (("pitch (x)", 7.91), ("roll (y)", 1.58), ("yaw (z)", 1.99)):
    check(f"classic {ax} at 15 deg", want,
          float(per[per.axis == ax].classic.iloc[0]))

grp = pd.read_csv(HERE / "rotation_slab_group.csv")
for tilt, want in ((10, -3.92), (15, -8.68), (20, -14.51)):
    check(f"spurious group difference at {tilt} deg", want,
          float(grp[grp.tilt_deg == tilt].classic.iloc[0]))

tol = pd.read_csv(HERE / "rotation_tolerance.csv")
pitch = tol[(tol["mode"] == "pitch (x)") & (tol.method == "classic")]
for col, want in (("tol_1pct", 4.0), ("tol_2pct", 6.3), ("tol_5pct", 11.0)):
    check(f"classic pitch tolerance {col}", want, float(pitch[col].iloc[0]), tol=0.03)

# --- reliability, post hemisphere fix ---
for tag, f, want_classic, want_slab in (
        ("HCP-A", "decoupled_roi_hcpa_b1500.csv", 0.957, 0.950),
        ("DLBS", "decoupled_roi_dlbs.csv", 0.594, 0.455)):
    d = pd.read_csv(HERE / f)
    lon = d[d.Subject_ID.isin(d.Subject_ID.value_counts()[lambda s: s >= 2].index)]
    check(f"{tag} classic ICC", want_classic,
          variance_components(lon.dropna(subset=["classic"]), "classic")["icc"], tol=0.01)
    check(f"{tag} refined_slab ICC", want_slab,
          variance_components(lon.dropna(subset=["refined_slab"]), "refined_slab")["icc"], tol=0.01)

# --- voxelwise measured axis, which is lambda2/lambda3 ---
for tag, f, want_icc, want_age in (
        ("HCP-A", "measured_pvs_axis_hcpa_b1500_all.csv", 0.950, -0.581),
        ("DLBS", "measured_pvs_axis_dlbs.csv", 0.545, -0.419)):
    d = pd.read_csv(HERE / f).dropna(subset=["pv_perp"])
    lon = d[d.Subject_ID.isin(d.Subject_ID.value_counts()[lambda s: s >= 2].index)]
    check(f"{tag} pv_perp ICC", want_icc,
          variance_components(lon, "pv_perp")["icc"], tol=0.01)
    s = d.dropna(subset=["Age"])
    check(f"{tag} pv_perp vs age", want_age,
          float(np.corrcoef(s.Age, s.pv_perp)[0, 1]), tol=0.02)
    # the selection effect on level: the ratio can never fall below one
    check(f"{tag} pv_perp minimum exceeds 1", 1.0,
          float(d.pv_perp.min() > 1.0) + 0.0, tol=1e-9)

# --- head pose, the paper's core claim ---
for tag, hf, af, want_pitch in (("DLBS", "head_rotation_dlbs.csv",
                                 "measured_pvs_axis_dlbs.csv", 0.332),
                                ("HCP-A", "head_rotation_hcpa.csv",
                                 "measured_pvs_axis_hcpa_b1500_all.csv", -0.028)):
    h, a = pd.read_csv(HERE / hf), pd.read_csv(HERE / af)
    for x in (h, a):
        x["Subject_ID"] = x.Subject_ID.astype(str); x["Visit"] = x.Visit.astype(str)
    m = (h.merge(a, on=["Subject_ID", "Visit"])
           .sort_values(["Subject_ID", "Visit"]).groupby("Subject_ID").first()
           .reset_index().dropna(subset=["Age", "pitch", "classic"]))
    check(f"{tag} pitch vs age", want_pitch,
          float(np.corrcoef(m.Age, m.pitch.abs())[0, 1]), tol=0.05)
    if tag == "DLBS":
        check("DLBS yaw vs age", 0.258, float(np.corrcoef(m.Age, m.yaw.abs())[0, 1]), tol=0.05)
        check("DLBS roll vs age", 0.036, float(np.corrcoef(m.Age, m.roll.abs())[0, 1]), tol=0.30)

        def beta(col, adj):
            z = lambda v: (np.asarray(v, float) - np.mean(v)) / np.std(v, ddof=1)
            cols = [np.ones(len(m)), z(m.Age)] + ([z(m.pitch.abs()), z(m.total)] if adj else [])
            return float(np.linalg.lstsq(np.column_stack(cols), z(m[col]), rcond=None)[0][1])
        drop = 100 * (1 - abs(beta("classic", True)) / abs(beta("classic", False)))
        check("DLBS classic age coefficient absorbed by pose (%)", 45.0, drop, tol=0.05)

# --- slice prescription: head pose from scanner metadata, which no brain
# property can influence, so the atrophy objection cannot apply to it ---
_sl = pd.read_csv(HERE / "slab_prescription_dlbs.csv")
_ag = pd.read_csv(HERE / "measured_pvs_axis_dlbs.csv")[["Subject_ID", "Visit", "Age"]]
for _x in (_sl, _ag):
    _x["Subject_ID"] = _x.Subject_ID.astype(str)
    _x["Visit"] = _x.Visit.astype(str)
_sm = (_sl.merge(_ag, on=["Subject_ID", "Visit"]).dropna()
          .sort_values(["Subject_ID", "Visit"]).groupby("Subject_ID").first().reset_index())
check("slab control participants", 156.0, float(len(_sm)), tol=0.01)
check("slab angulation median (deg)", 3.31, float(_sm.slab_tilt.median()), tol=0.02)
check("slab pitch vs age", -0.342, float(stats.pearsonr(_sm.Age, _sm.slab_pitch)[0]), tol=0.03)
check("affine pitch vs age, signed", 0.340, float(stats.pearsonr(_sm.Age, _sm.aff_pitch)[0]), tol=0.03)
check("slab against affine", -0.223,
      float(stats.pearsonr(_sm.slab_pitch, _sm.aff_pitch)[0]), tol=0.05)
check("combined head-in-bore pitch vs age", 0.428,
      float(stats.pearsonr(_sm.Age, (_sm.aff_pitch - _sm.slab_pitch).abs())[0]), tol=0.03)


# --- Table 2's Refined+ row, which had been left as dashes while Section 4.5
# reported values for it ---
for _tag, _f, _icc, _age in (("HCP-A", "decoupled_roi_hcpa_b1500.csv", 0.950, -0.476),
                             ("DLBS", "decoupled_roi_dlbs.csv", 0.455, -0.344)):
    _d = pd.read_csv(HERE / _f)
    _lon = _d[_d.Subject_ID.isin(_d.Subject_ID.value_counts()[lambda s: s >= 2].index)]
    check(f"{_tag} Refined+ ICC", _icc,
          variance_components(_lon.dropna(subset=["refinedplus_slab"]), "refinedplus_slab")["icc"],
          tol=0.01)
    _s = _d.dropna(subset=["Age", "refinedplus_slab"])
    check(f"{_tag} Refined+ vs age", _age,
          float(np.corrcoef(_s.Age, _s.refinedplus_slab)[0, 1]), tol=0.02)


# --- absorption using the header-derived pose, which atrophy cannot reach ---
_sl2 = pd.read_csv(HERE / "slab_prescription_dlbs.csv")
_hr2 = pd.read_csv(HERE / "head_rotation_dlbs.csv")
_ix2 = pd.read_csv(HERE / "measured_pvs_axis_dlbs.csv")[["Subject_ID", "Visit", "Age", "classic"]]
for _x in (_sl2, _hr2, _ix2):
    _x["Subject_ID"] = _x.Subject_ID.astype(str)
    _x["Visit"] = _x.Visit.astype(str)
_mm = (_sl2.merge(_hr2, on=["Subject_ID", "Visit"]).merge(_ix2, on=["Subject_ID", "Visit"])
          .dropna(subset=["Age", "classic", "slab_pitch", "aff_pitch", "pitch", "total"])
          .sort_values(["Subject_ID", "Visit"]).groupby("Subject_ID").first().reset_index())


def _zz(v):
    v = np.asarray(v, float)
    return (v - v.mean()) / v.std(ddof=1)


def _abeta(y, age, covs):
    _X = [np.ones(len(y)), _zz(age)] + [_zz(c) for c in covs]
    return float(np.linalg.lstsq(np.column_stack(_X), _zz(y), rcond=None)[0][1])


_b0 = _abeta(_mm.classic, _mm.Age, [])
for _lab, _covs, _want in (
        ("registration pose", [np.abs(_mm.pitch), _mm.total], 45.0),
        ("header slab angulation", [_mm.slab_pitch], 12.4),
        ("combined head-in-bore", [(_mm.aff_pitch - _mm.slab_pitch).abs()], 48.0),
        ("header and registration", [_mm.slab_pitch, np.abs(_mm.pitch), _mm.total], 51.7)):
    check(f"DLBS absorbed by {_lab} (%)", _want,
          100 * (1 - abs(_abeta(_mm.classic, _mm.Age, _covs)) / abs(_b0)), tol=0.03)


# --- radius robustness: the absorbed fraction at each radius, one rule for both ---
_r25 = pd.read_csv(HERE / "radius_robustness_dlbs.csv")[["Subject_ID", "Visit", "Age", "classic"]]
_r25.columns = ["Subject_ID", "Visit", "Age", "classic25"]
_r5 = pd.read_csv(HERE / "measured_pvs_axis_dlbs.csv")[["Subject_ID", "Visit", "classic"]]
_hr3 = pd.read_csv(HERE / "head_rotation_dlbs.csv")
for _x in (_r25, _r5, _hr3):
    _x["Subject_ID"] = _x.Subject_ID.astype(str)
    _x["Visit"] = _x.Visit.astype(str)
_rr = (_r25.merge(_r5, on=["Subject_ID", "Visit"]).merge(_hr3, on=["Subject_ID", "Visit"])
          .dropna(subset=["Age", "pitch", "total"])
          .sort_values(["Subject_ID", "Visit"]).groupby("Subject_ID").first().reset_index())
check("radius comparison participants", 155.0, float(len(_rr)), tol=0.01)
for _col, _want in (("classic25", 36.1), ("classic", 49.1)):
    _b0 = _abeta(_rr[_col], _rr.Age, [])
    _b1 = _abeta(_rr[_col], _rr.Age, [np.abs(_rr.pitch), _rr.total])
    check(f"radius absorbed, {_col} (%)", _want, 100 * (1 - abs(_b1) / abs(_b0)), tol=0.03)


# --- composition ---
h = pd.read_csv(HERE / "roi_placement_quality_hcpa_b1500.csv")
check("HCP-A off-tract vs classic", 0.466,
      float(np.corrcoef(h.slf_off_tract, h.classic)[0, 1]), tol=0.03)
check("HCP-A SLF off-tract median (%)", 19.4, float(h.slf_off_tract.median() * 100), tol=0.05)

# --- trigeminal neuralgia ---
tn = pd.read_csv(HERE / "tn_alps.csv")
hp = pd.read_csv(HERE / "head_rotation_tn.csv")
# The only input the verifier needs from outside revision/. It is a public
# OpenNeuro file, so allow the location to be overridden rather than requiring
# the M: volume to be mounted; that is what stopped this script running
# anywhere but the acquisition workstation.
_TN_PAR = Path(winpath(os.environ.get(
    "TN_PARTICIPANTS", "M:/ds005713-download/participants_v2.0.1.tsv")))
par = pd.read_csv(_TN_PAR, sep="\t")
m = tn.merge(par, on="BIDS_ID").merge(hp, on="BIDS_ID")
m["patient"] = (m.BIDS_ID.astype(str).str.extract(r"sub-(\d+)")[0].str.len() >= 3).astype(int)
m["age"] = pd.to_numeric(m.age, errors="coerce")
m = m.dropna(subset=["age", "classic"])
check("TN patients", 115, float(m.patient.sum()), tol=0.02)
check("TN controls", 53, float((1 - m.patient).sum()), tol=0.02)
# The manuscript reports the age- and sex-adjusted partial correlation, which is
# the like-for-like comparison against the group models; the raw value is -0.457.
m["sex_n"] = pd.to_numeric(m.sex, errors="coerce")
m = m.dropna(subset=["sex_n"])
Ctn = np.column_stack([np.ones(len(m)), m.age.to_numpy(float), m.sex_n.to_numpy(float)])


def _r(y):
    b, *_ = np.linalg.lstsq(Ctn, np.asarray(y, float), rcond=None)
    return np.asarray(y, float) - Ctn @ b


check("TN r(|pitch|, classic), age+sex adjusted", -0.414,
      float(np.corrcoef(_r(m.pitch.abs()), _r(m.classic))[0, 1]), tol=0.03)

# --- the trigeminal positioning difference is real but small, and the paper
# now says so. These guard the effect sizes rather than only the p values. ---
for _c, _dw, _lo, _hi in (("pitch", 0.33, 0.23, 2.72), ("total", 0.36, 0.39, 2.79)):
    _a = m[m.patient == 1][_c].abs()
    _b = m[m.patient == 0][_c].abs()
    _sp = np.sqrt(((len(_a) - 1) * _a.var(ddof=1) + (len(_b) - 1) * _b.var(ddof=1))
                  / (len(_a) + len(_b) - 2))
    check(f"TN {_c} group Cohen's d", _dw, float((_a.mean() - _b.mean()) / _sp), tol=0.05)
    _ci = stats.t.interval(0.95, len(_a) + len(_b) - 2,
                           loc=_a.mean() - _b.mean(),
                           scale=np.sqrt(_a.var(ddof=1) / len(_a) + _b.var(ddof=1) / len(_b)))
    check(f"TN {_c} difference CI low", _lo, float(_ci[0]), tol=0.06)
    check(f"TN {_c} difference CI high", _hi, float(_ci[1]), tol=0.06)
check("TN patient pitch SD", 4.98, float(m[m.patient == 1].pitch.abs().std()), tol=0.02)
check("TN control pitch SD", 3.10, float(m[m.patient == 0].pitch.abs().std()), tol=0.02)
check("TN r(|pitch|, classic) unadjusted", -0.46,
      float(np.corrcoef(m.pitch.abs(), m.classic)[0, 1]), tol=0.03)


# --- reconciling the small positioning difference with the 20% absorbed ---
_ctl = m[m.patient == 0].classic.mean()
check("TN group difference as % of control mean", 5.9,
      100 * (_ctl - m[m.patient == 1].classic.mean()) / _ctl, tol=0.05)
_b = np.polyfit(m.pitch.abs(), m.classic, 1)[0]
check("TN index slope, % per degree of pitch", -1.15, 100 * _b / m.classic.mean(), tol=0.05)
check("TN r(group, |pitch|)", 0.153, float(np.corrcoef(m.patient, m.pitch.abs())[0, 1]), tol=0.05)
check("TN transmitted product", -0.070,
      float(np.corrcoef(m.patient, m.pitch.abs())[0, 1]
            * np.corrcoef(m.pitch.abs(), m.classic)[0, 1]), tol=0.06)


# --- trigeminal pose adjustment, under the paper's own model ---
# m already carries pitch and total from the head_rotation_tn merge above.
_mt = m.dropna(subset=["pitch", "total"])


def _tn_cost(col):
    s = _mt.dropna(subset=[col])
    base = np.column_stack([np.ones(len(s)), s.age, s.sex_n])
    pose = np.column_stack([base, np.abs(s.pitch), s.total])

    def pr(C, y):
        def rz(v):
            b, *_ = np.linalg.lstsq(C, np.asarray(v, float), rcond=None)
            return np.asarray(v, float) - C @ b
        return float(np.corrcoef(rz(s.patient.astype(float)), rz(y))[0, 1])
    b0, b1 = pr(base, s[col]), pr(pose, s[col])
    return b0, b1, 100 * (1 - abs(b1) / abs(b0))


_b0, _b1, _c = _tn_cost("classic")
check("TN classic group coefficient", -0.219, _b0)
check("TN classic after pose adjustment", -0.175, _b1)
check("TN classic cost of adjustment (%)", 20.0, _c, tol=0.05)
_CORR = ("cross", "v2_slab_b8", "ALPS-PAS", "per-voxel", "pv_perp")
_costs = [_tn_cost(c)[2] for c in _CORR]
check("TN corrected variants, smallest cost (%)", 3.4, min(_costs), tol=0.15)
check("TN corrected variants, largest cost (%)", 7.6, max(_costs), tol=0.15)
# After adjustment the best correction edges past classic. The manuscript must
# not claim classic stays strongest; this fails if that reverts.
check("TN best corrected after adjustment", -0.180,
      min(_tn_cost(c)[1] for c in _CORR))

# --- statements that must appear verbatim ---
# The isotropic crossover is quoted twice: the thresholds table and the Discussion.
# The Results prose that restated it went when the correction stopped being
# recommended. 8.1 appears elsewhere as a deviation angle, so only 8.2 counts.

results.append((TEX.count(r"$8.2^{\circ}$") == 2,
                "isotropic crossover is 8.2 in both places", None, None))

for phrase in [r"$3.67\%$ (flat)", r"$9.1^{\circ}$", r"$45.0\%$",
               r"r=-0.414", r"$r=+0.332$", r"$r=+0.258$",
               # CP=0.216 retired with the measurement-location section: the claim that
               # the ALPS regions are among the most planar white matter is now made
               # qualitatively in the Discussion, so no planar coefficient appears
               # outside Methods.
               r"$r=-0.631$", r"$-0.581$", r"$-0.419$"]:
    results.append((in_tex(phrase), f"manuscript contains {phrase}", None, None))


# --- composition, the paragraph that was written pre-fix ---
# DLBS descriptive statistics use the post-QC placement set (507), which is also
# what the orientation-confound figure uses. The 379-session subset elsewhere is
# where every variant could be computed, not a different QC rule.
for _tag, _f, _mt, _ma, _q in (("HCP-A", "roi_placement_quality_hcpa_b1500.csv", 19.4, 20.1, 16.9),
                               ("DLBS", "roi_placement_quality_dlbs_all.csv", 18.6, 18.4, 22.9)):
    _d = pd.read_csv(HERE / _f)
    check(f"{_tag} SLF off-tract median (%)", _mt, float(_d.slf_off_tract.median() * 100), tol=0.02)
    check(f"{_tag} SLF left-right median (%)", _ma, float(_d.slf_off_axis.median() * 100), tol=0.02)
    check(f"{_tag} sessions above a quarter (%)", _q, float((_d.slf_off_tract > 0.25).mean() * 100), tol=0.02)
    # The screened-versus-unscreened ICC pairs are no longer quoted: removing the
    # off-tract voxels moved reliability by a few thousandths and the manuscript
    # no longer reports it, so there is nothing in the text to check them against.

# --- axis deviation and inter-fibre angles, DLBS on the QC'd 379-session set ---
for _tag, _f, _want in (("HCP-A", "roi_placement_quality_hcpa_b1500.csv", (8.1, 9.8, 8.7, 80.0)),
                        ("DLBS", "roi_placement_quality_dlbs_all.csv", (15.7, 7.7, 8.1, 76.9))):
    _d = pd.read_csv(HERE / _f)
    for _c, _w in zip(("theta_scr", "theta_slf", "theta_pvs", "theta_interfiber"), _want):
        check(f"{_tag} {_c} median (deg)", _w, float(_d[_c].median()), tol=0.01)


# --- between-individual variation in tract direction, and its stability ---
# The claim is that a common-space registration cannot supply the individual's
# tract direction. It rests on three things: the spread across participants in
# the cohort where posture has already been removed, the reproducibility of the
# angle across separate visits, and its independence from what residual rotation
# remains.
_q = pd.read_csv(HERE / "roi_placement_quality_hcpa_b1500.csv")
for _c, _p5, _p95, _icc in (("theta_scr", 3.4, 16.4, 0.915),
                            ("theta_slf", 4.7, 17.2, 0.918),
                            ("theta_pvs", 4.0, 17.0, 0.925)):
    _v = _q[_c].dropna()
    check(f"HCP-A {_c} 5th pct", _p5, float(np.percentile(_v, 5)), tol=0.03)
    check(f"HCP-A {_c} 95th pct", _p95, float(np.percentile(_v, 95)), tol=0.03)
    _d2 = _q.dropna(subset=[_c])
    _ni = _d2.groupby("Subject_ID")[_c].size()
    _d2 = _d2[_d2.Subject_ID.isin(_ni[_ni >= 2].index)]
    _ni = _d2.groupby("Subject_ID")[_c].size()
    _mi = _d2.groupby("Subject_ID")[_c].mean()
    _a, _N, _gr = len(_ni), int(_ni.sum()), _d2[_c].mean()
    _msb = float((_ni * (_mi - _gr) ** 2).sum() / (_a - 1))
    _msw = float(sum(((_d2[_d2.Subject_ID == _s][_c] - _mi[_s]) ** 2).sum()
                     for _s in _ni.index) / (_N - _a))
    _n0 = (_N - (_ni ** 2).sum() / _N) / (_a - 1)
    check(f"HCP-A {_c} repeat-visit ICC", _icc,
          (_msb - _msw) / (_msb + (_n0 - 1) * _msw), tol=0.02)
    check(f"HCP-A {_c} repeat-visit participants", 628.0, float(_a), tol=0.01)

# residual head rotation explains almost none of it
_hr = pd.read_csv(HERE / "head_rotation_hcpa.csv")
_m = _q.merge(_hr, on=["Subject_ID", "Visit"], how="inner")
_s = _m.dropna(subset=["theta_scr", "total"])
check("HCP-A theta_scr vs residual rotation r", 0.183,
      float(stats.pearsonr(_s.theta_scr, _s.total)[0]), tol=0.10)
for _c in ("theta_slf", "theta_pvs"):
    _s = _m.dropna(subset=[_c, "total"])
    _r, _p = stats.pearsonr(_s[_c], _s.total)
    check(f"HCP-A {_c} vs residual rotation is negligible", 1.0,
          float(abs(_r) < 0.05), tol=1e-9)
    check(f"HCP-A {_c} vs residual rotation is null", 1.0,
          float(_p > 0.1) + 0.0, tol=1e-9)


# --- source integrity: a broken command that LaTeX will not complain about ---
# Writing "\\ref{...}" from a non-raw Python string turns the backslash into a
# carriage return, because \\r is a valid escape and raises no warning. The result,
# CR followed by "ef{sec:foo}", typesets as the literal text "efsec:foo" with no
# LaTeX error and no undefined-reference warning, so only reading the PDF catches
# it. Four of these reached the manuscript before this check existed.
# The anatomical departure is a trait, not an age effect. This null is what makes
# the reorientation head-to-head interpretable: an age-uncorrelated term inflates
# between-participant variance without biasing a group age slope, which is why
# vecreg and the closed-form correction tie on the age association.
_tda = pd.read_csv(HERE / "tract_direction_age.csv").set_index("cohort")
check("theta_pvs vs age, HCP-A", -0.065, float(_tda.loc["HCP-A", "r_age"]), tol=0.05)
check("theta_pvs vs age p, HCP-A", 0.105, float(_tda.loc["HCP-A", "p_age"]), tol=0.05)
check("theta_pvs vs age, DLBS", 0.032, float(_tda.loc["DLBS", "r_age"]), tol=0.05)
check("theta_pvs vs age p, DLBS", 0.594, float(_tda.loc["DLBS", "p_age"]), tol=0.05)
check("theta_pvs age-null n, HCP-A", 628.0, float(_tda.loc["HCP-A", "n"]), tol=1e-9)
check("theta_pvs age-null n, DLBS", 284.0, float(_tda.loc["DLBS", "n"]), tol=1e-9)

# Reorientation applies an age-graded transformation. The affine rotation tracks
# age because head pitch does, which is the confound being removed. The Jacobian
# tracks age because of atrophy, which is a property of the resampling. The warp's
# own rotation is flat, which is what keeps this from being an indictment.
_rad = pd.read_csv(HERE / "registration_age_dependence.csv")
_ap = _rad[_rad.quantity.str.endswith("applied")]
_wr = _rad[_rad.quantity.str.endswith("local_vs_affine")]
_jd = _rad[_rad.quantity.str.endswith("jac_det")]
check("affine rotation vs age, min r", 0.329, float(_ap.r_age.min()), tol=0.02)
check("affine rotation vs age, max r", 0.339, float(_ap.r_age.max()), tol=0.02)
check("affine rotation vs age, worst p", 2.8e-5, float(_ap.p_age.max()), tol=0.10)
check("warp rotation vs age, min r", 0.014, float(_wr.r_age.min()), tol=0.10)
check("warp rotation vs age, max r", 0.095, float(_wr.r_age.max()), tol=0.05)
check("warp rotation vs age, none significant", 1.0,
      float(_wr.p_age.min() > 0.05), tol=1e-9)
check("jacobian vs age, min r", -0.329, float(_jd.r_age.min()), tol=0.02)
check("jacobian vs age, max r", -0.199, float(_jd.r_age.max()), tol=0.02)
check("age-dependence rows cover four region-hemispheres", 4.0,
      float(len(_ap)), tol=1e-9)

# --- the anatomical-axis variant, Table 4 row and the text that reads it ---
_ax_h = pd.read_csv(HERE / "measured_pvs_axis_hcpa_b1500_all.csv")
_ax_d = pd.read_csv(HERE / "measured_pvs_axis_dlbs.csv")
for _d, _tag, _want_icc, _want_r in ((_ax_h, "HCP-A", 0.955, -0.538),
                                     (_ax_d, "DLBS", 0.516, -0.356)):
    _lon = _d[_d.Subject_ID.isin(_d.Subject_ID.value_counts()[lambda s: s >= 2].index)]
    check(f"anat_x ICC {_tag}", _want_icc,
          float(variance_components(_lon.dropna(subset=["anat_x"]), "anat_x")["icc"]), tol=0.01)
    _s = _d.dropna(subset=["anat_x", "Age"])
    check(f"anat_x age r {_tag}", _want_r,
          float(stats.pearsonr(_s.Age, _s.anat_x)[0]), tol=0.02)
# best conditioned corrected variant in HCP-A, and within 0.002 of classic
_lh = _ax_h[_ax_h.Subject_ID.isin(_ax_h.Subject_ID.value_counts()[lambda s: s >= 2].index)]
_iccs = {_c: float(variance_components(_lh.dropna(subset=[_c]), _c)["icc"])
         for _c in ("classic", "cross", "v2_slab", "pv_perp", "anat_x")}
check("anat_x is the best conditioned corrected variant in HCP-A", 1.0,
      float(_iccs["anat_x"] == max(_iccs[c] for c in ("cross", "v2_slab", "pv_perp", "anat_x"))),
      tol=1e-9)
check("anat_x within 0.002 of classic ICC in HCP-A", 1.0,
      float(abs(_iccs["classic"] - _iccs["anat_x"]) <= 0.0025), tol=1e-9)
# exactly invariant when the registration rotates with the head
_inv = pd.read_csv(HERE / "anat_x_invariance.csv")
check("anat_x rotation departure is machine precision", 1.0,
      float(_inv.anat_x_rotated.max() < 1e-9), tol=1e-9)

# --- denominator contamination: the construct-validity claim as a number ---
# The headline is that template reorientation is indistinguishable from no
# correction in the aligned cohort, because it removes posture and the residual
# there is anatomy, which fixed template axes cannot reach.
_dc = {_c: pd.read_csv(HERE / f"denominator_contamination_{_c}.csv")
       for _c in ("hcpa", "dlbs")}
for _coh, _tag, _reg, _want in (("hcpa", "classic", "proj", 10.2),
                                ("hcpa", "classic", "assoc", 15.8),
                                ("hcpa", "vecreg", "proj", 10.1),
                                ("hcpa", "vecreg", "assoc", 16.0),
                                ("hcpa", "refined", "proj", 7.5),
                                ("hcpa", "anat_x", "assoc", 7.4),
                                ("dlbs", "classic", "proj", 29.7),
                                ("dlbs", "classic", "assoc", 8.9),
                                ("dlbs", "vecreg", "proj", 10.5),
                                ("dlbs", "vecreg", "assoc", 16.1),
                                ("dlbs", "anat_x", "proj", 6.3)):
    check(f"{_coh} {_tag} {_reg} lambda1 share",
          _want, 100 * float(_dc[_coh][f"{_tag}_{_reg}"].median()), tol=0.02)
# in HCP-A reorientation is within half a point of doing nothing, both regions
for _reg in ("proj", "assoc"):
    check(f"HCP-A reorientation matches classic, {_reg}", 1.0,
          float(abs(_dc["hcpa"][f"vecreg_{_reg}"].median()
                    - _dc["hcpa"][f"classic_{_reg}"].median()) < 0.005), tol=1e-9)
# every corrected denominator lands in the 7 to 8 percent band in both cohorts
for _coh in ("hcpa", "dlbs"):
    for _tag in ("refined", "anat_x"):
        for _reg in ("proj", "assoc"):
            check(f"{_coh} {_tag} {_reg} is at the dispersion floor", 1.0,
                  float(0.05 < _dc[_coh][f"{_tag}_{_reg}"].median() < 0.09), tol=1e-9)

# --- is the contamination age-graded, or only a level shift? ---
# This is what reconciles two findings that otherwise look contradictory: the
# denominators differ by a factor of two between reorientation and the
# correction, yet the age associations match. Posture is age-graded so its
# contamination biases a slope; anatomy is not, so its contamination shifts the
# level only.
_dcd = pd.read_csv(HERE / "denominator_contamination_dlbs.csv")
_age = (pd.read_csv(HERE / "measured_pvs_axis_dlbs.csv")[["Subject_ID", "Age"]]
        .drop_duplicates("Subject_ID"))
for _f in (_dcd, _age):
    _f["Subject_ID"] = _f.Subject_ID.astype(str)
_cm = (_dcd.groupby("Subject_ID").mean(numeric_only=True).reset_index()
       .merge(_age, on="Subject_ID"))
for _col, _want in (("classic_proj", 0.325), ("vecreg_proj", 0.131)):
    check(f"{_col} contamination vs age", _want,
          float(stats.pearsonr(_cm[_col], _cm.Age)[0]), tol=0.10)
for _col in ("refined_proj", "anat_x_proj"):
    check(f"{_col} contamination vs age is negligible", 1.0,
          float(abs(stats.pearsonr(_cm[_col], _cm.Age)[0]) < 0.05), tol=1e-9)
check("classic projection contamination is age-graded", 1.0,
      float(stats.pearsonr(_cm.classic_proj, _cm.Age)[1] < 0.001), tol=1e-9)
for _col in ("vecreg_proj", "vecreg_assoc", "refined_proj", "anat_x_proj", "anat_x_assoc"):
    check(f"{_col} contamination is age-flat", 1.0,
          float(stats.pearsonr(_cm[_col], _cm.Age)[1] > 0.05), tol=1e-9)

# --- does the direction error surviving registration still track age? ---
_rra = pd.read_csv(HERE / "registration_residual_age.csv")
_rr = _rra[_rra.hemi != "pooled"].set_index(["stage", "tract", "hemi"])
# The pooled test is the one the manuscript reports, since four per stage would
# need a multiplicity allowance the pooled version does not.
_rp = _rra[_rra.hemi == "pooled"].set_index(["stage", "tract"])
for _k, _r, _pmax in ((("native", "proj"), 0.331, 1e-4),
                      (("native", "combined"), 0.221, 0.01),
                      (("affine", "proj"), 0.115, None),
                      (("affine", "combined"), 0.020, None)):
    check(f"pooled {_k[0]} {_k[1]} vs age", _r, float(_rp.loc[_k, "r_age"]), tol=0.10)
    if _pmax is not None:
        check(f"pooled {_k[0]} {_k[1]} is significant", 1.0,
              float(_rp.loc[_k, "p_age"] < _pmax), tol=1e-9)
# registration removes the age dependence rather than reducing it
for _tag in ("proj", "assoc", "combined"):
    check(f"pooled affine {_tag} is age-flat", 1.0,
          float(_rp.loc[("affine", _tag), "p_age"] > 0.05), tol=1e-9)
for _k, _med, _r in ((("native", "proj", "L"), 19.66, 0.270),
                     (("native", "proj", "R"), 22.33, 0.365),
                     (("affine", "proj", "L"), 10.26, 0.042),
                     (("affine", "proj", "R"), 12.61, 0.169),
                     (("nonlinear", "proj", "R"), 13.31, 0.210),
                     (("native", "assoc", "L"), 10.44, -0.033),
                     (("affine", "assoc", "L"), 16.25, 0.001),
                     (("affine", "assoc", "R"), 14.14, -0.120)):
    check(f"{_k[0]} {_k[1]} {_k[2]} median angle", _med,
          float(_rr.loc[_k, "median"]), tol=0.02)
    if abs(_r) > 0.05:
        check(f"{_k[0]} {_k[1]} {_k[2]} vs age", _r,
              float(_rr.loc[_k, "r_age"]), tol=0.10)
# the projection direction is age-graded before registration in both hemispheres
for _h in ("L", "R"):
    check(f"native projection is age-graded, {_h}", 1.0,
          float(_rr.loc[("native", "proj", _h), "p_age"] < 0.001), tol=1e-9)
# the association direction is age-flat at every stage
for _s in ("native", "affine", "nonlinear"):
    for _h in ("L", "R"):
        check(f"{_s} association is age-flat, {_h}", 1.0,
              float(_rr.loc[(_s, "assoc", _h), "p_age"] > 0.05), tol=1e-9)
# registration moves the association direction further from its assumed axis
for _h in ("L", "R"):
    check(f"registration worsens the association angle, {_h}", 1.0,
          float(_rr.loc[("affine", "assoc", _h), "median"]
                > _rr.loc[("native", "assoc", _h), "median"]), tol=1e-9)

# --- LD-ALPS, from the authors' own implementation ---
# Their pitch claim replicates in DLBS; their reliability claim does not. Run
# unmodified from external/ld-alps.py, so a change here means their code or our
# adapter changed, not our reading of their paper.
_ld = pd.read_csv(HERE / "ld_alps_dlbs.csv")[["Subject_ID", "Visit", "ALPS_overall"]]
_ld = _ld.rename(columns={"ALPS_overall": "ld_alps"})
_mv = pd.read_csv(HERE / "measured_pvs_axis_dlbs.csv")
_hrd = pd.read_csv(HERE / "head_rotation_dlbs.csv")
for _f in (_ld, _mv, _hrd):
    _f["Subject_ID"] = _f.Subject_ID.astype(str)
    _f["Visit"] = _f.Visit.astype(str)
_lj = _mv.merge(_ld, on=["Subject_ID", "Visit"]).dropna(subset=["ld_alps", "Age"])
check("LD-ALPS sessions on the variant sample", 379.0, float(len(_lj)), tol=1e-9)
_llon = _lj[_lj.Subject_ID.isin(_lj.Subject_ID.value_counts()[lambda s: s >= 2].index)]
check("LD-ALPS ICC, DLBS", 0.455,
      float(variance_components(_llon, "ld_alps")["icc"]), tol=0.02)
check("LD-ALPS age r, DLBS", -0.376,
      float(stats.pearsonr(_lj.Age, _lj.ld_alps)[0]), tol=0.02)
_lo = (_lj.sort_values(["Subject_ID", "Visit"]).groupby("Subject_ID").first().reset_index()
       .merge(_hrd, on=["Subject_ID", "Visit"]))
check("LD-ALPS vs |pitch|", -0.218,
      float(stats.pearsonr(_lo.ld_alps, _lo.pitch.abs())[0]), tol=0.05)
check("classic vs |pitch| on the same sessions", -0.475,
      float(stats.pearsonr(_lo.classic, _lo.pitch.abs())[0]), tol=0.05)
# it is the least reliable variant, which is what does not replicate
check("LD-ALPS is the least reliable variant in DLBS", 1.0,
      float(variance_components(_llon, "ld_alps")["icc"]
            <= min(variance_components(_llon, _c)["icc"]
                   for _c in ("classic", "cross", "v2_slab", "pv_perp", "anat_x"))), tol=1e-9)
# and it agrees closely with the axes estimated from the tensor
for _c, _want in (("anat_x", 0.85), ("cross", 0.88)):
    check(f"LD-ALPS agrees with {_c}", _want,
          float(_lj[["ld_alps", _c]].corr().iloc[0, 1]), tol=0.03)

# --- the phenotype sweep null, adjusted against unadjusted ---
# The claim in 4.8 is that nothing survives age adjustment and a great deal
# survives without it. Both arms are regenerated by phenotype_sweep.py, the
# second with ALPS_SWEEP_UNADJUSTED=1.
_ps = pd.read_csv(HERE / "phenotype_sweep.csv")
_pv = [c[2:] for c in _ps.columns if c.startswith("q_")]
_pu = pd.read_csv(HERE / "phenotype_sweep_unadjusted.csv")
check("phenotypes tested in the HCP-A sweep", 219.0, float(len(_ps)), tol=1e-9)
check("both arms test the same phenotypes", 219.0, float(len(_pu)), tol=1e-9)
for _c in _pv:
    if _c in ("anat_x", "pv_perp"):
        # the only two with any survivor, and few: 3 and 7, one MRS family
        check(f"{_c} survivors after age adjustment are few", 1.0,
              float(0 < (_ps[f"q_{_c}"] < 0.05).sum() <= 7), tol=1e-9)
    else:
        check(f"{_c} finds nothing after age adjustment", 0.0,
              float((_ps[f"q_{_c}"] < 0.05).sum()), tol=1e-9)
    # and the same variant finds a hundred or more without adjusting for age
    check(f"{_c} finds many unadjusted", 1.0,
          float((_pu[f"q_{_c}"] < 0.05).sum() >= 100), tol=1e-9)
check("unadjusted survivor count spans 101 to 117", 1.0,
      float(101 <= min((_pu[f"q_{_c}"] < 0.05).sum() for _c in _pv)
            and max((_pu[f"q_{_c}"] < 0.05).sum() for _c in _pv) <= 117), tol=1e-9)

# --- attribution: the voxelwise variant is ALPS-PAS minus one step ---
# Presenting it as a new proposal would misattribute the formulation to us.
# Ajouz et al. supply the lambda2 over lambda3 form, Schilling et al. the
# observation that the classic index tracks it, and the contribution here is
# that the reduction is exact.
for _phrase in ("this construction with the selection rule removed",
                "due to Ajouz et al.",
                "the ALPS-PAS construction of Ajouz et al."):
    results.append((_phrase in TEX,
                    f"attribution present: {_phrase[:44]}", None, None))

# --- the voxelwise advantage survives disattenuation ---
# The paper previously explained it away as lower variance. Dividing each
# correlation by the square root of its reliability removes that advantage, and
# the ordering does not change, so the explanation fails and the text now says so.
for _f, _tag, _want_pv, _want_cl in (
        ("measured_pvs_axis_dlbs.csv", "DLBS", -0.447, -0.425),
        ("measured_pvs_axis_hcpa_b1500_all.csv", "HCP-A", -0.557, -0.440)):
    _dd = pd.read_csv(HERE / _f)
    _rep = _dd[_dd.Subject_ID.isin(_dd.Subject_ID.value_counts()[lambda s: s >= 2].index)]
    _one = _dd.sort_values(["Subject_ID", "Visit"]).groupby("Subject_ID").first().reset_index()
    _dis = {}
    for _c in ("classic", "cross", "v2_slab", "pv_perp"):
        _icc = variance_components(_rep.dropna(subset=[_c]), _c)["icc"]
        _s = _one[[_c, "Age"]].dropna()
        _dis[_c] = float(stats.pearsonr(_s[_c], _s.Age)[0]) / np.sqrt(max(_icc, 1e-9))
    check(f"disattenuated age r, pv_perp, {_tag}", _want_pv, _dis["pv_perp"], tol=0.03)
    check(f"disattenuated age r, classic, {_tag}", _want_cl, _dis["classic"], tol=0.03)
    check(f"pv_perp leads after disattenuation, {_tag}", 1.0,
          float(abs(_dis["pv_perp"]) == max(abs(v) for v in _dis.values())), tol=1e-9)

# --- nothing survives the eigenvalue ratio ---
# The paper's main result after revision. Every corrected variant's association
# with age or with clinical group vanishes once lambda2/lambda3 is partialled
# out, and the residuals that remain belong to variants whose axes miss v2.
_be = pd.read_csv(HERE / "beyond_eigenvalue_ratio.csv")
_key = _be.set_index(["cohort", "endpoint", "variant"])
for _k, _raw_r, _part in ((("hcpa", "age", "classic"), -0.430, 0.120),
                          (("hcpa", "age", "cross"), -0.436, 0.057),
                          (("hcpa", "age", "v2_slab"), -0.518, -0.033),
                          (("hcpa", "age", "anat_x"), -0.505, 0.003),
                          (("trigeminal", "patient", "ALPS-PAS"), -0.182, -0.013),
                          (("trigeminal", "patient", "per-voxel"), -0.169, -0.002)):
    check(f"{_k[0]} {_k[2]} raw", _raw_r, float(_key.loc[_k, "raw"]), tol=0.03)
    if abs(_part) > 0.05:
        check(f"{_k[0]} {_k[2]} given the ratio", _part,
              float(_key.loc[_k, "partial"]), tol=0.10)
    else:
        check(f"{_k[0]} {_k[2]} given the ratio is negligible", 1.0,
              float(abs(_key.loc[_k, "partial"]) < 0.05), tol=1e-9)
# in HCP-A every corrected variant lands within 0.06 of zero. The largest is the
# cross product at 0.057, which is why the assertion is 0.06 and not 0.03. The
# manuscript said 0.03 for several drafts while this test passed; it now says 0.057.
_h = _be[(_be.cohort == "hcpa") & (_be.variant.isin(["cross", "v2_sphere", "v2_slab", "anat_x"]))]
check("HCP-A corrected variants vanish given the ratio", 1.0,
      float(_h.partial.abs().max() < 0.06), tol=1e-9)
# no trigeminal variant retains a group effect
_t = _be[(_be.cohort == "trigeminal") & (_be.variant != "pv_perp")]
check("no trigeminal variant survives the ratio", 1.0,
      float((_t.p > 0.05).all()), tol=1e-9)
# the v2-estimating variants retain nothing in either aging cohort
_v = _be[_be.variant.isin(["v2_slab", "v2_sphere"]) & _be.partial.notna()
         & (_be.endpoint == "age")]
check("v2 variants retain nothing on age", 1.0,
      float(_v.partial.abs().max() < 0.06), tol=1e-9)
# and nothing significant anywhere, including the trigeminal cohort
_va = _be[_be.variant.isin(["v2_slab", "v2_sphere"]) & _be.p.notna()]
check("v2 variants never significant given the ratio", 1.0,
      float((_va.p > 0.05).all()), tol=1e-9)
# and the title names the three objects the paper relates
results.append(("Head Position, DTI-ALPS, and Radial Anisotropy" in TEX,
                "title names the three objects", None, None))

# --- the bound is proved, so check the algebra rather than only the data ---
_bp = pd.read_csv(HERE / "ratio_bound_proof.csv")
_cf = _bp[_bp.check == "closed_form"]
# closed form agrees with the quadratic expansion at small angles
_small = _cf[_cf.deg <= 5]
check("quadratic expansion matches at small angle", 1.0,
      float((_small.frac - _small.quad).abs().max() < 0.001), tol=1e-9)
# and R equals the bound exactly at alpha = 0
check("bound attained at zero angle", 1.0,
      float((_cf[_cf.deg == 0].frac - 1.0).abs().max() < 1e-12), tol=1e-9)
# real data: classic never exceeds it, regional-axis variants rarely
_rl = _bp[_bp.check == "real"].set_index("variant")
check("classic never exceeds the bound", 0.0,
      float(_rl.loc["classic", "pct_violating"]), tol=1e-9)
for _v in ("cross", "v2_slab", "anat_x"):
    if _v in _rl.index:
        check(f"{_v} exceeds the bound only rarely", 1.0,
              float(_rl.loc[_v, "pct_violating"] < 3.0), tol=1e-9)
# The derivation is in the text. The monotonicity apparatus it used to carry was
# redundant: lambda2 and lambda3 ARE the extremes of the perpendicular plane, so
# the inequality is immediate and the expansion still gives the rate.
results.append(("R(\\alpha) = \\frac" in TEX
                and "modulated by one angle and nothing else" in TEX,
                "the bound is given in closed form", None, None))

# --- attribution for the eigenvalue ratio ---
# Westin for the shape framework the quantity belongs to, Schilling for the
# connection to this index and for the aligned-case derivation of it in their
# Figure 8. Ours is the misaligned case: the bound, its monotonicity, the
# condition for equality, and the second-order rate. Not the quantity, not the
# observation, and not the aligned-case identity.
for _phrase in ("first expressed the index as",
                "state the same reduction from the assumed geometry"):
    results.append((_phrase in TEX,
                    f"ratio attribution: {_phrase[:40]}", None, None))

_raw = (HERE.parent / "mri_revision.tex").read_bytes()
_lone_cr = sum(1 for _i, _b in enumerate(_raw)
               if _b == 0x0D and (_i + 1 >= len(_raw) or _raw[_i + 1] != 0x0A))
check("no stray carriage returns in the manuscript", 0.0, float(_lone_cr), tol=1e-9)
# 0x09 is included because this manuscript legitimately contains no tabs, and a
# tab is what a corrupted \\times leaves behind.
for _c in (0x07, 0x08, 0x09, 0x0B, 0x0C):
    check(f"no control byte {hex(_c)} in the manuscript", 0.0,
          float(_raw.count(bytes([_c]))), tol=1e-9)
# every ref resolves, checked in the source rather than trusting the log
_txt = _raw.decode("utf-8")
_lab = set(re.findall(r"\\label\{([^}]*)\}", _txt))
_ref = set(re.findall(r"\\(?:page|eq|auto)?ref\{([^}]*)\}", _txt))
check("no undefined cross-references", 0.0, float(len(_ref - _lab)), tol=1e-9)


# --- how much of the tract-direction spread registration actually removes ---
# The manuscript claims a structural registration aligns brains without aligning
# tracts. The test carries each measured direction into template space and
# compares the spread about the cohort mean axis before and after. Directions are
# axial, so the mean axis is the principal eigenvector of the dyadic tensor.
_r = pd.read_csv(HERE / "registration_aligns_tracts.csv")
_one = (_r.sort_values(["Subject_ID", "Session"])
          .groupby(["Subject_ID", "hemi"]).first().reset_index())
check("registration test participants", 284.0, float(_one.Subject_ID.nunique()), tol=0.01)


def _disp(_s, _tag, _stage):
    _V = _s[[f"{_tag}_{_stage}_{_c}" for _c in "xyz"]].to_numpy()
    _T = (_V[:, :, None] * _V[:, None, :]).mean(axis=0)
    _m = np.linalg.eigh(_T)[1][:, -1]
    return np.degrees(np.arccos(np.clip(np.abs(_V @ _m), 0, 1)))


_aff, _nl = [], []
for _tag in ("proj", "assoc"):
    for _h in ("L", "R"):
        _s = _one[_one.hemi == _h].dropna(subset=[f"{_tag}_nonlinear_x"])
        _n = np.median(_disp(_s, _tag, "native"))
        _aff.append(100 * (1 - np.median(_disp(_s, _tag, "affine")) / _n))
        _nl.append(100 * (1 - np.median(_disp(_s, _tag, "nonlinear")) / _n))
        # the nonlinear stage never beats the affine
        check(f"{_tag} {_h} nonlinear no better than affine", 1.0,
              float(_nl[-1] <= _aff[-1] + 0.5) + 0.0, tol=1e-9)
        check(f"{_tag} {_h} native vs affine significant", 1.0,
              float(stats.wilcoxon(_disp(_s, _tag, "native"),
                                   _disp(_s, _tag, "affine")).pvalue < 1e-8) + 0.0,
              tol=1e-9)
check("affine removes at least", 13.0, float(min(_aff)), tol=0.06)
check("affine removes at most", 20.0, float(max(_aff)), tol=0.06)
check("full registration removes at least", 7.0, float(min(_nl)), tol=0.10)
check("full registration removes at most", 14.0, float(max(_nl)), tol=0.10)
for _tag, _w in (("proj", 5.53), ("assoc", 6.39)):
    check(f"{_tag} local warp rotation vs affine (deg)", _w,
          float(_r[f"{_tag}_local_vs_affine"].median()), tol=0.02)


# --- the asymmetry survives in the aligned cohort, so it is not positional ---
for _tag, _hf, _w in (("HCP-A", "hemisphere_age_hcpa_b1500.csv", (3.4, 3.8)),
                      ("DLBS", "hemisphere_age_dlbs.csv", (5.4, 9.7))):
    _d = pd.read_csv(HERE / _hf)
    for _v, _want in zip(("classic", "refined"), _w):
        _L, _R = _d[f"{_v}_L"], _d[f"{_v}_R"]
        check(f"{_tag} {_v} left-right asymmetry (%)", _want,
              float(100 * (_L.mean() - _R.mean()) / _R.mean()), tol=0.10)


# --- hemispheres, which had been reported with left and right reversed ---
# The convention was verified against the images rather than assumed. NIfTI world
# coordinates are RAS+, so world x < 0 is anatomical left, and in these volumes JHU
# label 26 (SCR L) occupies 100% of the x < 0 voxels while label 25 (SCR R) occupies
# 100% of x > 0. The ("L", xc < 0, 26, 42) pairing in hemisphere_age.py is therefore
# correct, and recomputing 22 sessions reproduced the stored file exactly.
for _tag, _hf, _mf, _np, _want in (
        ("HCP-A", "hemisphere_age_hcpa_b1500.csv", "measured_pvs_axis_hcpa_b1500_all.csv",
         809, (-0.411, -0.408, -0.399, -0.413)),
        ("DLBS", "hemisphere_age_dlbs.csv", "measured_pvs_axis_dlbs.csv",
         156, (-0.324, -0.285, -0.272, -0.168))):
    _d, _m = pd.read_csv(HERE / _hf), pd.read_csv(HERE / _mf)
    for _x in (_d, _m):
        _x["k"] = _x.Subject_ID.astype(str) + "|" + _x.Visit.astype(str)
    _d = _d[_d.k.isin(set(_m.k))]
    # the left hemisphere is the higher one; this fails if the reversal returns
    for _v in ("classic", "refined"):
        check(f"{_tag} left exceeds right, {_v}", 1.0,
              float(_d[f"{_v}_L"].mean() > _d[f"{_v}_R"].mean()) + 0.0, tol=1e-9)
    _one = (_d.sort_values(["Subject_ID", "Visit"]).groupby("Subject_ID").first()
              .reset_index().dropna(subset=["Age", "classic_L", "classic_R",
                                            "refined_L", "refined_R"]))
    check(f"{_tag} hemisphere participants", float(_np), float(len(_one)), tol=0.01)
    for _c, _w in zip(("classic_L", "refined_L", "classic_R", "refined_R"), _want):
        check(f"{_tag} {_c} vs age", _w,
              float(stats.pearsonr(_one.Age, _one[_c])[0]), tol=0.02)


# --- region volume and composition absorption, same convention for both:
# all sessions of the placement-quality set, the two region sizes as separate covariates ---
def _absorb(_d, _col, _extra):
    def _z(v):
        v = np.asarray(v, float); return (v - v.mean()) / v.std(ddof=1)
    _y, _age = _z(_d[_col]), _z(_d.Age)
    _b0 = np.linalg.lstsq(np.column_stack([np.ones(len(_d)), _age]), _y, rcond=None)[0][1]
    _X = np.column_stack([np.ones(len(_d)), _age] + [_z(e) for e in _extra])
    _b1 = np.linalg.lstsq(_X, _y, rcond=None)[0][1]
    return 100 * (1 - abs(_b1) / abs(_b0))


for _tag, _f, _vol, _comp in (("HCP-A", "roi_placement_quality_hcpa_b1500.csv",
                               (34.0, 33.4), (18.3, 18.1)),
                              ("DLBS", "roi_placement_quality_dlbs_all.csv",
                               (33.5, 38.4), None)):
    _d = pd.read_csv(HERE / _f).dropna(subset=["Age", "classic", "refined_slab",
                                               "n_scr", "n_slf", "slf_off_tract", "scr_off_tract"])
    for _c, _w in zip(("classic", "refined_slab"), _vol):
        check(f"{_tag} {_c} volume absorbed (%)", _w, _absorb(_d, _c, [_d.n_scr, _d.n_slf]), tol=0.02)
    if _comp:
        for _c, _w in zip(("classic", "refined_slab"), _comp):
            check(f"{_tag} {_c} composition absorbed (%)", _w,
                  _absorb(_d, _c, [_d.slf_off_tract, _d.scr_off_tract]), tol=0.02)


# --- rotation crossovers, interpolated from the tolerance curves ---
_cv = pd.read_csv(HERE / "rotation_tolerance_curves.csv")
for _mode, _want in (("isotropic", 8.2), ("pitch (x)", 9.1)):
    _s = _cv[_cv["mode"] == _mode].sort_values("sigma")
    _x, _y = _s.sigma.to_numpy(float), _s.classic.to_numpy(float)
    _r = float(_s.refined.iloc[0])
    _i = int(np.where(_y > _r)[0][0])
    check(f"classic/refined crossover, {_mode} (deg)", _want,
          float(np.interp(_r, [_y[_i-1], _y[_i]], [_x[_i-1], _x[_i]])), tol=0.02)
# roll must never reach the refined level in range, which the text asserts
_roll = _cv[_cv["mode"] == "roll (y)"]
check("roll never crosses refined", 1.0,
      float(_roll.classic.max() < float(_roll.refined.iloc[0])) + 0.0, tol=1e-9)

# --- observed DLBS pose medians quoted in the rotation section ---
_hd = pd.read_csv(HERE / "head_rotation_dlbs.csv")
# compared at the precision the manuscript states, since a relative tolerance is
# meaningless for sub-degree values
for _c, _w in (("pitch", 10.7), ("roll", 0.8), ("yaw", 0.7)):
    check(f"DLBS median |{_c}| (deg, 1dp)", _w,
          round(float(_hd[_c].abs().median()), 1), tol=1e-9)


# --- the composition figure hard-codes panel (c); it drifted from the text once ---
_fs = (HERE / "build_composition_figure.py").read_text(encoding="utf-8")
for _name, _want in (("classic_v", "[45.0, 34.0, 18.3]"), ("refined_v", "[20.3, 33.4, 18.1]")):
    results.append((f"{_name} = {_want}" in _fs,
                    f"composition figure {_name} matches the text", None, None))


# --- placement reproducibility across visits ---
_pr = pd.read_csv(HERE / "placement_reproducibility.csv")
for _tag, _col, _want in (("HCP-A", "n_scr", 0.953), ("HCP-A", "n_slf", 0.928),
                          ("DLBS", "n_scr", 0.779), ("DLBS", "n_slf", 0.734),
                          ("HCP-A", "slf_off_tract", 0.896)):
    _r = _pr[(_pr.cohort == _tag) & (_pr.col == _col)]
    check(f"{_tag} {_col} placement ICC", _want, float(_r.icc.iloc[0]), tol=0.01)
# the claim the section rests on: in DLBS the regions are more reproducible than
# the index computed from them
_d = _pr[_pr.cohort == "DLBS"].set_index("col").icc
check("DLBS placement exceeds index reliability", 1.0,
      float(min(_d["n_scr"], _d["n_slf"]) > max(_d["classic"], _d["refined_slab"])) + 0.0,
      tol=1e-9)


# --- does between-visit change grow with the interval? ---
_vi = pd.read_csv(HERE / "visit_interval_test.csv").set_index(["cohort", "variant"])
for _c, _v, _r in (("HCP-A", "classic", 0.287), ("HCP-A", "refined_slab", 0.242),
                   ("DLBS", "classic", -0.104), ("DLBS", "refined_slab", -0.131)):
    check(f"{_c} {_v} change vs visit gap", _r, float(_vi.loc[(_c, _v), "r"]), tol=0.03)
# the claim: accumulation is detectable in HCP-A and not in DLBS, so only the
# HCP-A intraclass correlations may be read as lower bounds
check("HCP-A accumulates, DLBS does not", 1.0,
      float(_vi.loc[("HCP-A", "classic"), "r"] > 0
            and _vi.loc[("HCP-A", "classic"), "p"] < 0.05
            and _vi.loc[("DLBS", "classic"), "p"] > 0.05) + 0.0, tol=1e-9)


# --- measurement region size: the corrected index gains where classic does not ---
_rv = pd.read_csv(HERE / "roi_variants_hcpa_b1500.csv")
_rvl = _rv[_rv.Subject_ID.isin(_rv.Subject_ID.value_counts()[lambda s: s >= 2].index)]
_icc = {k: variance_components(_rvl.dropna(subset=[k]), k)["icc"]
        for k in ("classic_sphere", "refined_sphere", "classic_band", "refined_band")}
check("sphere classic ICC", 0.957, _icc["classic_sphere"], tol=0.01)
check("sphere refined ICC", 0.940, _icc["refined_sphere"], tol=0.01)
check("band refined ICC", 0.947, _icc["refined_band"], tol=0.01)
# the claim: the refined index improves in the larger region and the gap narrows
check("refined gains in the band, classic does not", 1.0,
      float(_icc["refined_band"] > _icc["refined_sphere"]
            and _icc["classic_band"] < _icc["classic_sphere"]) + 0.0, tol=1e-9)


# --- the two kinds of angular error, which the paper must not conflate ---
_ax = pd.read_csv(HERE / "axis_error_sensitivity.csv").set_index("deg")
for _deg, _within, _off in ((2.0, 0.04, -0.36), (5.0, 0.11, -1.76),
                            (10.0, 0.26, -6.10), (15.0, 0.43, -12.26),
                            (20.0, 0.62, -19.35)):
    check(f"axis error within plane at {_deg:.0f} deg (%)", _within,
          float(_ax.loc[_deg, "within_plane_pct"]), tol=0.05)
    check(f"frame off tract at {_deg:.0f} deg (%)", _off,
          float(_ax.loc[_deg, "frame_off_tract_pct"]), tol=0.05)
# the claim the section rests on
check("frame error exceeds in-plane error 30-fold at 20 deg", 31.0,
      abs(_ax.loc[20.0, "frame_off_tract_pct"]) / abs(_ax.loc[20.0, "within_plane_pct"]),
      tol=0.05)


# --- the same two errors at the angles they actually take ---
_ob = pd.read_csv(HERE / "axis_error_observed.csv").set_index("case").pct
check("axis estimate cost, HCP-A (%)", 0.026, float(_ob["axis estimate, HCP-A"]), tol=0.05)
check("axis estimate cost, DLBS (%)", 0.090, float(_ob["axis estimate, DLBS"]), tol=0.05)
check("head rotation cost, DLBS (%)", -6.825, float(_ob["head rotation, DLBS"]), tol=0.02)
check("observed errors differ 76-fold", 76.0,
      abs(_ob["head rotation, DLBS"]) / abs(_ob["axis estimate, DLBS"]), tol=0.05)


# --- region size robustness: 2.5 mm against 5 mm on identical sessions ---
def _radius_pair(_small_f, _big_f):
    _s = pd.read_csv(HERE / _small_f)
    _b = pd.read_csv(HERE / _big_f).rename(columns={"classic": "classic5",
                                                    "refined_slab": "refined5"})
    for _d in (_s, _b):
        _d["k"] = _d.Subject_ID.astype(str) + "|" + _d.Visit.astype(str)
    return _s.merge(_b[["k", "classic5", "refined5", "slf_off_tract"]], on="k")


_jh = _radius_pair("radius_robustness_hcpa.csv", "roi_placement_quality_hcpa_b1500.csv")
_jd = _radius_pair("radius_robustness_dlbs.csv", "roi_placement_quality_dlbs_all.csv")
_lh = _jh[_jh.Subject_ID.isin(_jh.Subject_ID.value_counts()[lambda s: s >= 2].index)]
_ld = _jd[_jd.Subject_ID.isin(_jd.Subject_ID.value_counts()[lambda s: s >= 2].index)]
for _tag, _lon, _want in (("HCP-A", _lh, (0.950, 0.955, 0.941, 0.949)),
                          ("DLBS", _ld, (0.573, 0.637, 0.379, 0.518))):
    for _c, _w in zip(("classic", "classic5", "refined", "refined5"), _want):
        check(f"{_tag} {_c} ICC at its radius", _w,
              variance_components(_lon.dropna(subset=[_c]), _c)["icc"], tol=0.015)
for _tag, _j, _small, _big in (("HCP-A", _jh, 14.4, 19.5), ("DLBS", _jd, 12.5, 18.6)):
    check(f"{_tag} off-tract at 2.5 mm (%)", _small,
          float(_j.slf_off_tract_x.median() * 100), tol=0.03)
    check(f"{_tag} off-tract at 5 mm (%)", _big,
          float(_j.slf_off_tract_y.median() * 100), tol=0.03)
# the claim the section rests on: the smaller region is the cleaner one, in both
check("smaller region is less contaminated", 1.0,
      float(_jh.slf_off_tract_x.median() < _jh.slf_off_tract_y.median()
            and _jd.slf_off_tract_x.median() < _jd.slf_off_tract_y.median()) + 0.0, tol=1e-9)

# --- every geometric term removed at once ---
_jg = pd.read_csv(HERE / "joint_geometry_adjustment.csv").set_index("adjustment")
for _row, _c, _r in (("unadjusted", -0.451, -0.381), ("all three", -0.215, -0.235)):
    check(f"DLBS age coefficient, {_row}, classic", _c, float(_jg.loc[_row, "classic"]), tol=0.02)
    check(f"DLBS age coefficient, {_row}, refined", _r, float(_jg.loc[_row, "refined"]), tol=0.02)
check("geometry absorbs about half for classic", 52.0,
      float(_jg.loc["all three", "classic_absorbed_pct"]), tol=0.03)
# the ordering reverses once geometry is removed, which the section rests on
check("refined exceeds classic after full adjustment", 1.0,
      float(abs(_jg.loc["all three", "refined"]) > abs(_jg.loc["all three", "classic"])) + 0.0,
      tol=1e-9)

# The correspondence to lambda2/lambda3 tracks residual axis error, which is the
# alpha^2 term of Section 2.7 seen across cohorts and the comparison against the
# r = 0.72 Schilling et al. report in this same cohort.
for _coh, _f, _exp in (("hcpa", "measured_pvs_axis_hcpa_b1500_all.csv",
                        {"classic": 0.880, "cross": 0.850, "v2_slab": 0.936}),
                       ("dlbs", "measured_pvs_axis_dlbs.csv",
                        {"classic": 0.427, "cross": 0.308, "v2_slab": 0.934,
                         "anat_x": 0.396})):
    _d = pd.read_csv(HERE / _f)
    _d = _d.sort_values(["Subject_ID", "Visit"]).groupby("Subject_ID").first().reset_index()
    for _v, _e in _exp.items():
        _s = _d[[_v, "pv_perp"]].dropna()
        check(f"{_coh} {_v} tracks the eigenvalue ratio", _e,
              float(stats.pearsonr(_s[_v], _s.pv_perp)[0]), tol=3e-3)

_att = " ".join(TEX.split())
check("Wright credited with the identity, and first", 1.0,
      float("Wright et al.\\ first expressed the index as" in _att
            and _att.index("Wright et al.\\ first expressed")
                < _att.index("Schilling et al.\\ state the same reduction")))
check("their equal-ratio assumption named", 1.0,
      float("equal ratios in the two tracts" in _att))
check("proportionality kept, and its reason given", 1.0,
      float(r"\propto\lambda_2/\lambda_3" in TEX
            and "is the right symbol" in " ".join(TEX.split())))
check("v2 is not the vessel direction, stated against our own variant", 1.0,
      float("tested that directly against segmented vasculature" in TEX
            and "not the" in TEX and "direction of the vessel" in TEX))
check("no claim that Schilling studied between-individual fiber direction",
      0.0, float("variability in fiber" in TEX))
check("Schilling noted to apply no orientation correction", 1.0,
      float("along fixed scanner axes with no reorientation" in TEX))
# our comparator is a ratio of regional means, matching the index; theirs is
# an ROI mean of the voxelwise ratio. The code must match what the text claims.
import inspect as _insp
import measured_pvs_axis as _mpa
_src = _insp.getsource(_mpa)
check("pv_perp is built as a ratio of regional means", 1.0,
      float("l2[mp_s].mean() + l2[ma_s].mean()" in _src
            and "l3[mp_s].mean() + l3[ma_s].mean()" in _src))
# The comparator difference is an exact covariance identity, not a Taylor
# approximation. Verified numerically here, including the two cases in which it
# vanishes, so the text cannot drift from the algebra it states.
import numpy as _np
_rng = _np.random.default_rng(0)
_cases = (("general", _rng.uniform(0.2, 0.6, 40000), _rng.uniform(1.0, 2.0, 40000)),
          ("lambda3 constant", _np.full(40000, 0.4), _rng.uniform(1.0, 2.0, 40000)),
          ("ratio constant", _rng.uniform(0.2, 0.6, 40000), _np.full(40000, 1.5)))
for _lbl, _l3, _R in _cases:
    _l2 = _R * _l3
    _lhs = float(_R.mean() - _l2.mean() / _l3.mean())
    _rhs = float(-_np.cov(_R, _l3, bias=True)[0, 1] / _l3.mean())
    check(f"gap identity is exact, {_lbl}", 1.0, float(abs(_lhs - _rhs) < 1e-12))
    if _lbl != "general":
        check(f"gap vanishes, {_lbl}", 1.0, float(abs(_lhs) < 1e-12))
# and the sign: asymmetry in the lowest-lambda3 voxels makes the plain mean larger
_l3s = _rng.uniform(0.2, 0.6, 40000)
_Rs = 2.4 - 2.0 * _l3s
check("gap is positive when asymmetry sits at low lambda3", 1.0,
      float((_Rs.mean() - (_Rs * _l3s).mean() / _l3s.mean()) > 0))
check("comparator difference stated as an exact identity", 1.0,
      float(r"\operatorname{Cov}(R,\lambda_3)" in TEX
            and "exact rather than approximate" in TEX))
check("weighted-mean reading given at the voxel level", 1.0,
      float("weighted mean of the voxelwise ratios" in TEX))
check("conditions for the gap to vanish are stated", 1.0,
      float("uncorrelated with $\\lambda_3$ inside the region" in TEX))
# sorting bias: sorted eigenvalues give lambda2/lambda3 a noise floor above 1.
# The artifact predicts a LARGER ratio in poorer data, so it cannot produce the
# negative age association. Checked against the analysis, not asserted.
_sb = pd.read_csv(HERE / "sorting_bias_check.csv")
_qa = _sb[_sb.test == "quality_vs_age"].set_index("quality").r
check("motion rises with age in HCP-A", 0.086, float(_qa["motion_rms"]), tol=2e-3)
check("outlier slices rise with age in HCP-A", 0.138, float(_qa["pct_outliers"]), tol=2e-3)
_iq = _sb[_sb.test == "index_vs_quality_given_age"]
check("no variant rises with poorer data as the artifact requires", 0.0,
      float((_iq.p < 0.05).sum()))
check("quality effects on the index are bounded", 1.0,
      float(_iq.r.abs().max() < 0.07))
_ag = _sb[(_sb.test == "age_given_quality") & (_sb.variant == "pv_perp")].iloc[0]
check("ratio age association before quality adjustment", -0.583,
      float(_ag.raw), tol=2e-3)
check("ratio age association after quality adjustment", -0.576,
      float(_ag.r), tol=2e-3)
check("quality adjustment does not explain the age effect", 1.0,
      float(abs(_ag.r - _ag.raw) < 0.02 and _ag.r < 0))
_fl = pd.read_csv(HERE / "sorting_bias_floor.csv")
def _rec(_s, _tr):
    return float(_fl[(_fl.snr == _s) & (_fl.true_ratio == _tr)].recovered.iloc[0])
check("noise floor at SNR 20, true ratio 1", 1.102, _rec(20, 1.0), tol=3e-3)
check("noise floor at SNR 30, true ratio 1", 1.066, _rec(30, 1.0), tol=3e-3)
# bounded, not relative: these are ~0.005 and a relative tolerance is meaningless
check("bias at SNR 20, true ratio 1.5, is under 0.02", 1.0,
      float(0 <= _rec(20, 1.5) - 1.5 < 0.02))
check("bias at SNR 30, true ratio 1.5, is under 0.02", 1.0,
      float(0 <= _rec(30, 1.5) - 1.5 < 0.02))
check("floor at degeneracy exceeds the bias at 1.5 by 10x", 1.0,
      float((_rec(20, 1.0) - 1.0) > 10 * (_rec(20, 1.5) - 1.5)))
check("bias falls as the pair leaves degeneracy", 1.0,
      float(all(abs(_rec(_s, 1.5) - 1.5) < abs(_rec(_s, 1.0) - 1.0)
                for _s in (10, 20, 30, 50))))
check("bound stated without the redundant monotonicity apparatus", 0.0,
      float("mathrm{d}R" in TEX or "not a tendency but a bound" in TEX))
check("bound, equality condition and rate all retained", 1.0,
      float("le\\lambda_2" in TEX and "ge\\lambda_3" in TEX
            and "O(\\alpha^4)" in TEX))
check("equality ruled out on invariance grounds, before any data", 1.0,
      float("cannot always be equal" in " ".join(TEX.split())
            and "so rotation leaves it unchanged" in " ".join(TEX.split())))
check("both conditions for the bound are stated up front", 1.0,
      float("two conditions govern the approach" in " ".join(TEX.split())
            and "not bounded by the ratio at all" in " ".join(TEX.split())))
check("the invariance argument precedes the bound", 1.0,
      float(TEX.index("cannot always be equal") < TEX.index("as a bound")))
check("reduction stated in the Introduction", 1.0,
      float(TEX.index("reduces to the ratio of the eigenvalues") < TEX.index("section{Methods")))
check("degenerate-perturbation mechanism given", 1.0,
      float("separate by $" in TEX and "4c^2" in TEX
            and "first order in the noise standard" in TEX))
# Our term is radial anisotropy, in the axial/radial convention. Schilling and
# Wright call the same quantity radial asymmetry, credited but not adopted,
# because this paper also reports real left-right differences.
check("named in the axial/radial convention and credited", 1.0,
      float("radial anisotropy" in TEX
            and "call the same quantity radial asymmetry" in TEX
            and r"\cite{ref7,ref27}" in TEX
            and "radial diffusion anisotropy" not in TEX))
check("floor measured, not assumed", 1.0,
      float("the floor was measured rather than assumed" in TEX))
check("sorting bias treated in the manuscript", 1.0,
      float("holds by construction rather than by anatomy" in TEX
            and "carries a floor above one" in TEX))
check("pooling stated as a weighted average, not a bare ratio", 1.0,
      float("a ratio of sums is" in TEX and "weighted average of the two regional" in TEX))
check("our contribution scoped to the misaligned case", 1.0,
      float("as a bound" in " ".join(TEX.split())
            or "is bounded by" in " ".join(TEX.split())))
check("no claim that Schilling gave no derivation", 0.0,
      float("derivation not given there" in TEX
            or "derivation they did not give" in TEX))
for _s in (r"$r=0.56$ in HCP and", r"$r=0.72$ in HCP-A, is measured in a cohort",
           r"classic falls to $0.427$", r"the measured axis holds at $0.934$"):
    check("manuscript states " + _s[:34], 1.0, float(_s in TEX))

# LD-ALPS is an independent construction and must land inside the bound, or the
# bound is an artifact of how we estimate axes. Read from data, not asserted.
_ld = pd.read_csv(HERE / "measured_pvs_axis_dlbs.csv")
_lo = pd.read_csv(HERE / "ld_alps_dlbs.csv")[["Subject_ID", "Visit", "ALPS_overall"]]
_lo = _lo.rename(columns={"ALPS_overall": "ld_alps"})
for _f in (_ld, _lo):
    _f["Subject_ID"] = _f.Subject_ID.astype(str)
    _f["Visit"] = _f.Visit.astype(str)
_ld = _ld.merge(_lo, on=["Subject_ID", "Visit"], how="inner")
_ld = _ld.sort_values(["Subject_ID", "Visit"]).groupby("Subject_ID").first().reset_index()
_ld = _ld.dropna(subset=["ld_alps", "pv_perp"])
check("LD-ALPS mean in DLBS", 1.499, float(_ld.ld_alps.mean()), tol=3e-3)
check("LD-ALPS sits below the eigenvalue ratio", 1.0,
      float(_ld.ld_alps.mean() < _ld.pv_perp.mean()))
check("LD-ALPS bound violations match a regional axis", 0.64,
      float((_ld.ld_alps > _ld.pv_perp + 1e-9).mean() * 100), tol=0.15)
check("LD-ALPS tracks the ratio no better than the cross product", 1.0,
      float(abs(stats.pearsonr(_ld.ld_alps, _ld.pv_perp)[0] - 0.316) < 0.01))
_bl = pd.read_csv(HERE / "beyond_eigenvalue_ratio.csv")
_r = _bl[(_bl.cohort == "dlbs") & (_bl.variant == "ld_alps")].iloc[0]
check("LD-ALPS retains nothing significant beyond the ratio", 1.0,
      float(_r.p > 0.05))
check("LD-ALPS presented as independent validation of the bound", 1.0,
      float("One member of the family is not ours" in TEX
            and "rather than of how" in TEX))
# The Discussion subsection this used to guard was cut as a restatement of the
# bound; the guard moved with the surviving sentences into the Results, which is
# where the variants are now read as a family rather than recommended.
check("variant section describes rather than prescribes", 1.0,
      float("rather than as candidate methods" in TEX
            and "menu but a series" in TEX))
check("no recommendation is drawn", 1.0,
      float("no recommendation about which quantity to compute" in " ".join(TEX.split())))

# ALPS-PAS: the second published method. Per-voxel eigenvector selection means
# the per-voxel inequality holds exactly, so it must never exceed the bound.
_tn = pd.read_csv(HERE / "tn_alps.csv").dropna(
    subset=["ALPS-PAS", "pv_perp", "cross", "v2_sphere"])
check("ALPS-PAS mean in the trigeminal cohort", 1.609,
      float(_tn["ALPS-PAS"].mean()), tol=3e-3)
check("ALPS-PAS never exceeds the bound", 0.0,
      float((_tn["ALPS-PAS"] > _tn.pv_perp + 1e-9).sum()))
check("ALPS-PAS tracks the ratio more closely than any variant of ours", 1.0,
      float(stats.pearsonr(_tn["ALPS-PAS"], _tn.pv_perp)[0] > 0.95))
check("regional axes violate at a dispersion-set rate, cross", 0.58,
      float((_tn.cross > _tn.pv_perp + 1e-9).mean() * 100), tol=0.15)
check("regional axes violate at a dispersion-set rate, sphere", 11.11,
      float((_tn.v2_sphere > _tn.pv_perp + 1e-9).mean() * 100), tol=0.3)
_bp = pd.read_csv(HERE / "beyond_eigenvalue_ratio.csv")
_pas = _bp[(_bp.cohort == "trigeminal") & (_bp.variant == "ALPS-PAS")].iloc[0]
check("ALPS-PAS retains nothing beyond the ratio", 1.0, float(_pas.p > 0.5))
# Guarded the old phrasing "Read together the two published methods", which was
# ungrammatical (plural verb after "neither"). The claim is what matters: both
# published methods are presented together, and neither was built around the bound.
check("both published methods presented together", 1.0,
      float("the second published method in the family" in TEX
            and "The two published methods" in TEX
            and "Neither was built around this bound" in TEX))

# anat_x corrects in-plane rotation only, so it should track the ratio where the
# residual error is small and fail where the dominant error is pitch.
for _c, _f, _e in (("hcpa", "measured_pvs_axis_hcpa_b1500_all.csv", 0.931),
                   ("dlbs", "measured_pvs_axis_dlbs.csv", 0.396)):
    _d = pd.read_csv(HERE / _f)
    _d = _d.sort_values(["Subject_ID", "Visit"]).groupby("Subject_ID").first().reset_index()
    _s = _d[["anat_x", "pv_perp"]].dropna()
    check(f"anat_x tracks the ratio in {_c}", _e,
          float(stats.pearsonr(_s.anat_x, _s.pv_perp)[0]), tol=3e-3)
check("anat_x split attributed to pitch, not only anatomy", 1.0,
      float("the dominant misalignment in DLBS is" in TEX
            and "outside that plane" in TEX))

# The second-order rate, checked against measured angles rather than assumed.
# v2_slab's own alpha is zero by construction, so its shortfall is dispersion;
# the others should be that in quadrature with their own misalignment.
_sd = pd.read_csv(HERE / 'shortfall_decomposition.csv').set_index('variant')
check('dispersion reference, effective angle', 16.21,
      float(_sd.loc['v2_slab', 'effective_deg']), tol=2e-3)
for _v, _obs, _pred in (('anat_x', 18.10, 18.50), ('cross', 20.75, 19.97)):
    check(f'{_v} effective angle', _obs,
          float(_sd.loc[_v, 'effective_deg']), tol=3e-3)
    check(f'{_v} quadrature prediction', _pred,
          float(_sd.loc[_v, 'predicted_deg']), tol=3e-3)
    check(f'{_v} quadrature agrees within one degree', 1.0,
          float(abs(_sd.loc[_v, 'effective_deg']
                    - _sd.loc[_v, 'predicted_deg']) < 1.0))
check('classic excluded from the quadrature test', 1.0,
      float('not tract-locked' in str(_sd.loc['classic', 'role'])))
check('shortfall decomposition stated in the manuscript', 1.0,
      float('should combine in quadrature' in TEX
            and 'reported only for scale' in TEX))
check('ceiling on better tract estimation stated', 1.0,
      float('Neither reaches' in TEX and 'neither can' in TEX))
# The recommendation must be the same quantity everywhere. These fail if the
# earlier directional recommendation returns anywhere in the paper.
for _stale in ('keep the directional form and recommend it',
               'We therefore recommend the eigenvalue ratio',
               'should compute the eigenvalue',
               'None of this is an argument for replacing the index with the ratio',
               'the formulation we adopted'):
    check(f'stale recommendation absent: {_stale[:34]}', 0.0,
          float(_stale in TEX))
check('the remainder is stated to have no basis for interpretation', 1.0,
      float('no basis for an interpretation' in TEX))
check('the Conclusion states the bound', 1.0,
      float('bounded above by' in TEX[TEX.index('section{Conclusion'):]
            and 'second order in the angular error'
            in TEX[TEX.index('section{Conclusion'):]))
check('the Conclusion keeps the head-position finding', 1.0,
      float('Head position in the scanner is not random'
            in TEX[TEX.index('section{Conclusion'):]))
check('the Conclusion credits the two published corrections', 1.0,
      float('published corrections behave the same way'
            in TEX[TEX.index('section{Conclusion'):]))
# Equality needs alpha = 0 per voxel, which a regional axis cannot give. The
# violation rates must therefore split by construction: per-voxel variants never
# exceed the bound, regional ones do at a rate set by dispersion.
_eq = pd.read_csv(HERE / 'tn_alps.csv').dropna(
    subset=['ALPS-PAS', 'per-voxel', 'pv_perp', 'cross', 'v2_sphere'])
for _v in ('ALPS-PAS', 'per-voxel'):
    check(f'per-voxel selection never exceeds the bound: {_v}', 0.0,
          float((_eq[_v] > _eq.pv_perp + 1e-9).sum()))
for _v in ('cross', 'v2_sphere'):
    check(f'regional axis does exceed it sometimes: {_v}', 1.0,
          float((_eq[_v] > _eq.pv_perp + 1e-9).sum() > 0))
check('pooled v2 still falls short of the bound', 1.0,
      float(0.90 < float((pd.read_csv(HERE / 'measured_pvs_axis_dlbs.csv')
                          .eval('v2_slab / pv_perp')).median()) < 0.93))
check('equality condition stated as per-voxel, not on average', 1.0,
      float('in every voxel, not on average' in TEX
            and 'cannot satisfy this' in TEX))
check('the second condition on pooling is stated', 1.0,
      float('equals one number' in TEX
            and 'neither of which holds in tissue' in TEX))
# alpha must be defined before it is used anywhere in the manuscript
_def = TEX.find("Let $\\alpha$ be the angle")
_use = TEX.find("$\\alpha$")
check("alpha is defined in the text", 1.0, float(_def > 0))
check("alpha is defined before its first use", 1.0, float(0 < _def <= _use))
check("alpha is defined against v2, not the scanner or the vessel", 1.0,
      float("against the second" in TEX and "rather than against the scanner" in TEX))

# The AC-PC claim must not outrun its source. Taoka et al. 2024 state that the
# transverse plane is conventionally taken at that line and that it is the
# standard for ALPS evaluation. They do not state what the 2017 report recorded.
_flat = " ".join(TEX.split())
for _over in ("The originators aligned the slice prescription",
              "not stated in the original report"):
    check(f"unsourced AC-PC claim absent: {_over[:32]}", 0.0, float(_over in _flat))
check("Taoka credited for the acquisition-plane statement", 1.0,
      float("should be performed on images acquired with the plane along"
            in " ".join(TEX.split())))

check("the 2017 original method paper is cited", 1.0,
      float("bibitem{ref26}" in TEX and "ref26}" in TEX.split("bibitem{ref26}")[0]
            and "10.1007/s11604-017-0617-z" in TEX))

# Prescribing the imaging plane is not a correction: nothing is undone. It shows
# orientation was considered at acquisition. Coauthor's distinction, guarded here
# because the wrong framing was reintroduced once already by a merge.
check("acquisition prescription not called a correction", 0.0,
      float("orientation correction performed at acquisition" in _flat))

# AC-PC alignment is an assumption the method carries, and the headers say it is
# not satisfied. Compensation would need a slope near +1; it is negative.
_ac = pd.read_csv(HERE / 'acpc_assumption.csv').set_index('basis')
check('head pitch, per participant', 10.50,
      float(_ac.loc['per_participant', 'head_pitch']), tol=3e-3)
check('prescribed pitch, per participant', 1.64,
      float(_ac.loc['per_participant', 'slab_pitch']), tol=5e-3)
check('prescription does not compensate: slope is negative', 1.0,
      float(_ac.loc['per_participant', 'slope'] < 0))
check('compensation slope, per participant', -0.254,
      float(_ac.loc['per_participant', 'slope']), tol=5e-3)
check('residual exceeds the head pitch it should remove', 1.0,
      float(_ac.loc['per_participant', 'residual']
            > _ac.loc['per_participant', 'head_pitch']))
check('near half of slabs are laid within a degree of axial', 47.1,
      float(_ac.loc['per_participant', 'near_zero_pct']), tol=5e-3)
_rp = _ac.loc['repeat_visits']
check('prescribed pitch ICC across visits', 0.498, float(_rp.icc_slab), tol=5e-3)
check('head pitch ICC across visits', 0.549, float(_rp.icc_head), tol=5e-3)
check('between-visit change exceeds the prescription itself', 1.0,
      float(_rp.median_change > _rp.slab_pitch))
check('prescription no steadier than the head position', 1.0,
      float(_rp.icc_slab < _rp.icc_head))
check('research standard framing, not clinical excuse', 1.0,
      float('held to a higher standard' in _flat
            and 'not met there' in _flat))
check('per-session result agrees in sign', 1.0,
      float(_ac.loc['per_session', 'slope'] < 0))
check('AC-PC stated as the method intent, not observed practice', 1.0,
      float('evaluation in the ALPS method should be performed' in _flat))
check('no claim that AC-PC prescription is conventional practice', 0.0,
      float('conventionally prescribed' in _flat or 'A prescription convention' in _flat))
check('the assumption is tested, not asserted', 1.0,
      float('whether it does what the method assumes' in _flat
            and 'reproducibly enough for the scanner axes to stand in' in _flat))
# Figure 3's 'two sides coincide in 72 of 78' was only ever printed by the
# figure builder. Now recomputed from the masks and checked.
_hs = pd.read_csv(HERE / 'hemisphere_slice_agreement.csv')
check('sessions with a hand-drawn mask', 78.0, float(len(_hs)))
check('sessions where the two sides share a slice', 72.0,
      float(_hs.sides_coincide.sum()))
check('caption states the coincidence count', 1.0,
      float('coincide in $72$ of $78$ sessions' in ' '.join(TEX.split())))

# Placement reliability, both sides on variance_components, the estimator the
# rest of the manuscript uses. Previously one side came from a second pipeline,
# which is how 0.559 came to sit beside 0.594 for the same quantity.
_mi = pd.read_csv(HERE / "manual_vs_atlas_icc.csv").set_index("placement")
check("hand-drawn placement ICC", 0.3783, float(_mi.loc["hand-drawn", "icc"]), tol=3e-3)
check("atlas ICC on the hand-drawn sessions", 0.6112,
      float(_mi.loc["atlas, hand-drawn sessions only", "icc"]), tol=3e-3)
check("atlas ICC on the full cohort", 0.5945, float(_mi.loc["atlas", "icc"]), tol=3e-3)
check("atlas placement is the more reliable", 1.0,
      float(_mi.loc["atlas", "icc"] > _mi.loc["hand-drawn", "icc"]))
_f = " ".join(TEX.split())
check("the second-pipeline ICC is gone", 0.0, float("$0.559$" in _f))
check("placement sentence quotes the single-estimator figures", 1.0,
      float("$0.378$" in _f and "$0.611$" in _f and "$0.594$" in _f))


# AABC Term 8b acknowledgment, from the consortium's own page. The bracketed
# [publication/press release/presentation] is a choice; ours is 'publication'.
# Compared against the rendered text so a LaTeX escape like St.\ Louis does not
# read as a change.
_AABC = (
    "Data, methods used, and/or research reported in this publication were "
    "provided in whole or in part by the Aging Adult Vulnerability and "
    "Resiliency in the Aging Adult Brain Connectome (AABC) project "
    "(U19AG073585) and the Human Connectome Project in Aging (HCP-A, "
    "U01AG052564) funded by the National Institute of Aging of the National "
    "Institutes of Health. HCP-A was further supported by funds provided by "
    "the McDonnell Center for Neuroscience at Washington University in "
    "St. Louis.")


def _rendered(path):
    s = Path(path).read_text(encoding='utf-8')
    return ' '.join(s.replace(chr(92) + ' ', ' ').split())


for _f in ('mri_revision.tex', 'mri_title_page.tex'):
    check(f'AABC acknowledgment verbatim in {_f}', 1.0,
          float(_AABC in _rendered(HERE.parent / _f)))

# The public repository is a processing platform, not the paper's analysis code.
# Data availability must not claim otherwise.
_da = ' '.join(TEX.split())
check('no claim that the analysis scripts are in the public package', 0.0,
      float('analysis scripts for every number reported here' in _da))
check('data availability states what the package does contain', 1.0,
      float('An implementation of every variant reported here is available' in _da))
check('the regeneration limit is stated', 1.0,
      float('cannot be regenerated from that package alone' in _da))

check('data availability cites the analysis repository', 1.0,
      float('github.com/snhwang/dti-alps-analysis' in ' '.join(TEX.split())))
check('no stale on-request claim for the scripts', 0.0,
      float('analysis scripts, and derived values' in ' '.join(TEX.split())))

_ov = ' '.join(TEX.split())
check('Methods does not restate the Introduction', 0.0,
      float('the refinement is to measure rather than assume' in _ov))

# Repositioning sensitivity, on the sample Section 3.5 and Figure 7 report.
# Needs three conditions together: sphere cohort, the DLBS analysis
# participants, and motion QC at 0.5 mm. No single script applied all three,
# which is why repositioning_table_manual.csv (78 obs, 19 subjects) looked like
# the source and is not.
_rq = pd.read_csv(HERE / 'repositioning_sphere_qc.csv').set_index('metric')
check('repositioning sample, observations', 580.0,
      float(_rq.loc['Classic', 'n_obs']))
check('repositioning sample, participants', 156.0,
      float(_rq.loc['Classic', 'n_subjects']))
check('classic slope per degree', 0.783,
      float(_rq.loc['Classic', 'slope_pct_per_deg']), tol=5e-3)
check('refined slope per degree', 0.260,
      float(_rq.loc['Refined', 'slope_pct_per_deg']), tol=5e-3)
check('classic interval excludes zero', 1.0,
      float(_rq.loc['Classic', 'ci_lo'] > 0))
check('refined interval contains zero', 1.0,
      float(_rq.loc['Refined', 'ci_lo'] < 0 < _rq.loc['Refined', 'ci_hi']))
_f = ' '.join(TEX.split())
check('manuscript quotes the reproduced slope', 1.0,
      float('$+0.78\\%$ per degree' in _f))

# The interpretation section concedes the discriminative literature before
# contesting its attribution, and reframes attribution rather than practice.
_rf = ' '.join(TEX.split())
check('discriminative findings conceded first', 1.0,
      float('The index discriminates, and that is not in question here' in _rf))
check('reframe is about attribution, not practice', 1.0,
      float('about attribution, not about practice' in _rf
            and 'Reported group differences stand' in _rf))
check('no recommendation on which quantity to compute', 1.0,
      float('no recommendation about which quantity to compute' in _rf))

# The abstract must credit the known relationship and must not recommend a
# practice the Discussion declines to recommend.
_ab = ' '.join(TEX.split())
_ab = _ab[_ab.index('begin{abstract}') + len('begin{abstract}'):_ab.index('end{abstract}')]
check('abstract credits the reported relationship', 1.0,
      float('Schilling et al.' in _ab and 'report that the index tracks' in _ab))
check('abstract makes no practice recommendation', 0.0,
      float('should be computed along measured tract' in _ab
            or 'captures it more simply' in _ab))
check('abstract states the attribution reframe', 1.0,
      float('Reported group differences stand' in _ab))
check('abstract within 250 words', 1.0, float(len(_ab.split()) <= 250)),

# The bound is a Theory section preceding Methods, not a Methods subsection.
_th = TEX.index('section{Theory}')
check('Theory precedes Methods', 1.0, float(_th < TEX.index('section{Methods}')))
check('Theory follows the Introduction', 1.0,
      float(TEX.index('section{Introduction') < _th))
check('the measured-axis construction sits with the methods', 1.0,
      float(TEX.index('The cross product of Section') > TEX.index('section{Methods}')))
check('the stale label is gone', 0.0, float('sec:measured-axis' in TEX))

# Wright's route was observation, not assumption, and their premise concerns v2
# rather than the vessel. Both distinctions are load-bearing for the attribution.
_wr = ' '.join(TEX.split())
check('Wright credited with an observed alignment', 1.0,
      float('spatially coherent and aligned with $x$ in both tracts' in _wr))
check('identity separated from its perivascular reading', 1.0,
      float('bears on the perivascular reading of the identity and not on' in _wr))

# The regional ratio is formed as a ratio of means, as the index is. Forming it
# as a mean of voxelwise ratios let near-zero lambda3 voxels dominate and put the
# body of the corpus callosum at a median of 13 with a maximum of 924.
_al = pd.read_csv(HERE / 'alps_location_special.csv')
_rat = [c for c in _al.columns if not c.endswith(' CP') and c not in ('sid', 'Age')]
check('every regional ratio is physiological', 1.0,
      float(all(_al[c].max() < 4 for c in _rat)))
check('twelve regions compared', 12.0, float(len(_rat)))
from scipy import stats as _st
_ages = {}
for _c in _rat:
    _s = _al[[_c, 'Age']].dropna()
    _ages[_c] = float(_st.pearsonr(_s[_c], _s.Age)[0])
check('the ratio falls with age in every region', 12.0,
      float(sum(v < 0 for v in _ages.values())))
check('the genu falls faster than the ALPS projection region', 1.0,
      float(_ages['Genu CC'] < _ages['SCR (ALPS proj)']))
check('genu age correlation', -0.762, _ages['Genu CC'], tol=3e-3)
check('the widespread decline is credited to Wright', 1.0,
      float('Wright et al.' in ' '.join(TEX.split())
            and 'falls with age across white matter generally' in ' '.join(TEX.split())))
check('our version is reported as agreement, not as a finding', 1.0,
      float('Our twelve-label comparison agrees' in ' '.join(TEX.split())))

# Novelty claims must not outrun prior work. Burles et al. showed pitch differs
# by sex, so 'has not been asked' was contradicted by our own later citation.
_nv = ' '.join(TEX.split())
for _over in ('has not been asked', 'not previously been tested',
              'nobody has', 'has never been'):
    check(f'no unqualified novelty claim: {_over[:28]}', 0.0, float(_over in _nv))
check('Burles credited for the sex finding up front', 1.0,
      float('pitch differs by sex in ADNI' in _nv))
check('our question scoped to age and clinical group', 1.0,
      float('covaries with age and with clinical group' in _nv))

# Wright published the v2-coherence observation and the noise argument first.
# We had credited both to Schilling. Third instance of the same slip in one audit.
_co = ' '.join(TEX.split())
check('coherence argument credited to Wright', 1.0,
      float('Wright et al.\ report the second eigenvector to be spatially coherent'
            in _co))
check('Schilling keeps the quantification and high-b result', 1.0,
      float('quantify that coherence' in _co))


# The two-region geometry, Appendix A. The claims there are analytic, so these
# check identities rather than sampled values: the closed forms must reproduce
# direct tensor rotation, pitch must be signed, roll must equal yaw, and the
# stated sign-change condition must predict the sign in every case.
_tr = pd.read_csv(HERE / 'tract_orthogonality_rotation.csv')

check('closed forms match direct rotation to machine precision', 1.0,
      float(_tr.closed_minus_direct.abs().max() < 1e-12))
_moving = _tr[_tr.degrees > 0]
check('pitch lowers the index at every ratio and angle', 1.0,
      float((_moving[_moving.rotation == 'pitch']['pct_change'] < 0).all()))
_ry = _moving.pivot_table(index=['rho', 'degrees'], columns='rotation',
                          values='index')
check('roll equals yaw exactly', 1.0,
      float((_ry['roll'] - _ry['yaw']).abs().max() < 1e-12))

# kappa > 1 always, so -(kappa - 1) is strictly negative. The transverse
# coefficient changes sign at kappa = rho^2 + rho - 1, which is the manuscript's
# stated condition, and it has to agree with the sign actually observed.
for _rho in sorted(_tr.rho.unique()):
    _k = float(_tr[_tr.rho == _rho].kappa.iloc[0])
    _obs = _tr[(_tr.rho == _rho) & (_tr.degrees == 15)
               & (_tr.rotation == 'roll')]['pct_change'].iloc[0]
    check(f'sign condition predicts roll at rho {_rho}', 1.0,
          float((_k > _rho ** 2 + _rho - 1) == (_obs > 0)))

_r15 = _tr[(_tr.rho == 1.5) & (_tr.degrees == 15)].set_index('rotation')
check('pitch at 15 degrees, rho 1.5', -11.81,
      float(_r15.loc['pitch', 'pct_change']), tol=2e-3)
check('roll and yaw at 15 degrees, rho 1.5', 0.55,
      float(_r15.loc['roll', 'pct_change']), tol=1e-2)
_kap = float(_tr[_tr.rho == 1.5].kappa.iloc[0])
check('coefficient ratio at rho 1.5', 24.0,
      abs(-(_kap - 1) / ((_kap + 1 - 1.5 - 1.5 ** 2) / 3.0)), tol=1e-2)
for _rho, _kappa in ((1.2, 2.64), (1.5, 3.00), (1.72, 3.26), (2.0, 3.60)):
    check(f'kappa tabulated at rho {_rho}', _kappa,
          float(_tr[_tr.rho == _rho].kappa.iloc[0]), tol=3e-3)

_ta = pd.read_csv(HERE / 'tract_orthogonality_alignment.csv')
_ta = _ta.set_index(['cohort', 'measure'])
for _coh, _x, _cr in (('HCP-A', 7.01, 8.79), ('DLBS', 8.93, 11.68)):
    check(f'{_coh} v2 to scanner x', _x,
          float(_ta.loc[(_coh, 'v2_to_x'), 'median_deg']), tol=2e-3)
    check(f'{_coh} v2 to common perpendicular', _cr,
          float(_ta.loc[(_coh, 'v2_to_cross'), 'median_deg']), tol=2e-3)
    check(f'{_coh} common perpendicular is the worse axis', 1.0,
          float(_ta.loc[(_coh, 'v2_to_cross'), 'median_deg']
                > _ta.loc[(_coh, 'v2_to_x'), 'median_deg']))

# The DLBS medians are the misalignment terms the shortfall decomposition uses,
# so the two tables have to agree or one of them is stale.
check('alignment table agrees with the shortfall decomposition, anat_x',
      float(_sd.loc['anat_x', 'misalign_deg']),
      float(_ta.loc[('DLBS', 'v2_to_x'), 'median_deg']), tol=3e-3)
check('alignment table agrees with the shortfall decomposition, cross',
      float(_sd.loc['cross', 'misalign_deg']),
      float(_ta.loc[('DLBS', 'v2_to_cross'), 'median_deg']), tol=3e-3)

# The two-region sufficient statistic. R depends on the two misalignments only
# through S, and a separation delta between the regional v2 directions costs
# exactly what one region at delta/2 would. Both are identities, so both are
# re-derived here rather than read from a file.
def _R_of_S(S, rho=1.72):
    return (2 * rho - (rho - 1) * S) / (2 + (rho - 1) * S)


def _R_two(ap, aa, rho=1.72):
    """Ratio of sums, built region by region."""
    _n = (rho * np.cos(ap) ** 2 + np.sin(ap) ** 2
          + rho * np.cos(aa) ** 2 + np.sin(aa) ** 2)
    _d = (rho * np.sin(ap) ** 2 + np.cos(ap) ** 2
          + rho * np.sin(aa) ** 2 + np.cos(aa) ** 2)
    return _n / _d


_rng = np.random.default_rng(0)
_wa, _wb = 0.0, 0.0
for _ in range(4000):
    _ap, _aa = _rng.uniform(0, np.pi / 2, 2)
    _S = np.sin(_ap) ** 2 + np.sin(_aa) ** 2
    _wa = max(_wa, abs(_R_two(_ap, _aa) - _R_of_S(_S)))
check('R depends on the two angles only through S', 1.0, float(_wa < 1e-12))

for _deg in (5, 10, 14.617, 17.14, 20, 30, 45):
    _d = np.radians(_deg)
    _half = _d / 2
    _single = ((1.72 * np.cos(_half) ** 2 + np.sin(_half) ** 2)
               / (1.72 * np.sin(_half) ** 2 + np.cos(_half) ** 2))
    _wb = max(_wb, abs(_R_of_S(1 - np.cos(_d)) - _single))
check('a delta separation costs exactly what delta/2 costs, exactly', 1.0,
      float(_wb < 1e-12))

_tf = pd.read_csv(HERE / 'tract_orthogonality_floor.csv').set_index('cohort')
for _coh, _delta, _floor, _fc, _cc in (('HCP-A', 14.617, 7.309, 1.82, 4.26),
                                       ('DLBS', 17.139, 8.570, 2.49, 6.50)):
    check(f'{_coh} regional v2 separation', _delta,
          float(_tf.loc[_coh, 'v2_disagreement_deg']), tol=3e-3)
    check(f'{_coh} floor is half the separation', _floor,
          float(_tf.loc[_coh, 'floor_deg']), tol=3e-3)
    check(f'{_coh} floor cost', _fc,
          float(_tf.loc[_coh, 'floor_cost_pct']), tol=1e-2)
    check(f'{_coh} cross cost', _cc,
          float(_tf.loc[_coh, 'cross_cost_pct']), tol=1e-2)
    check(f'{_coh} the determined axis does not reach the floor', 1.0,
          float(_tf.loc[_coh, 'cross_cost_pct']
                > _tf.loc[_coh, 'floor_cost_pct']))

_tg = ' '.join(TEX.split())
check('the pitch result is given in closed form', 1.0,
      float('(\\kappa - 1)\\sin^2\\theta' in _tg))
check('the sign of the pitch effect is argued, not sampled', 1.0,
      float('Pitch always lowers the index, with no tissue exception' in _tg))
check('roll-yaw equality is proved, not observed', 1.0,
      float('proves their equality rather than observing it' in _tg))
check('the transverse sign change is stated', 1.0,
      float('\\kappa = \\rho^2 + \\rho - 1' in _tg))
check('signed bias is distinguished from scatter', 1.0,
      float('reappears as an effect on that covariate' in _tg))
check('the sufficient statistic is given', 1.0,
      float('S = \\sin^2\\alpha_{\\text{p}} + \\sin^2\\alpha_{\\text{a}}' in _tg))
check('the delta/2 equality is called exact', 1.0,
      float('as an identity rather than as a small-angle approximation' in _tg))
check('the floor is stated as a measurement', 1.0,
      float('median $14.6^{\\circ}$ apart' in _tg))

check('the variant family is closed, not sampled', 1.0,
      float('The same algebra closes the family' in _tg
            and 'No third behavior is available' in _tg))
check('closure names both angles and their opposite effects', 1.0,
      float('a tilted numerator lifts the index above the ratio' in _tg))
check('existence of a shared axis is not confused with alignment', 1.0,
      float('determined but not aligned' in _tg))
check('the section disclaims the perivascular premise', 1.0,
      float('rather than from any perivascular premise' in _tg))

print(f"{'':4s} {'check':<52s} {'claimed':>10s} {'actual':>10s}")
nfail = 0
for ok, label, claimed, actual in results:
    mark = "ok  " if ok else "FAIL"
    nfail += not ok
    c = f"{claimed:10.3f}" if isinstance(claimed, float) else f"{'':>10s}"
    a = f"{actual:10.3f}" if isinstance(actual, float) else f"{'':>10s}"
    print(f"{mark} {label:<52s} {c} {a}")
print(f"\n{len(results) - nfail}/{len(results)} passed"
      + ("" if nfail == 0 else f"  --  {nfail} DISAGREEMENTS"))
