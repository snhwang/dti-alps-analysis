"""Does ALPS, or the radial anisotropy it reduces to, track anything within a
participant over time?

The cross-sectional sweep collapsed every participant to their first session and
found that DTI-ALPS associates with about half of an aging cohort's phenotypes
until age and sex enter, and with nothing afterwards. That leaves the more
interesting question untouched, because both cohorts have repeat visits that the
sweep discarded. HCP-A has 628 of 809 participants with more than one session
and DLBS has all 156.

The design here is within-participant, which is strictly stronger than the
cross-sectional one. Each participant's index and phenotype are centered on
their own mean, so every time-invariant confound cancels exactly: age at
baseline, sex, education, head size, habitual posture, scanner, and anything
else that does not change within a person. What remains is the question worth
asking. When this person's index moved, did their phenotype move with it?

One control is essential. Within a participant, visits advance in time, so the
index and the phenotype both drift with age and would correlate for that reason
alone. Within-person centered age is therefore partialled out of every test, and
what is reported is the association net of shared drift.

The radial anisotropy ratio is included alongside the variants, since the paper
argues every variant is bounded by it and approaches it. It is pv_perp, the
mean lambda2 over the mean lambda3 within each hemisphere's two regions,
averaged across hemispheres exactly as the variants are.

    python phenotype_longitudinal.py --cohort hcpa
    python phenotype_longitudinal.py --cohort dlbs
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
DIFF = HERE.parent.parent / "diffusion"
AABC = DIFF / "HCP" / "AABC2_subjects_2026_02_05_14_29_11.csv"
DLBS_TSV = DIFF / "DLBS" / "ds004856_participants.tsv"
VARIANTS = ["classic", "cross", "v2_sphere", "v2_slab", "pv_perp", "anat_x", "ratio"]
# The one thing a directional index can ask that the rotation-invariant ratio
# cannot: does the fast perpendicular direction agree between two tracts whose
# fibers run differently? Agreement across tracts is evidence of a common
# structure, whereas a large l2/l3 in a single voxel is not. These angles are
# carried alongside the variants so the question can be tested rather than
# argued, and the age+ratio arm asks whether they add anything once the ratio
# itself is accounted for.
ANGLES = ["v2_proj_to_assoc", "v2_to_x", "cross_to_x"]
# Identifiers, dates and administrative fields are not phenotypes. Age at each
# measurement is excluded too: it is the covariate being partialled out, so
# testing the index against it is close to circular and its apparent
# significance is an artifact of the design rather than a finding.
DROP = re.compile(r"id|date|visit|event|site|scanner|version|complete|_dt$|days"
                  r"|^age", re.I)


def bh(p):
    p = np.asarray(p, float)
    ok = ~np.isnan(p)
    q = np.full_like(p, np.nan)
    v = p[ok]
    o = np.argsort(v)
    adj = np.empty_like(v)
    adj[o] = np.minimum.accumulate(
        (v[o] * len(v) / np.arange(1, len(v) + 1))[::-1])[::-1]
    q[ok] = np.clip(adj, 0, 1)
    return q


def index_table(cohort: str) -> pd.DataFrame:
    # The canonical tables are the re-sphered ones, where every region is a
    # true sphere of fixed radius drawn in native space at the warped centre.
    # ALPS_WARPED_MASK reads the first submission's warped masks instead, so
    # the placement rule can be varied without editing anything.
    _wm = os.environ.get("ALPS_WARPED_MASK", "")
    _sx = "_warpedmask" if _wm else ""
    # HCP-A carries the _all suffix and DLBS does not, because the DLBS tables
    # are built from participants with two or more visits while the HCP-A ones
    # keep every session. Deriving one name from the other reads nothing.
    f = (f"measured_pvs_axis_hcpa_b1500_all{_sx}.csv" if cohort == "hcpa"
         else f"measured_pvs_axis_dlbs{_sx}.csv")
    if not (HERE / f).exists():
        raise SystemExit(f"{f} is not present")
    if _wm:
        print(f"using the first submission's warped masks: {f}")
    d = pd.read_csv(HERE / f)
    d["Subject_ID"] = d.Subject_ID.astype(str)
    d["Visit"] = d.Visit.astype(str)
    # pv_perp, not (l2_proj + l2_assoc) / (l3_proj + l3_assoc). The two are the
    # same quantity and agree at r = 0.9999, differing only in whether the
    # hemispheres are combined before or after the division. pv_perp divides
    # within each hemisphere and then averages, which is how every variant it
    # is compared against is built, so this keeps the comparison like for like.
    d["ratio"] = d["pv_perp"]

    # Conditioning of the tensor fit, the last alternative explanation for a
    # directional variant retaining anything. An estimated eigenvector is only
    # as good as the tensor it comes from, so a session whose fit degrades has
    # a noisier axis and a different index for reasons unrelated to biology.
    # Regional FA and mean diffusivity are computed from the same eigenvalues
    # the index uses, per region, and carried as covariates.
    for reg in ("proj", "assoc"):
        L = np.column_stack([d[f"l{i}_{reg}"].to_numpy(float) for i in (1, 2, 3)])
        md = L.mean(1)
        num = np.sqrt(((L - md[:, None]) ** 2).sum(1))
        den = np.sqrt((L ** 2).sum(1))
        d[f"fa_{reg}"] = np.clip(np.sqrt(1.5) * np.divide(
            num, den, out=np.zeros_like(num), where=den != 0), 0, 1)
        d[f"md_{reg}"] = md
    return d


def long_phenotypes(cohort: str) -> tuple[pd.DataFrame, list[str]]:
    """One row per subject-visit, with every repeatedly measured phenotype."""
    if cohort == "hcpa":
        a = pd.read_csv(AABC, low_memory=False)
        a["Subject_ID"] = a.id_event.astype(str).str.split("_").str[0]
        a["Visit"] = a.id_event.astype(str).str.split("_").str[1]
        a = a[a.Visit.isin(["V1", "V2", "V3", "V4"])]
        num = [c for c in a.columns
               if pd.api.types.is_numeric_dtype(a[c]) and not DROP.search(c)
               and a[c].notna().sum() >= 100 and a[c].nunique() > 4]
        return a[["Subject_ID", "Visit"] + num], num

    # DLBS stores repeated measures as wave-suffixed columns, so they are
    # melted into subject-visit rows to match the imaging table.
    t = pd.read_csv(DLBS_TSV, sep="\t", low_memory=False)
    t["Subject_ID"] = t.participant_id.astype(str)
    stems = {}
    for c in t.columns:
        m = re.match(r"^(.*)_W([123])$", c)
        if m and pd.api.types.is_numeric_dtype(t[c]) and not DROP.search(m.group(1)):
            stems.setdefault(m.group(1), {})[f"ses-wave{m.group(2)}"] = c
    rows = []
    for stem, byv in stems.items():
        if len(byv) < 2:                       # needs repeats to be longitudinal
            continue
        for v, col in byv.items():
            rows.append(t[["Subject_ID"]].assign(Visit=v, _stem=stem,
                                                 _val=t[col]))
    long = pd.concat(rows, ignore_index=True)
    wide = long.pivot_table(index=["Subject_ID", "Visit"], columns="_stem",
                            values="_val").reset_index()
    num = [c for c in wide.columns if c not in ("Subject_ID", "Visit")]
    return wide, num


def within(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Center each column on the participant's own mean."""
    g = df.groupby("Subject_ID")
    out = df.copy()
    for c in cols:
        out[c] = df[c] - g[c].transform("mean")
    return out


def partial_corr(x, y, Zc):
    """Correlation of x and y with every column of Zc regressed out of both."""
    Zc = np.atleast_2d(np.asarray(Zc, float))
    if Zc.shape[0] != len(x):
        Zc = Zc.T
    ok = ~(np.isnan(x) | np.isnan(y) | np.isnan(Zc).any(axis=1))
    x, y, z = x[ok], y[ok], Zc[ok]
    if len(x) < 20 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan, len(x)
    Z = np.column_stack([np.ones(len(z)), z])

    def rz(v):
        b, *_ = np.linalg.lstsq(Z, v, rcond=None)
        return v - Z @ b
    rx, ry = rz(x), rz(y)
    # pv_perp is the ratio, so in any arm that partials the ratio out its
    # residual is floating-point residue rather than signal. Correlating that
    # residue returns an arbitrary number that happens to look like a clean
    # zero here and looked like a significant result elsewhere. Testing
    # against exactly zero misses it, since the residue is never exactly zero,
    # so compare it against the variable's own spread. A collapsed residual is
    # then reported as undefined instead of as a variant that retains nothing.
    if (np.std(rx) <= 1e-8 * max(np.std(x), 1e-30)
            or np.std(ry) <= 1e-8 * max(np.std(y), 1e-30)):
        return np.nan, np.nan, len(x)
    r = float(np.corrcoef(rx, ry)[0, 1])
    dof = len(x) - Z.shape[1] - 1
    t = r * np.sqrt(dof / max(1 - r ** 2, 1e-12))
    return r, float(2 * stats.t.sf(abs(t), dof)), len(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=["hcpa", "dlbs"], default="hcpa")
    args = ap.parse_args()

    d = index_table(args.cohort)
    ph, num = long_phenotypes(args.cohort)
    ph["Subject_ID"] = ph.Subject_ID.astype(str)
    ph["Visit"] = ph.Visit.astype(str)
    m = d.merge(ph, on=["Subject_ID", "Visit"], how="inner")

    # Only participants with at least two usable sessions carry within-person
    # information; everyone else contributes exactly zero after centering.
    n_ses = m.groupby("Subject_ID").size()
    m = m[m.Subject_ID.isin(n_ses[n_ses >= 2].index)].copy()
    variants = [v for v in VARIANTS + ANGLES if v in m.columns]
    print(f"{args.cohort}: {len(m)} sessions from {m.Subject_ID.nunique()} "
          f"participants with repeats")
    print(f"{len(num)} repeatedly measured phenotypes, {len(variants)} variants\n")

    # Two alternative explanations have to be excluded before a within-person
    # association means anything.
    #
    #   same-day state  A participant who is unwell on a visit may both score
    #                   worse and move more, coupling the phenotype to data
    #                   quality rather than to the index.
    #   head pose       This paper's own variable. If the index moves with
    #                   posture and posture drifts with the person's condition,
    #                   the association is postural.
    extra = {}
    if args.cohort == "hcpa":
        mo = pd.read_csv(DIFF / "HCP" / "motion_rms_n1379.csv")
        mo = mo.rename(columns={"subject_id": "Subject_ID", "visit": "Visit"})
        mo["Subject_ID"] = mo.Subject_ID.astype(str)
        mo["Visit"] = mo.Visit.astype(str)
        m = m.merge(mo[["Subject_ID", "Visit", "motion_rms"]],
                    on=["Subject_ID", "Visit"], how="left")
        extra["motion"] = ["motion_rms"]
    # Registration quality, which anat_x inherits and the ratio does not.
    rq = HERE / f"registration_quality_{args.cohort}.csv"
    if rq.exists():
        q = pd.read_csv(rq)
        q["Subject_ID"] = q.Subject_ID.astype(str)
        q["Visit"] = q.Visit.astype(str)
        m = m.merge(q[["Subject_ID", "Visit", "det", "aniso", "shear"]],
                    on=["Subject_ID", "Visit"], how="left")
        extra["reg"] = ["det", "aniso", "shear"]

    # Region volume. In the cross-sectional age models this is the single
    # largest adjustment, taking the HCP-A classic coefficient from -0.448 to
    # -0.300, so leaving it out of a within-person model would hold this sweep
    # to a lower standard than the paper's own age associations. Unlike site or
    # sex it is not time-invariant, so within-person centering does not remove
    # it and it has to be carried explicitly.
    sph = (DIFF / "HCP" / "hcpa_alps_spheres_5mm.csv" if args.cohort == "hcpa"
           else DIFF / "DLBS" / "dlbs_alps_spheres_5mm.csv")
    if sph.exists():
        s = pd.read_csv(sph)
        s["Subject_ID"] = s.Subject_ID.astype(str)
        s["Visit"] = (s.Visit if "Visit" in s.columns else s.Session).astype(str)
        if {"n_proj", "n_assoc"} <= set(s.columns):
            s["nvox"] = (pd.to_numeric(s.n_proj, errors="coerce")
                         + pd.to_numeric(s.n_assoc, errors="coerce"))
            m = m.merge(s[["Subject_ID", "Visit", "nvox"]],
                        on=["Subject_ID", "Visit"], how="left")
            extra["roi"] = ["nvox"]
    cond = [c for c in ("fa_proj", "fa_assoc", "md_proj", "md_assoc")
            if c in m.columns]
    if cond:
        extra["cond"] = cond

    hr = pd.read_csv(HERE / f"head_rotation_{args.cohort}.csv")
    hr["Subject_ID"] = hr.Subject_ID.astype(str)
    hr["Visit"] = hr.Visit.astype(str)
    hr["abs_pitch"] = hr.pitch.abs()
    m = m.merge(hr[["Subject_ID", "Visit", "abs_pitch", "total"]],
                on=["Subject_ID", "Visit"], how="left")
    extra["pose"] = ["abs_pitch", "total"]

    cov_cols = sorted({c for v in extra.values() for c in v})
    # Is this the participant's own earliest session? Ordering by visit label is
    # enough here since both cohorts label visits chronologically.
    first = m.groupby("Subject_ID").Visit.transform("min")
    m["is_first"] = (m.Visit == first).astype(float)
    w = within(m, variants + num + ["Age"] + cov_cols + ["is_first"])

    # Each arm is (covariates, columns that must be present). The motion cache
    # covers fewer sessions than the imaging, so "age+motion" runs on about half
    # the data. Comparing it against the full-sample age arm confounds
    # confound-removal with power loss, and reads as motion explaining the
    # association when the correlations have barely moved. The matched arm
    # restricts to the motion subsample without adjusting for it, and is the
    # only fair comparator.
    arms = {"age": (["Age"], [])}
    for k, cols in extra.items():
        if k == "motion":
            arms["age|mot-sub"] = (["Age"], cols)
        arms[f"age+{k}"] = (["Age"] + cols, [])
    # Does anything survive once the ratio itself is partialled out? This is the
    # operational form of "is there information beyond radial anisotropy".
    if "ratio" in variants:
        arms["age+ratio"] = (["Age", "ratio"], [])
        # Practice effects are the main threat to any within-person cognitive
        # result. Scores rise on re-testing, and the rise is a step at first
        # exposure rather than a linear trend, so partialling age does not
        # remove it. A first-visit indicator absorbs the step. If an
        # association survives age, the ratio and this, it is not practice.
        arms["age+ratio+practice"] = (["Age", "ratio", "is_first"], [])
        # The full model. Everything that has been offered as an alternative
        # explanation, in one place: age drift, the radial anisotropy, practice
        # at first exposure, head pose, and registration quality.
        full = (["Age", "ratio", "is_first"] + extra.get("pose", [])
                + extra.get("reg", []) + extra.get("cond", [])
                + extra.get("roi", []))
        arms["everything"] = (full, [])
        # Motion enters last and on its own matched sample. The motion cache
        # covers about half the sessions, so the arm that adjusts for it and the
        # arm that does not have to be compared on the same participants, or the
        # sample does the work rather than the covariate.
        if extra.get("motion"):
            arms["everything|mot-sub"] = (full, extra["motion"])
            arms["everything+motion"] = (full + extra["motion"], [])
    print("covariate arms:", {k: v for k, v in arms.items()})
    print()

    rows = []
    for arm, (cols, require) in arms.items():
        ww = w.dropna(subset=require) if require else w
        for v in variants:
            # Partialling a variable out of itself is degenerate, and pv_perp is
            # the ratio to r=0.9999 so it is degenerate too.
            if v in cols or (arm == "age+ratio" and v == "pv_perp"):
                continue
            for c in num:
                if ww[c].notna().sum() < 40:
                    continue
                r, p, n = partial_corr(ww[v].to_numpy(float),
                                       ww[c].to_numpy(float),
                                       ww[cols].to_numpy(float))
                if not np.isnan(r):
                    rows.append({"arm": arm, "variant": v, "phenotype": c,
                                 "n": n, "r": r, "p": p})
    out = pd.DataFrame(rows)
    if out.empty:
        print("no testable phenotype-variant pairs")
        return
    # Assign per-variant BH through a transform. Concatenating groupby results
    # reorders the rows relative to the frame and silently pairs each q with the
    # wrong p, which showed up as p=0.475 carrying q=0.021.
    out["q"] = out.groupby(["arm", "variant"]).p.transform(
        lambda s: bh(s.to_numpy()))
    out = out.sort_values(["arm", "variant", "p"])
    # The re-sphered run must not land on the published filename. The verifier
    # reads that file, so overwriting it would move the manuscript's numbers
    # without anything recording that the placement rule had changed.
    _tag = args.cohort + ("_warpedmask" if os.environ.get("ALPS_WARPED_MASK")
                          else "")
    out.to_csv(HERE / f"phenotype_longitudinal_{_tag}.csv", index=False)

    print("=== within-participant, survivors at BH q<0.05 ===")
    print(f"{'variant':<12s} " + "  ".join(f"{a:>14s}" for a in arms))
    for v in variants:
        cells = []
        for a in arms:
            gg = out[(out.arm == a) & (out.variant == v)]
            cells.append(f"{int((gg.q < .05).sum())} of {len(gg)}")
        print(f"{v:<12s} " + "  ".join(f"{c:>14s}" for c in cells))

    print("\n   strongest association per variant, age arm:")
    for v in variants:
        gg = out[(out.arm == "age") & (out.variant == v)]
        if len(gg):
            t = gg.iloc[0]
            print(f"      {v:<12s} {t.phenotype[:36]:<36s} "
                  f"r={t.r:+.3f}  q={t.q:.2e}")
    print(f"\n   wrote phenotype_longitudinal_{_tag}.csv")


if __name__ == "__main__":
    main()
