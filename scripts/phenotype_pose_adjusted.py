"""
Does adjusting for head position change the phenotype associations?

Two things could happen and both are worth knowing.

  Unmasking   If head position adds variance unrelated to the phenotype, removing
              it could reveal an association that was previously buried. That
              would be the most useful possible outcome of this whole project:
              a real biomarker relationship that positioning had been hiding.

  Deflation   If an apparent association ran partly through positioning, removing
              it should shrink the association, as it did for age in DLBS.

HCP-A is the harder case to expect anything from, because head pose there has no
relation to age and adjustment did nothing to the age coefficient. DLBS carries
amyloid and tau PET and real positioning, so it is the more plausible place for
an effect.

Reported with and without pose adjustment for every phenotype with adequate
coverage, FDR-corrected across the sweep, for classic and the corrected variants.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import atomic_io  # noqa: F401  writes become atomic on import
from scipy import stats

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
DIFF = HERE.parent.parent / "diffusion"
AABC = DIFF / "HCP" / "AABC2_subjects_2026_02_05_14_29_11.csv"
VARIANTS = ["classic", "cross", "v2_slab", "anat_x"]
MIN_N = 60
DROP = re.compile(r"age|days_from|yearquarter|^id|guid|pedid|_nda$|visit|wave|"
                  r"pctcompl|count$|^mr_|qc|scanner|site|^bulk|idps|msmall", re.I)


def bh(p):
    p = np.asarray(p, float); ok = ~np.isnan(p)
    q = np.full_like(p, np.nan); n = int(ok.sum())
    if n == 0:
        return q
    idx = np.argsort(p[ok])
    r = p[ok][idx] * n / (np.arange(n) + 1)
    r = np.minimum.accumulate(r[::-1])[::-1]
    o = np.empty(n); o[idx] = np.clip(r, 0, 1); q[ok] = o
    return q


def partial(y, x, C):
    def rz(v):
        b, *_ = np.linalg.lstsq(C, np.asarray(v, float), rcond=None)
        return np.asarray(v, float) - C @ b
    a, b = rz(x), rz(y)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return np.nan, np.nan
    r = float(np.corrcoef(a, b)[0, 1])
    dof = len(y) - C.shape[1]
    t = r * np.sqrt(dof / max(1 - r * r, 1e-12))
    return r, float(2 * (1 - stats.t.cdf(abs(t), dof)))


# ---------- HCP-A sweep ----------
d = pd.read_csv(HERE / "measured_pvs_axis_hcpa_b1500_all.csv")
h = pd.read_csv(HERE / "head_rotation_hcpa.csv")
for x in (d, h):
    x["Subject_ID"] = x.Subject_ID.astype(str); x["Visit"] = x.Visit.astype(str)
d = (d.merge(h, on=["Subject_ID", "Visit"])
       .sort_values(["Subject_ID", "Visit"]).groupby("Subject_ID").first().reset_index())

a = pd.read_csv(AABC, low_memory=False)
a["Subject_ID"] = a.id_event.astype(str).str.split("_").str[0]
num = [c for c in a.columns
       if pd.api.types.is_numeric_dtype(a[c]) and not DROP.search(c)
       and a[c].notna().sum() >= 100 and a[c].nunique() > 4]
ph = a.groupby("Subject_ID")[num].first().reset_index()
sx = a.groupby("Subject_ID")["sex"].first().reset_index()
m = d.merge(ph, on="Subject_ID").merge(sx, on="Subject_ID", how="left")
m["sex_n"] = (m.sex.astype(str).str.upper().str[0] == "M").astype(float)
print(f"HCP-A: {len(m)} participants with both head pose and phenotypes\n")

rows = []
for c in num:
    s = m[[c, "Age", "sex_n", "pitch", "total"] + VARIANTS].copy()
    s[c] = pd.to_numeric(s[c], errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < MIN_N or s[c].nunique() < 5:
        continue
    base = np.column_stack([np.ones(len(s)), s.Age, s.sex_n])
    pose = np.column_stack([np.ones(len(s)), s.Age, s.sex_n, s.pitch.abs(), s.total])
    rec = {"phenotype": c, "n": len(s)}
    for k in VARIANTS:
        r0, p0 = partial(s[c], s[k], base)
        r1, p1 = partial(s[c], s[k], pose)
        rec[f"{k}_raw"], rec[f"{k}_pose"] = r0, r1
        rec[f"p_{k}_pose"] = p1
    rows.append(rec)

out = pd.DataFrame(rows)
for k in VARIANTS:
    out[f"q_{k}"] = bh(out[f"p_{k}_pose"].values)
out.to_csv(HERE / "phenotype_pose_adjusted.csv", index=False)
print(f"tested {len(out)} phenotypes")
for k in VARIANTS:
    sig = out[out[f"q_{k}"] < 0.05]
    print(f"  {k:<10s} surviving FDR after pose adjustment: {len(sig)}")
    for r in sig.head(6).itertuples():
        print(f"      {getattr(r,'phenotype'):<38s} "
              f"{getattr(r, k+'_raw'):+.3f} -> {getattr(r, k+'_pose'):+.3f}")

shift = (out[[f"{k}_pose" for k in VARIANTS]].abs().to_numpy()
         - out[[f"{k}_raw" for k in VARIANTS]].abs().to_numpy())
print(f"\n  median change in |partial r| on adjusting for pose: {np.nanmedian(shift):+.4f}")
print(f"  phenotypes where |r| rose by more than 0.05: {int((shift > 0.05).sum())}")

# ---------- DLBS PET ----------
print("\nDLBS amyloid and tau PET")
dl = pd.read_csv(HERE / "measured_pvs_axis_dlbs.csv")
hd = pd.read_csv(HERE / "head_rotation_dlbs.csv")
for x in (dl, hd):
    x["Subject_ID"] = x.Subject_ID.astype(str); x["Visit"] = x.Visit.astype(str)
dl = (dl.merge(hd, on=["Subject_ID", "Visit"])
        .sort_values(["Subject_ID", "Visit"]).groupby("Subject_ID").first().reset_index())
amy = (pd.read_csv(DIFF / "DLBS" / "dlbs_amyloid_w1.csv")[["participant_id", "GlobalSUVR"]]
         .rename(columns={"participant_id": "Subject_ID", "GlobalSUVR": "amyloid"}))
tau = pd.concat([pd.read_csv(DIFF / "DLBS" / f"dlbs_tau_w{w}.csv")[
    ["participant_id", "TemporalMetaSUVR"]] for w in (2, 3)])
tau = (tau.rename(columns={"participant_id": "Subject_ID", "TemporalMetaSUVR": "tau"})
          .dropna().groupby("Subject_ID").first().reset_index())

for nm, pet in (("amyloid", amy), ("tau", tau)):
    col = pet.columns[1]
    s = dl.merge(pet, on="Subject_ID").dropna(subset=[col, "Age", "pitch"] + VARIANTS)
    if len(s) < 30:
        print(f"  {nm}: {len(s)} matched, skipped"); continue
    base = np.column_stack([np.ones(len(s)), s.Age])
    pose = np.column_stack([np.ones(len(s)), s.Age, s.pitch.abs(), s.total])
    print(f"  {nm} (n={len(s)})")
    for k in VARIANTS:
        r0, p0 = partial(s[col], s[k], base)
        r1, p1 = partial(s[col], s[k], pose)
        print(f"    {k:<10s} {r0:+.3f} (p={p0:.3f})  ->  {r1:+.3f} (p={p1:.3f})")
