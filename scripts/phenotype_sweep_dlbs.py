"""The same phenotype sweep, in DLBS rather than HCP-A.

The HCP-A sweep tests every numeric variable in the AABC table against every
variant. This does the same for the Dallas Lifespan Brain Study, whose
phenotypes are different in kind: cognition (MMSE at three waves), education,
anthropometrics, and PET, both amyloid and tau. Amyloid and tau matter here
because the glymphatic literature is largely about clearance of exactly those
proteins, so they are the outcomes DTI-ALPS is most often claimed to speak to.

Age and sex are adjusted throughout, for the reason the HCP-A sweep gives: a
variant with a steeper age slope would otherwise inherit an advantage on every
age-related outcome without carrying any information about the outcome. Wave-1
age is used, since the diffusion cohort is anchored there.

Benjamini-Hochberg is applied across the sweep, separately per variant. That
control is within a variant and not across the five, so a hit appearing in one
variant alone is roughly five times likelier than the nominal rate, and the
output says so rather than leaving it to be inferred.

    python phenotype_sweep_dlbs.py

Writes phenotype_sweep_dlbs.csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import atomic_io  # noqa: F401  writes become atomic on import
from scipy import stats

HERE = Path(__file__).resolve().parent
DIFF = HERE.parent.parent / "diffusion"
VARIANTS = ["classic", "cross", "v2_sphere", "v2_slab", "pv_perp", "anat_x", "ld_alps"]
MIN_N = 60

# identifiers, dates, wave bookkeeping and age itself carry no phenotype
DROP = ("age", "id", "s#", "wave", "date", "visit", "mritotau", "mritoamy")


def fdr(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = np.empty_like(p)
    n = len(p)
    prev = 1.0
    for rank, i in enumerate(o[::-1], 1):
        prev = min(prev, p[i] * n / (n - rank + 1))
        q[i] = prev
    return q


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-n", type=int, default=MIN_N)
    args = ap.parse_args()

    d = pd.read_csv(HERE / "measured_pvs_axis_dlbs.csv")
    # LD-ALPS is computed by the authors' own implementation, so it joins here
    # rather than being produced alongside the tensor-derived variants.
    _ld = HERE / "ld_alps_dlbs.csv"
    if _ld.exists():
        L = pd.read_csv(_ld)[["Subject_ID", "Visit", "ALPS_overall"]].rename(
            columns={"ALPS_overall": "ld_alps"})
        for f in (d, L):
            f["Subject_ID"] = f.Subject_ID.astype(str)
            f["Visit"] = f.Visit.astype(str)
        d = d.merge(L, on=["Subject_ID", "Visit"], how="left")
    d = (d.sort_values(["Subject_ID", "Visit"]).groupby("Subject_ID").first().reset_index())
    d["participant_id"] = d.Subject_ID.astype(str)

    tabs = [pd.read_csv(DIFF / "DLBS" / "ds004856_participants.tsv", sep="\t",
                        low_memory=False)]
    for extra in ("dlbs_tau_w2.csv", "dlbs_tau_w3.csv"):
        f = DIFF / "DLBS" / extra
        if f.exists():
            t = pd.read_csv(f)
            if "participant_id" in t.columns:
                tabs.append(t.drop(columns=[c for c in t.columns
                                            if c in ("S#", "pet_wave")], errors="ignore"))
    a = tabs[0]
    for t in tabs[1:]:
        fresh = [c for c in t.columns if c == "participant_id" or c not in a.columns]
        a = a.merge(t[fresh], on="participant_id", how="outer")
    a["participant_id"] = a.participant_id.astype(str)

    m = d.merge(a, on="participant_id", how="inner")
    sex = next((c for c in m.columns if c.lower() in ("sex", "gender", "sex_bids")), None)
    m["sex_n"] = (pd.to_numeric(m[sex], errors="coerce") if sex is not None
                  else pd.Series(np.zeros(len(m)), index=m.index))
    if sex is not None and m.sex_n.isna().all():
        m["sex_n"] = m[sex].astype("category").cat.codes.astype(float)

    cand = [c for c in m.columns
            if pd.api.types.is_numeric_dtype(m[c])
            and not any(k in c.lower() for k in DROP)
            and c not in VARIANTS + ["sex_n", "Age", "pv_perp", "v2_slab", "v2_sphere",
                                     "v2_to_x", "v2_to_cross", "cross_to_x"]]
    print(f"{len(m)} participants merged; {len(cand)} candidate phenotypes")

    rows = []
    for c in cand:
        s = m[[c, "Age", "sex_n"] + VARIANTS].dropna()
        if len(s) < args.min_n or s[c].nunique() < 5:
            continue
        C = np.column_stack([np.ones(len(s)), s.Age, s.sex_n])

        def rz(v):
            b, *_ = np.linalg.lstsq(C, np.asarray(v, float), rcond=None)
            return np.asarray(v, float) - C @ b

        y = rz(s[c])
        rec = {"phenotype": c, "n": len(s)}
        for k in VARIANTS:
            r = float(np.corrcoef(y, rz(s[k]))[0, 1])
            t = r * np.sqrt((len(s) - 4) / max(1 - r * r, 1e-12))
            rec[k] = r
            rec[f"p_{k}"] = float(2 * stats.t.sf(abs(t), len(s) - 4))
        rows.append(rec)

    # Guard against the covariate silently doing nothing. If the sex column were
    # not found, or Age were missing, the design would collapse to an intercept
    # and every correlation here would be an unadjusted one wearing an adjusted
    # label. Assert that adjustment actually moves the numbers.
    if rows:
        _probe = max(rows, key=lambda r: abs(r.get("classic", 0.0)))
        _s = m[[_probe["phenotype"], "Age", "sex_n"] + VARIANTS].dropna()
        _raw = float(np.corrcoef(_s[_probe["phenotype"]], _s["classic"])[0, 1])
        if abs(_raw - _probe["classic"]) < 1e-6:
            raise SystemExit(
                f"adjustment had no effect on {_probe['phenotype']}: the covariates are "
                "not being applied. Check that Age and a sex column survived the merge.")
        print(f"  adjustment check: {_probe['phenotype']} classic "
              f"{_raw:+.4f} unadjusted -> {_probe['classic']:+.4f} adjusted")
        print(f"  sex coded for {int(m.sex_n.notna().sum())} participants, "
              f"{int(m.sex_n.sum())} male")
        print()

    out = pd.DataFrame(rows)
    if out.empty:
        print(f"no phenotype reached n >= {args.min_n}")
        return
    for k in VARIANTS:
        out[f"q_{k}"] = fdr(out[f"p_{k}"])
    out.to_csv(HERE / "phenotype_sweep_dlbs.csv", index=False)

    print(f"tested {len(out)} phenotypes with n >= {args.min_n}\n")
    for k in VARIANTS:
        hit = out[out[f"q_{k}"] < 0.05].sort_values(f"q_{k}")
        print(f"{k}: {len(hit)} phenotypes surviving FDR")
        for r in hit.itertuples():
            print(f"    {r.phenotype:38s} n={r.n:4d} r={getattr(r, k):+.3f}  "
                  f"q={getattr(r, 'q_' + k):.4f}")
    print("\nFDR is within variant, not across the five, so a hit in one variant")
    print("alone is about five times likelier than the nominal rate.")

    keep = [c for c in out.phenotype if any(t in c.lower()
            for t in ("mmse", "suvr", "amy", "tau", "meta"))]
    if keep:
        print("\ncognition and PET specifically:")
        for r in out[out.phenotype.isin(keep)].itertuples():
            cells = "  ".join(f"{k}={getattr(r, k):+.3f}" for k in VARIANTS)
            print(f"    {r.phenotype:34s} n={r.n:4d}  {cells}")


if __name__ == "__main__":
    main()
