"""Does the acquisition actually satisfy the alignment the method assumes?

Evaluating diffusivities along fixed scanner axes is defensible only if those
axes already sit close to the anatomical ones. Taoka et al. state that the
transverse plane is usually taken at the anterior commissure to posterior
commissure line and that this is the standard for evaluation in the ALPS method.
That is an assumption the method rests on rather than a property of any
particular dataset, and it is checkable, because slice prescription is written
into the header before the first volume is collected.

The test. If an operator were prescribing on the AC-PC line, the slab would be
angulated to compensate for however the head was lying, so the prescribed pitch
would track the head pitch with a slope near +1 and the residual would be
smaller than the head pitch alone. If the slab is simply laid down near axial,
the slope is near zero and the residual is whatever the head was doing.

Both quantities are rotations about the same scanner x axis in the same frame:
slab_pitch is read from the prescription, aff_pitch is the head pitch recovered
from the subject-to-template affine.

    python acpc_assumption.py

Writes acpc_assumption.csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import atomic_io  # noqa: F401  writes become atomic on import
from scipy import stats

HERE = Path(__file__).resolve().parent


def main() -> None:
    argparse.ArgumentParser().parse_args()
    d = pd.read_csv(HERE / "slab_prescription_dlbs.csv").dropna()
    one = (d.sort_values(["Subject_ID", "Visit"])
             .groupby("Subject_ID").first().reset_index())

    rows = []
    for label, f in (("per_session", d), ("per_participant", one)):
        r, p = stats.pearsonr(f.slab_pitch, f.aff_pitch)
        slope = float(np.polyfit(f.aff_pitch, f.slab_pitch, 1)[0])
        resid = (f.aff_pitch - f.slab_pitch).abs().median()
        rows.append(dict(basis=label, n=len(f),
                         head_pitch=float(f.aff_pitch.abs().median()),
                         slab_pitch=float(f.slab_pitch.abs().median()),
                         slab_tilt=float(f.slab_tilt.abs().median()),
                         near_zero_pct=float((f.slab_pitch.abs() < 1.0).mean() * 100),
                         r=float(r), p=float(p), slope=slope,
                         residual=float(resid)))

    # Is the prescription reproducible within a participant? An alignment set
    # against a landmark should be. One set by eye should not.
    rep = d.groupby("Subject_ID").filter(lambda g: len(g) >= 2)

    def icc(df, col):
        g = df.groupby("Subject_ID")[col]
        k = g.size().mean()
        msb = g.mean().var(ddof=1) * k
        msw = g.apply(lambda s: s.var(ddof=1)).mean()
        return float((msb - msw) / (msb + (k - 1) * msw))

    change = (rep.sort_values(["Subject_ID", "Visit"]).groupby("Subject_ID")
              .slab_pitch.apply(lambda s: s.diff().abs().dropna()))
    repro = dict(basis="repeat_visits", n=len(rep),
                 n_participants=int(rep.Subject_ID.nunique()),
                 icc_slab=icc(rep, "slab_pitch"),
                 icc_head=icc(rep, "aff_pitch"),
                 median_change=float(change.median()),
                 slab_pitch=float(rep.slab_pitch.abs().median()))
    rows.append(repro)

    out = pd.DataFrame(rows)
    out.to_csv(HERE / "acpc_assumption.csv", index=False)

    for row in out[out.basis != "repeat_visits"].itertuples():
        print(f"\nDLBS, {row.basis.replace('_', ' ')}, n = {row.n}\n")
        print(f"   head pitch, from the affine        median |{row.head_pitch:5.2f}| deg")
        print(f"   prescribed pitch, from the header  median |{row.slab_pitch:5.2f}| deg")
        print(f"   prescribed slabs within 1 deg of axial   {row.near_zero_pct:5.1f}%")
        print(f"   prescribed vs head pitch           r = {row.r:+.3f}  "
              f"p = {row.p:.2g}")
        print(f"   compensation slope                 {row.slope:+.3f}  "
              f"(+1 would be full, 0 none)")
        print(f"   residual after prescription        median |{row.residual:5.2f}| deg")
        print(f"   head pitch alone                   median |{row.head_pitch:5.2f}| deg")

    print(f"\nReproducibility, {repro['n_participants']} participants with repeat "
          f"visits, {repro['n']} sessions\n")
    print(f"   prescribed pitch                   ICC {repro['icc_slab']:+.3f}")
    print(f"   head pitch                         ICC {repro['icc_head']:+.3f}")
    print(f"   median between-visit change        {repro['median_change']:5.2f} deg")
    print(f"   median prescribed pitch itself     {repro['slab_pitch']:5.2f} deg")

    print("\n   The prescription does not follow the head. The slope is negative rather")
    print("   than near +1, so where a tilt is applied it goes the wrong way on")
    print("   average, and the residual exceeds the head pitch it was meant to remove.")
    print("   It is also not reproducible: the change between visits exceeds the")
    print("   quantity itself, and it is no steadier than the head position it would")
    print("   have to track.")
    print("\n   This is a research study with a fixed protocol, where the alignment")
    print("   should if anything be held to a higher standard than routine clinical")
    print("   acquisition. It is not satisfied here.")


if __name__ == "__main__":
    main()
