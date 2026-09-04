"""Do the measurement spheres fit inside the tracts they are meant to sample?

Re-running the contamination analysis on the redrawn spheres raised the fiber
share by about one and a half points in every cell, in both cohorts, for every
variant including the tract-locked ones. Those can only pick up lambda1 from
per-voxel scatter about the regional mean direction, so the voxels themselves
must have become more heterogeneous in orientation. The redrawn regions are the
smaller of the two, about a third fewer voxels, so it is not a size effect.

The obvious candidate is shape. The warp stretches the template sphere along
whatever direction the local deformation favours, which near a tract tends to
follow the tract. A true sphere is isotropic, so for the same volume it reaches
proportionally further across the fibre, toward the neighbouring tract.

That is measurable. The pipeline already has the JHU labels the direction
estimator uses, so this asks, per session and per region, what fraction of the
region's voxels lie inside the corresponding tract label, for both placements.
If the redrawn sphere sits further outside the label, the shape explanation
holds and the region could be constrained to the label rather than left free.

A small sample answers this. It is a geometric property of the placement, not
an association, so it does not need the cohort.

    python sphere_in_tract.py --cohort dlbs --limit 25
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_paths import winpath                                  # noqa: E402

HERE = Path(__file__).resolve().parent
DIFF = HERE.parent.parent / "diffusion"
OUT = Path(winpath("Q:/dti_output"))
FA_MIN, SLAB_MM = 0.2, 8.0
# projection pairs with SCR, association with SLF, per hemisphere
REGIONS = [("proj", 1, {"L": 26, "R": 25}), ("assoc", 2, {"L": 42, "R": 41})]


def main() -> None:
    import nibabel as nib

    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=["dlbs", "hcpa"], default="dlbs")
    ap.add_argument("--limit", type=int, default=25)
    ap.add_argument("--radius", type=float, default=5.0)
    args = ap.parse_args()
    shell = "_b1500" if args.cohort == "hcpa" else ""

    src = pd.read_csv(DIFF / ("HCP/hcpa_alps_spheres_5mm.csv" if args.cohort == "hcpa"
                              else "DLBS/dlbs_alps_spheres_5mm.csv"))
    src = src[src.status == "ok"].head(args.limit)

    rows = []
    for r in src.itertuples():
        sd = OUT / r.DTI_Session_ID / "processed"
        try:
            limg = nib.load(str(sd / "atlas" / "jhu_labels_registered.nii.gz"))
            lab = limg.get_fdata().astype(int)
            sph = nib.load(str(sd / "atlas" / "sphere_roi"
                               / "sphere_roi_combined.nii.gz")).get_fdata().astype(int)
            ev = nib.load(str(sd / f"tensor_eigenvalues{shell}.nii.gz")).get_fdata()
        except Exception:
            continue
        md = ev.mean(-1)
        nu = np.sqrt(((ev - md[..., None]) ** 2).sum(-1))
        de = np.sqrt((ev ** 2).sum(-1))
        fa = np.clip(np.sqrt(1.5) * np.divide(nu, de, out=np.zeros_like(nu),
                                              where=de != 0), 0, 1)
        ii, jj, kk = np.indices(lab.shape)
        A = limg.affine
        xw = A[0, 0] * ii + A[0, 1] * jj + A[0, 2] * kk + A[0, 3]
        yw = A[1, 0] * ii + A[1, 1] * jj + A[1, 2] * kk + A[1, 3]
        zw = A[2, 0] * ii + A[2, 1] * jj + A[2, 2] * kk + A[2, 3]

        for name, code, labels in REGIONS:
            for hemi, side in (("L", xw < 0), ("R", xw > 0)):
                warped = (sph == code) & side
                if not warped.any():
                    continue
                cx, cy, cz = xw[warped].mean(), yw[warped].mean(), zw[warped].mean()
                d2 = (xw - cx) ** 2 + (yw - cy) ** 2 + (zw - cz) ** 2
                drawn = d2 <= args.radius ** 2
                tract = lab == labels[hemi]
                # the band the direction estimator uses, for its width
                z0 = float(np.median(zw[sph > 0])) if (sph > 0).any() else 0.0
                band = tract & (np.abs(zw - z0) <= SLAB_MM) & (fa >= FA_MIN)

                # How much room is there actually? Measured as the distance
                # from the sphere centre to the far edge of the label along
                # each axis, because a tract can hold many voxels and still be
                # narrow across. A half-width below the radius means no sphere
                # of that radius fits, however large the label is in total.
                hw = {}
                if tract.any():
                    for ax, w in (("x", xw), ("y", yw), ("z", zw)):
                        off = w[tract] - {"x": cx, "y": cy, "z": cz}[ax]
                        hw[ax] = float(min(abs(off.min()), abs(off.max())))

                for tag, m in (("warped", warped & (fa >= FA_MIN)),
                               ("drawn", drawn & (fa >= FA_MIN))):
                    if not m.any():
                        continue
                    rows.append({
                        "session": r.DTI_Session_ID, "region": name, "hemi": hemi,
                        "placement": tag, "voxels": int(m.sum()),
                        "in_tract_pct": 100 * float((m & tract).sum() / m.sum()),
                        "band_voxels": int(band.sum()),
                        "halfwidth_x": hw.get("x"), "halfwidth_y": hw.get("y"),
                        "halfwidth_z": hw.get("z"),
                    })

    d = pd.DataFrame(rows)
    if d.empty:
        print("no sessions read")
        return
    d.to_csv(HERE / f"sphere_in_tract_{args.cohort}.csv", index=False)
    print(f"{d.session.nunique()} sessions, radius {args.radius} mm\n")
    print("fraction of each region's voxels lying inside its JHU tract label:")
    for name in ("proj", "assoc"):
        for tag in ("warped", "drawn"):
            s = d[(d.region == name) & (d.placement == tag)]
            if s.empty:
                continue
            print(f"   {name:<6s} {tag:<7s} {s.in_tract_pct.median():5.1f}%  "
                  f"(IQR {s.in_tract_pct.quantile(.25):.0f}-{s.in_tract_pct.quantile(.75):.0f})"
                  f"   {s.voxels.median():4.0f} voxels")
    print("\nhalf-width of the tract label about the sphere centre, mm:")
    for name in ("proj", "assoc"):
        s2 = d[(d.region == name) & (d.placement == "drawn")]
        if s2.empty:
            continue
        print(f"   {name:<6s} x {s2.halfwidth_x.median():4.1f}   "
              f"y {s2.halfwidth_y.median():4.1f}   z {s2.halfwidth_z.median():4.1f}"
              f"      (a 5 mm sphere needs 5.0 in every direction)")
    b = d.groupby("region").band_voxels.median()
    print("\ndirection-estimation band, same label restricted to a "
          f"{2 * SLAB_MM:.0f} mm axial slab:")
    for name in ("proj", "assoc"):
        if name in b:
            print(f"   {name:<6s} {b[name]:5.0f} voxels")
    print(f"\n   wrote sphere_in_tract_{args.cohort}.csv")


if __name__ == "__main__":
    main()
