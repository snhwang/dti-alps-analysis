"""Fetch the diffusion arm of OpenNeuro ds001907, which the current release lost.

ds001907 is "ANT: Healthy aging and Parkinson's disease". Its current release,
3.2.0, carries diffusion for eight control subjects and none for patients, so a
plain `aws s3 sync` or a fresh datalad clone returns a dataset that looks
useless for a patient-control diffusion comparison.

Version 3.0.2 carried the full set: 85 images over 44 subjects, 24 patients and
20 controls, most with two sessions. Note the direction, since the dataset's own
README states it backwards; see ds001907_common. Release 3.1.0 dropped 77. Its
changelog records only

    - Added file containing difference in days between scans
    - Fixed empty field in TSVs causing BIDS error

with no mention of removing imaging, and the deletions were committed by a
different person than the one who added the data. That reads as a bad sync
rather than a decision, so the 3.0.2 content is treated here as the intended
dataset. Nothing was actually deleted server-side: the snapshot API redirects
to a versioned S3 object that still answers 200.

Every annex key embeds the MD5 and the byte size, so each download is verified
against the key rather than trusted. A file that fails is deleted, not kept.

    python fetch_ds001907_dwi.py --dry-run
    python fetch_ds001907_dwi.py

Writes into M:/ds001907-download/ in BIDS layout and a manifest CSV beside it.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

TAG = "3.0.2"
REPO = "https://github.com/OpenNeuroDatasets/ds001907.git"
API = "https://openneuro.org/crn/datasets/ds001907/snapshots/{tag}/files/{key}"
DEST = Path(r"M:\ds001907-download")
CLONE = Path(r"C:\tmp\ds001907git")


def git(*a, cwd=CLONE):
    r = subprocess.run(["git", *a], cwd=cwd, capture_output=True, text=True)
    return r.stdout


def ensure_clone() -> None:
    """The git repo holds annex pointers only, so it is small and cheap."""
    if (CLONE / ".git").exists():
        return
    CLONE.parent.mkdir(parents=True, exist_ok=True)
    print(f"cloning pointers into {CLONE} ...")
    subprocess.run(["git", "clone", "--quiet", "--no-checkout", REPO, str(CLONE)],
                   check=True)


def listing(include: str = "/dwi/", exclude: str = "derivatives/"):
    """Every matching file at TAG, with the expected md5 and size where annexed."""
    out = []
    for f in git("ls-tree", "-r", "--name-only", TAG).split("\n"):
        f = f.strip()
        if include not in f or (exclude and f.startswith(exclude)):
            continue
        ptr = git("show", f"{TAG}:{f}")
        m = re.search(r"MD5E-s(\d+)--([0-9a-f]{32})", ptr)
        out.append((f, int(m.group(1)) if m else None,
                    m.group(2) if m else None, None if m else ptr))
    return out


def fetch(rel: str) -> bytes:
    url = API.format(tag=TAG, key=rel.replace("/", ":"))
    with urllib.request.urlopen(url, timeout=300) as r:
        return r.read()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    # The anatomicals are fetched the same verified way, as a control on the
    # pose measurement: a group difference that only appears in the FA
    # registration is a registration artefact, not a posture finding.
    ap.add_argument("--include", default="/dwi/",
                    help="path fragment to fetch, e.g. /anat/")
    args = ap.parse_args()

    ensure_clone()
    files = listing(args.include)
    annexed = [f for f in files if f[1] is not None]
    plain = [f for f in files if f[1] is None]
    total = sum(f[1] for f in annexed)
    print(f"version {TAG}: {len(files)} {args.include.strip('/')} files, "
          f"{len(annexed)} annexed "
          f"({total/1e9:.2f} GB), {len(plain)} small text files\n")
    if args.dry_run:
        for f, size, md5, _ in annexed[:5]:
            print(f"   {size/1e6:8.1f} MB  {f}")
        print("   ...")
        return

    DEST.mkdir(parents=True, exist_ok=True)
    rows, done, failed, skipped = [], 0, [], 0
    t0 = time.time()
    for i, (rel, size, md5, inline) in enumerate(files, 1):
        out = DEST / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists() and (size is None or out.stat().st_size == size):
            skipped += 1
            rows.append((rel, size or out.stat().st_size, "cached"))
            continue
        try:
            blob = inline.encode() if inline is not None else fetch(rel)
        except Exception as e:                                  # noqa: BLE001
            failed.append((rel, repr(e)[:70]))
            continue
        if md5 is not None:
            got = hashlib.md5(blob).hexdigest()
            if got != md5 or len(blob) != size:
                failed.append((rel, f"md5 {got[:8]} != {md5[:8]}"))
                continue
        out.write_bytes(blob)
        done += 1
        rows.append((rel, len(blob), "verified" if md5 else "text"))
        if size and size > 1e6:
            gb = sum(r[1] for r in rows) / 1e9
            print(f"   [{i:3d}/{len(files)}] {gb:5.2f} GB  {rel.split('/')[0]}",
                  flush=True)

    import csv
    name = f"manifest_{args.include.strip('/') or 'all'}.csv"
    with open(DEST / name, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "bytes", "status"])
        w.writerows(rows)

    print(f"\n   {done} downloaded, {skipped} already present, "
          f"{len(failed)} failed, in {(time.time()-t0)/60:.1f} min")
    for rel, why in failed[:10]:
        print(f"     FAIL {rel}  {why}")
    if failed:
        sys.exit(f"{len(failed)} file(s) failed verification")
    print(f"   manifest -> {DEST / name}")


if __name__ == "__main__":
    main()
