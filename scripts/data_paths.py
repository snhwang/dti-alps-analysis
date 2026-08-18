"""Resolve the data locations on either Windows or WSL.

The data sits on lettered drives: the processed sessions on Q:, the trigeminal
release on M:, the HCP structural packages on F: and P:. Windows addresses those
as "Q:/dti_output". WSL mounts the same volumes as "/mnt/q/dti_output". Scripts
that hardcode the first form do not merely fail under WSL, they open their
output for writing, truncate it, and only then discover the input is missing.
That is how six committed CSVs went to one byte.

atomic_io fixes the damage, this fixes the cause. One function, used wherever a
drive letter used to be written out:

    OUT = winpath("Q:/dti_output")

On Windows it hands back the path unchanged. On Linux it rewrites the drive
letter to the corresponding /mnt mount. Nothing is guessed: if the translated
path does not exist the caller sees a normal missing-path error, in the right
place, rather than an empty frame five layers down.

The point is that the analysis then runs in whichever environment is convenient,
including the WSL venv that already carries numpy, pandas, scipy, nibabel and
matplotlib, without a second interpreter having to be maintained in parallel.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path, PurePosixPath

_DRIVE = re.compile(r"^([A-Za-z]):[/\\](.*)$", re.S)

# WSL exposes drives under /mnt by default, but the prefix is configurable in
# wsl.conf, so allow it to be overridden rather than assuming.
MOUNT_ROOT = os.environ.get("WSL_DRIVE_ROOT", "/mnt")


def on_windows() -> bool:
    return sys.platform.startswith("win")


def winpath(p) -> Path:
    """A drive-lettered path, valid on this platform.

    Accepts and returns paths without a drive letter unchanged, so it is safe to
    wrap anything.
    """
    s = str(p)
    if on_windows():
        return Path(s)
    m = _DRIVE.match(s)
    if not m:
        return Path(s)
    drive, rest = m.group(1).lower(), m.group(2).replace("\\", "/")
    return Path(str(PurePosixPath(MOUNT_ROOT) / drive / rest))


def require(p, what: str = "data") -> Path:
    """winpath, but fail immediately and legibly if it is not there.

    Use at module import in scripts that would otherwise run far enough to
    truncate an output before noticing the input is absent.
    """
    q = winpath(p)
    if not q.exists():
        raise SystemExit(
            f"Cannot find {what} at {q}\n"
            f"  (asked for {p}; platform is {sys.platform})\n"
            f"  On WSL the drives appear under {MOUNT_ROOT}. Check the volume is "
            f"mounted, or set WSL_DRIVE_ROOT if your wsl.conf uses another prefix."
        )
    return q


def refined_repo() -> Path:
    """The dti-alps-refined checkout.

    Set DTI_ALPS_REFINED to your clone. Otherwise it is looked for beside the
    directory holding this repository, which is how it sits in the working tree.
    """
    env = os.environ.get("DTI_ALPS_REFINED")
    if env:
        return Path(winpath(env))
    return Path(__file__).resolve().parent.parent.parent / "dti-alps-refined"


def refined_rois() -> Path:
    """The four measurement regions in template space, shipped with the package."""
    return refined_repo() / "dti_alps_refined" / "rois"
