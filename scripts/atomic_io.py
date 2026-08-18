"""Make result writing atomic, so a failed run cannot destroy a good result.

Importing this module is the whole interface. It replaces DataFrame.to_csv with
a version that writes to a temporary file beside the target and renames it into
place only after the write succeeds.

Why it exists. Running the analysis scripts under WSL truncated six committed
CSVs to one byte. The scripts address the data by drive letter, WSL exposes it
as /mnt/q instead, and pandas had already opened each output for writing before
the missing input was discovered. Opening for writing truncates. Every one of
those files was recoverable only because it was committed, which is luck rather
than design.

The failure class is broader than the wrong shell. Any interruption at all,
Ctrl-C, a full disk, an exception halfway through, a stale path, leaves a
truncated or partial CSV where a valid one used to be. Worse than losing it is
half-writing it, since a short file still parses and the verifier will happily
compare the manuscript against whatever survived.

os.replace is atomic within a filesystem, so a reader sees either the old file
or the new one and never a partial one.

    import atomic_io  # noqa: F401

Place it after the pandas import. Nothing else changes; every to_csv call in the
process becomes atomic, including calls made inside pandas itself.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

_original_to_csv = pd.DataFrame.to_csv


def _atomic_to_csv(self, path_or_buf=None, *args, **kwargs):
    # Only paths can be written atomically. None means "return a string", and a
    # file handle is the caller's to manage, so both pass straight through.
    if path_or_buf is None or not isinstance(path_or_buf, (str, os.PathLike)):
        return _original_to_csv(self, path_or_buf, *args, **kwargs)

    target = Path(path_or_buf)
    tmp = target.with_name(f".{target.name}.partial")
    try:
        result = _original_to_csv(self, tmp, *args, **kwargs)
        os.replace(tmp, target)          # atomic within one filesystem
        return result
    except BaseException:
        # Leave the existing target untouched and take the partial file with us.
        # BaseException so that Ctrl-C is covered too, which is one of the ways
        # a half-written result gets left behind.
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def install() -> None:
    """Idempotent, so importing from several modules in one process is safe."""
    if getattr(pd.DataFrame.to_csv, "_atomic", False):
        return
    _atomic_to_csv._atomic = True
    pd.DataFrame.to_csv = _atomic_to_csv


install()
