"""Shared helpers for the MRI revision analyses."""

from __future__ import annotations

import pandas as pd


def parse_age(series: pd.Series) -> pd.Series:
    """
    Numeric age, handling the HCP-A top-coded category.

    HCP-A records the oldest participants as the string "90 or older" rather
    than a number. Passing the column straight to pd.to_numeric(errors="coerce")
    turns those into NaN, and a later dropna then removes the oldest
    participants without any warning, which both shrinks the sample and
    truncates the very end of the age range that a lifespan analysis depends on.
    In the AABC table this affects 138 visits from 89 participants.

    Top-coded values are mapped to 90, matching the convention used across the
    rest of this diffusion work. DLBS ages are already numeric, so this is a
    no-op there.
    """
    s = series.astype(str).str.strip()
    s = s.str.replace(r"^\s*90\s*(or|and)\s*older\s*$", "90", regex=True, case=False)
    s = s.str.replace(r"^\s*(90|89)\+\s*$", "90", regex=True)
    return pd.to_numeric(s, errors="coerce")
