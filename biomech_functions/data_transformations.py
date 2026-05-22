from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd


# --------------------------
# Utilities
# --------------------------
def first_non_null_map(df_pitch: pd.DataFrame, cols: Iterable[str]) -> dict:
    """Return {col: first_non_null(value)} for columns present in df_pitch."""
    out = {}
    for c in cols:
        if c in out:  # defensive
            continue
        if c in df_pitch.columns:
            out[c] = first_non_null(df_pitch[c])
    return out

def dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure unique column names (append _n for repeats)."""
    seen = {}
    new_cols = []
    for c in df.columns:
        if c not in seen:
            seen[c] = 1
            new_cols.append(c)
        else:
            k = seen[c] + 1
            seen[c] = k
            new_cols.append(f"{c}_{k}")
    df.columns = new_cols
    return df

# --------------------------
# Outcome / name logic (as provided)
# --------------------------
def first_non_null(series_like) -> object:
    """Return first non-null value from a Series-like; else NaN."""
    try:
        s = pd.Series(series_like).dropna()
        return s.iloc[0] if len(s) else np.nan
    except Exception:
        return np.nan

def split_timestamp(ts_val):
    """Split 'YYYY-MM-DD HH:MM:SS.sss' (or pandas.Timestamp) into (date_str, time_str)."""
    if pd.isna(ts_val):
        return np.nan, np.nan
    try:
        if isinstance(ts_val, pd.Timestamp):
            return ts_val.strftime("%Y-%m-%d"), ts_val.strftime("%H:%M:%S.%f")[:-3]
        ts_str = str(ts_val).strip()
        if " " in ts_str:
            d, t = ts_str.split(" ", 1)
            return d.strip(), t.strip()
        return ts_str, np.nan
    except Exception:
        return np.nan, np.nan

def parse_batter_name(gcs_path, jersey_num):
    """Parse player name from GCS_PATH using jersey number as anchor, ending at _Home/_Away."""
    if not isinstance(gcs_path, str) or gcs_path.strip() == "":
        return np.nan
    try:
        jersey_clean = str(int(jersey_num))
    except Exception:
        jersey_clean = str(jersey_num).strip() if jersey_num is not None else ""
    tokens = [t for t in re.split(r"[\\/._\-\s]+", gcs_path) if t]
    if not tokens:
        return np.nan
    lower = [t.lower() for t in tokens]
    try:
        end_idx = next(i for i, t in enumerate(lower) if t in ("home", "away"))
    except StopIteration:
        return np.nan
    last_numeric_idx, last_exact = None, None
    for i, tok in enumerate(tokens[:end_idx]):
        if tok.isdigit():
            last_numeric_idx = i
            if jersey_clean and tok == jersey_clean:
                last_exact = i
    anchor_idx = last_exact if last_exact is not None else last_numeric_idx
    if anchor_idx is None or anchor_idx + 1 >= end_idx:
        return np.nan
    start_idx = anchor_idx + 1
    for i in range(anchor_idx + 1, end_idx):
        if tokens[i].isdigit():
            start_idx = i + 1
    name_tokens = tokens[start_idx:end_idx]
    if not name_tokens:
        return np.nan
    name = " ".join(name_tokens)
    name = re.sub(r"(?i)(?<!\s)(Jr|Sr|II|III|IV)$", r" \1", name)
    name = re.sub(r"\s{2,}", " ", name).strip()
    return name if name else np.nan
