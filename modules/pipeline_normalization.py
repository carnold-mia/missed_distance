from __future__ import annotations

import logging
import re
from typing import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CANONICAL_ID_COLUMNS = (
    "MLBAM_GUID",
    "MLBAM_GAME_ID",
    "MLBAM_PLAYER_ID",
    "SESSION_DATE",
)

INTERNAL_ID_COLUMNS = ("SESSION_ID", "PITCH_ID")
FRAME_GROUP_COLUMNS = ("MLBAM_GAME_ID", "MLBAM_GUID")

REQUIRED_GEOMETRY_COLUMNS = (
    "CENTER_TX",
    "CENTER_TY",
    "CENTER_TZ",
    "TOP_TX",
    "TOP_TY",
    "TOP_TZ",
    "KNOB_TX",
    "KNOB_TY",
    "KNOB_TZ",
    "LEFTFOOT_TX",
    "RIGHTFOOT_TX",
)

BALL_CENTER_COLUMNS = ("CENTER_TX", "CENTER_TY", "CENTER_TZ")

def normalize_mlbam_hitting_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize enriched batting motion rows for the missed-distance engine.

    The returned frame keeps Kinatrax SESSION_ID/PITCH_ID as internal keys while
    validating MLBAM_GUID and MLBAM_GAME_ID as the canonical output identity.
    """
    normalized = _standardize_column_names(df)
    _require_columns(normalized, CANONICAL_ID_COLUMNS, label="canonical MLBAM IDs")
    _require_columns(normalized, INTERNAL_ID_COLUMNS, label="internal Kinatrax IDs")
    _require_columns(normalized, REQUIRED_GEOMETRY_COLUMNS, label="geometry")

    normalized = normalized.copy()
    _null_absent_ball_centers(normalized)

    sort_cols = [
        col
        for col in ("MLBAM_GAME_ID", "MLBAM_GUID", "SESSION_ID", "PITCH_ID", "TIMESTAMP")
        if col in normalized.columns
    ]
    if sort_cols:
        normalized = normalized.sort_values(sort_cols).reset_index(drop=True)

    if "FRAME" not in normalized.columns:
        normalized["FRAME"] = (
            normalized.groupby(list(FRAME_GROUP_COLUMNS), dropna=False)
            .cumcount()
            .add(1)
            .astype(int)
        )
    else:
        normalized["FRAME"] = pd.to_numeric(normalized["FRAME"], errors="coerce")
        if normalized["FRAME"].isna().any():
            raise ValueError("FRAME contains non-numeric values after normalization.")
        normalized["FRAME"] = normalized["FRAME"].astype(int)
        _ensure_one_based_frame(normalized)

    _warn_null_identifiers(normalized, ("MLBAM_GUID", "MLBAM_GAME_ID"))
    return normalized


def _null_absent_ball_centers(df: pd.DataFrame) -> None:
    """Treat all-zero ball center coordinates as missing ball observations."""
    center_cols = list(BALL_CENTER_COLUMNS)
    coords = df.loc[:, center_cols].apply(pd.to_numeric, errors="coerce")
    absent_ball = coords.notna().all(axis=1) & np.isclose(coords, 0.0).all(axis=1)
    if not absent_ball.any():
        return

    df.loc[absent_ball, center_cols] = np.nan
    logger.info(
        "Set %s absent ball frames to null where CENTER_TX/CENTER_TY/CENTER_TZ were all zero.",
        f"{int(absent_ball.sum()):,}",
    )


def _ensure_one_based_frame(df: pd.DataFrame) -> None:
    """Shift per-pitch FRAME labels from 0-based to 1-based when needed."""
    group_cols = [col for col in FRAME_GROUP_COLUMNS if col in df.columns]
    if not group_cols:
        min_frame = df["FRAME"].min()
        if pd.notna(min_frame) and int(min_frame) == 0:
            df["FRAME"] = df["FRAME"] + 1
            logger.info("Shifted FRAME labels from 0-based to 1-based.")
        return

    min_frame_by_group = df.groupby(group_cols, dropna=False)["FRAME"].transform("min")
    zero_based_rows = min_frame_by_group.eq(0)
    if not zero_based_rows.any():
        return

    groups_shifted = df.loc[zero_based_rows, group_cols].drop_duplicates().shape[0]
    df.loc[zero_based_rows, "FRAME"] = df.loc[zero_based_rows, "FRAME"] + 1
    logger.info(
        "Shifted FRAME labels from 0-based to 1-based for %s pitch groups.",
        f"{groups_shifted:,}",
    )


def _standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized.columns = [_normalize_column_name(col) for col in normalized.columns]
    return normalized


def _normalize_column_name(column: object) -> str:
    text = str(column).strip()
    text = re.sub(r"[\s\-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.upper()


def _require_columns(
    df: pd.DataFrame,
    columns: Iterable[str],
    *,
    label: str,
) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required {label} columns: {missing}")


def _warn_null_identifiers(df: pd.DataFrame, columns: Iterable[str]) -> None:
    """Log a warning if any canonical identifier columns contain nulls.

    Rows are kept in the dataset — downstream steps that require a
    non-null GUID will naturally skip them during groupby/join operations.
    """
    cols = list(columns)
    affected = {c: int(df[c].isna().sum()) for c in cols if df[c].isna().any()}
    if affected:
        logger.warning("Null canonical identifiers (rows kept): %s", affected)
