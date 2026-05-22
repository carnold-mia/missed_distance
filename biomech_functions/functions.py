"""Missed-distance discrete outputs for Kinatrax hitting motion data.

The active pipeline is hitting-only and uses MLBAM game/GUID identifiers when
provided by the caller. Legacy SESSION_ID/PITCH_ID values may still be used as
internal grouping keys, but the MLBAM path drops them from final outputs.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.signal import butter, filtfilt

from .data_transformations import (
    dedupe_columns,
    first_non_null,
    first_non_null_map,
    parse_batter_name,
    split_timestamp,
)
from .schema import OUTPUT_SCHEMA

__all__ = [
    "SWEET_SPOTS",
    "construct_bat_80_lcs",
    "compute_discrete_and_time_series",
]


# --------------------------
# Config / constants
# --------------------------
FPS = 300.0
DT = 1.0 / FPS

SWEET_SPOTS = {"K80": 0.80}
AXES = ("X", "Y", "Z")
SPACES_LOCAL_GLOBAL = ("LOCAL", "GLOBAL")
SPACES_GLOBAL_LOCAL = ("GLOBAL", "LOCAL")
BAND_LO_FRACTION = 0.79
BAND_HI_FRACTION = 0.85

LOWPASS_CUTOFF_HZ = 30.0
LOWPASS_ORDER = 4
R80_UNIT_VECTOR_DECIMALS = 6
R80_ORTHOGONAL_DECIMALS = 6
R80_STABLE_SPEED_THRESHOLD = 2.0
R80_STABLE_MIN_FRAMES = 5
VALIDATION_PLOT_ROOT = Path("fig_outputs/MLBAM_GAME_GUID_MD_VALIDATION")

DEFAULT_GROUP_ID_COLS = ("SESSION_ID", "PITCH_ID")
LEGACY_INTERNAL_ID_COLS = ("SESSION_ID", "PITCH_ID")
CANONICAL_MLBAM_ID_COLS = (
    "MLBAM_GAME_ID",
    "MLBAM_GUID",
    "MLBAM_PLAYER_ID",
    "SESSION_DATE",
)

_REQUIRED_GEOMETRY = {
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
}

META_DESIRED_METRICS = [
    "MLBAM_GUID",
    "MLBAM_PLAYER_ID",
    "MLBAM_GAME_ID",
    "SESSION_DATE",
    "SESSION_ID",
    "SESSION_TIMESTAMP",
    "PITCH_ID",
    "PITCH_TIMESTAMP",
    "PLAYER_JERSEY_NUMBER",
    "TEAM_NAME",
    "GCS_PATH",
    "CREATED_AT",
    "ITEM",
    "BAD",
    "R",
    "TAKE",
    "SWING",
    "MISS",
    "BALL_CONTACT",
    "CHECK_SWING",
    "BALL_START",
    "BAT_START",
    "BALL_MIN",
    "BALL_MIN_DIST_MISS",
    "DOWNSWING",
    "DS",
    "BAT_STOP",
    "LOAD",
    "SETUP",
    "START_DATA",
    "BALL_PITCH_VELOCITY",
    "HANDEDNESS",
    "HEIGHT",
    "KT_DATA_TYPE",
    "MASS",
    "MAX_BAT_SPEED_MPH",
]

EVENT_COLS = ("DOWNSWING", "DS", "BALL_START", "BAT_START", "BALL_MIN", "BAT_STOP")
EVENT_COLORS = {
    "DOWNSWING": "tab:purple",
    "DS": "tab:purple",
    "BALL_START": "tab:green",
    "BAT_START": "tab:blue",
    "BALL_MIN": "tab:pink",
    "BAT_STOP": "tab:gray",
    "BatBall_Min": "black",
}

FOOT_ANKLE_COLS = (
    "LEFTFOOT_TX", "LEFTFOOT_TY", "LEFTFOOT_TZ",
    "RIGHTFOOT_TX", "RIGHTFOOT_TY", "RIGHTFOOT_TZ",
    "LEFTANKLE_TX", "LEFTANKLE_TY", "LEFTANKLE_TZ",
    "RIGHTANKLE_TX", "RIGHTANKLE_TY", "RIGHTANKLE_TZ",
)
FOOT_ANKLE_START_SOURCES = {
    "LEFTFOOT_AT_START_DATA_X": "LEFTFOOT_TX",
    "LEFTFOOT_AT_START_DATA_Y": "LEFTFOOT_TY",
    "LEFTFOOT_AT_START_DATA_Z": "LEFTFOOT_TZ",
    "RIGHTFOOT_AT_START_DATA_X": "RIGHTFOOT_TX",
    "RIGHTFOOT_AT_START_DATA_Y": "RIGHTFOOT_TY",
    "RIGHTFOOT_AT_START_DATA_Z": "RIGHTFOOT_TZ",
    "LEFTANKLE_AT_START_DATA_X": "LEFTANKLE_TX",
    "LEFTANKLE_AT_START_DATA_Y": "LEFTANKLE_TY",
    "LEFTANKLE_AT_START_DATA_Z": "LEFTANKLE_TZ",
    "RIGHTANKLE_AT_START_DATA_X": "RIGHTANKLE_TX",
    "RIGHTANKLE_AT_START_DATA_Y": "RIGHTANKLE_TY",
    "RIGHTANKLE_AT_START_DATA_Z": "RIGHTANKLE_TZ",
}


# --------------------------
# Validation / identifiers
# --------------------------
def _validate_schema(
    df: pd.DataFrame,
    group_id_cols: Sequence[str] = DEFAULT_GROUP_ID_COLS,
) -> None:
    required = set(_REQUIRED_GEOMETRY).union(group_id_cols)
    missing = sorted(column for column in required if column not in df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def _as_tuple(values: object) -> tuple:
    if isinstance(values, tuple):
        return values
    return (values,)


def _group_key_map(group_id_cols: Sequence[str], group_values: object) -> dict[str, object]:
    return dict(zip(group_id_cols, _as_tuple(group_values)))


def _safe_identifier_label(group_key: dict[str, object]) -> str:
    parts = []
    for value in group_key.values():
        text = "unknown" if pd.isna(value) else str(value)
        clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
        parts.append(clean or "unknown")
    return "_".join(parts)


def _first_id_values(
    df_pitch: pd.DataFrame,
    output_id_cols: Sequence[str],
    group_key: dict[str, object],
) -> dict[str, object]:
    values: dict[str, object] = {}
    for col in output_id_cols:
        if col in group_key:
            values[col] = group_key[col]
        elif col in df_pitch.columns:
            values[col] = first_non_null(df_pitch[col])
    return values


def _plot_label(output_ids: dict[str, object], group_key: dict[str, object]) -> str:
    canonical = {
        col: output_ids[col]
        for col in ("MLBAM_GAME_ID", "MLBAM_GUID")
        if col in output_ids
    }
    return _safe_identifier_label(canonical or output_ids or group_key)


# --------------------------
# Vector helpers
# --------------------------
def _to_vec3(df: pd.DataFrame, px: str, py: str, pz: str) -> np.ndarray:
    """Return a 3 x N float array from X/Y/Z columns."""
    return df[[px, py, pz]].to_numpy(float).T


def _sweet_spot_positions(knob: np.ndarray, top: np.ndarray) -> dict[str, np.ndarray]:
    """Return K80 sweet-spot and contact-band loci along knob-to-top."""
    bat = top - knob
    return {
        "ss_k80": knob + SWEET_SPOTS["K80"] * bat,
        "ss_top": knob + BAND_HI_FRACTION * bat,
        "ss_bottom": knob + BAND_LO_FRACTION * bat,
    }


def _fixed_lowpass_filter(
    data: np.ndarray,
    *,
    fs_hz: float = FPS,
    cutoff_hz: float = LOWPASS_CUTOFF_HZ,
    order: int = LOWPASS_ORDER,
) -> np.ndarray:
    """Apply the fixed KT-style zero-phase 30 Hz Butterworth filter."""
    arr = np.asarray(data, dtype=float)
    if arr.size == 0:
        return arr.copy()

    if cutoff_hz <= 0.0:
        raise ValueError(f"cutoff_hz must be positive; got {cutoff_hz}")
    if fs_hz <= 0.0:
        raise ValueError(f"fs_hz must be positive; got {fs_hz}")

    b, a = butter(order, cutoff_hz, btype="low", analog=False, fs=fs_hz)
    padlen = 3 * (max(len(a), len(b)) - 1)
    n_samples = arr.shape[0]
    if n_samples <= padlen:
        return arr.copy()
    return filtfilt(b, a, arr, axis=0)


def _vector_derivative(vector_xyz: np.ndarray) -> np.ndarray:
    if vector_xyz.shape[1] <= 1:
        return np.zeros_like(vector_xyz, dtype=float)
    return np.gradient(vector_xyz, DT, axis=1)


def _filtered_vector_velocity(vector_xyz: np.ndarray) -> np.ndarray:
    """Differentiate a 3 x N vector series, filter XYZ velocity, return 3 x N."""
    finite_frames = np.isfinite(vector_xyz).all(axis=0)
    if finite_frames.all():
        velocity = _vector_derivative(vector_xyz).T
        return _fixed_lowpass_filter(velocity).T

    filtered = np.full_like(vector_xyz, np.nan, dtype=float)
    for start_idx, end_idx in _valid_frame_runs(finite_frames):
        segment = vector_xyz[:, start_idx:end_idx]
        velocity = _vector_derivative(segment).T
        filtered[:, start_idx:end_idx] = _fixed_lowpass_filter(velocity).T
    return filtered


def _valid_frame_runs(valid: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open frame ranges for contiguous valid observations."""
    runs: list[tuple[int, int]] = []
    start_idx: int | None = None
    for idx, is_valid in enumerate(valid):
        if is_valid and start_idx is None:
            start_idx = idx
        elif not is_valid and start_idx is not None:
            runs.append((start_idx, idx))
            start_idx = None
    if start_idx is not None:
        runs.append((start_idx, len(valid)))
    return runs


def _speed_from_velocity(velocity_xyz: np.ndarray) -> np.ndarray:
    return np.linalg.norm(velocity_xyz, axis=0)


def _local_selection_distance(miss_vector_local: np.ndarray) -> np.ndarray:
    """Return full 3D local miss-vector norm used to select T_MIN_LOCAL."""
    return np.linalg.norm(miss_vector_local, axis=0)


def _normalize_vectors(vectors: np.ndarray, *, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = np.zeros_like(vectors, dtype=float)
    valid = norms.squeeze(axis=1) > eps
    normalized[valid] = vectors[valid] / norms[valid]
    return normalized


def _fallback_axis_candidates(y_hat: np.ndarray) -> np.ndarray:
    """Build deterministic perpendicular axis candidates when bat motion is degenerate."""
    references = np.eye(3)
    candidates = np.zeros_like(y_hat, dtype=float)
    for idx, y_axis in enumerate(y_hat):
        ref = references[np.argmin(np.abs(references @ y_axis))]
        candidates[idx] = np.cross(y_axis, ref)
    return candidates


def _backfill_axis_candidates(
    candidates: np.ndarray,
    *,
    anchor_indices: Sequence[int | None] | None = None,
    eps: float = 1e-8,
) -> np.ndarray:
    """Fill degenerate axis candidates from nearby valid frames."""
    filled = candidates.copy()
    norms = np.linalg.norm(filled, axis=1)
    valid_indices = np.flatnonzero(norms > eps)
    if valid_indices.size == 0:
        return filled

    for anchor_value in anchor_indices or ():
        if anchor_value is None:
            continue
        anchor = int(np.clip(anchor_value, 0, len(filled) - 1))
        if norms[anchor] > eps:
            filled[:anchor] = filled[anchor]
            last_valid = anchor
            for idx in range(anchor + 1, len(filled)):
                if norms[idx] > eps:
                    last_valid = idx
                else:
                    filled[idx] = filled[last_valid]
            return filled

    first_valid = valid_indices[0]
    filled[:first_valid] = filled[first_valid]

    last_valid = first_valid
    for idx in range(first_valid + 1, len(filled)):
        if norms[idx] > eps:
            last_valid = idx
        else:
            filled[idx] = filled[last_valid]
    return filled


def _interpolate_xyz_series(points: np.ndarray) -> np.ndarray:
    """Fill small marker gaps before filtering orientation inputs."""
    filled = np.asarray(points, dtype=float).copy()
    for axis_idx in range(filled.shape[1]):
        series = pd.Series(filled[:, axis_idx], dtype=float)
        if series.notna().any():
            filled[:, axis_idx] = (
                series.interpolate(limit_direction="both")
                .ffill()
                .bfill()
                .to_numpy(dtype=float)
            )
    return filled


def _smooth_orientation_points(points: np.ndarray) -> np.ndarray:
    """Smooth XYZ points used to derive R_80 orientation."""
    filled = _interpolate_xyz_series(points)
    if not np.isfinite(filled).all():
        return points.copy()
    return _fixed_lowpass_filter(filled)


def _centered_motion_vectors(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return centered BAT_80 motion vectors and speed."""
    motion = np.zeros_like(points, dtype=float)
    if points.shape[0] <= 1:
        return motion, np.zeros(points.shape[0], dtype=float)

    motion = np.gradient(points, axis=0)
    speed = np.linalg.norm(motion, axis=1) * FPS
    return motion, speed


def _enforce_axis_continuity(
    x_hat: np.ndarray,
    z_hat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Flip transverse axes when they reverse direction frame-to-frame."""
    x_continuous = x_hat.copy()
    z_continuous = z_hat.copy()
    for idx in range(1, len(x_continuous)):
        if not np.isfinite(x_continuous[idx]).all() or not np.isfinite(x_continuous[idx - 1]).all():
            continue
        if float(np.dot(x_continuous[idx], x_continuous[idx - 1])) < 0.0:
            x_continuous[idx] *= -1.0
            z_continuous[idx] *= -1.0
    return x_continuous, z_continuous


def _first_stable_speed_index(
    speed: np.ndarray,
    *,
    threshold: float = R80_STABLE_SPEED_THRESHOLD,
    min_frames: int = R80_STABLE_MIN_FRAMES,
) -> int | None:
    """Return the first index where speed stays above threshold for a short run."""
    finite_speed = np.asarray(speed, dtype=float)
    above_threshold = np.isfinite(finite_speed) & (finite_speed >= threshold)
    if above_threshold.size == 0:
        return None

    run_length = max(1, int(min_frames))
    if above_threshold.size < run_length:
        return int(np.flatnonzero(above_threshold)[0]) if above_threshold.any() else None

    for idx in range(above_threshold.size - run_length + 1):
        if bool(above_threshold[idx : idx + run_length].all()):
            return idx
    return None


def _stabilize_preswing_axes(
    x_hat: np.ndarray,
    y_hat: np.ndarray,
    z_hat: np.ndarray,
    stable_idx: int | None,
    *,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Hold BAT_80 transverse roll before the first stable swing frame.

    Local Y remains the real bat direction on every frame. X is the first stable
    swing-frame X axis projected into each pre-swing bat-normal plane, and Z
    completes the frame.
    """
    if stable_idx is None or stable_idx <= 0 or stable_idx >= len(x_hat):
        return x_hat, z_hat

    x_ref = x_hat[stable_idx]
    if not np.isfinite(x_ref).all() or np.linalg.norm(x_ref) <= eps:
        return x_hat, z_hat

    x_stable = x_hat.copy()
    z_stable = z_hat.copy()
    for idx in range(stable_idx):
        y_axis = y_hat[idx]
        if not np.isfinite(y_axis).all() or np.linalg.norm(y_axis) <= eps:
            continue

        x_projected = x_ref - float(np.dot(x_ref, y_axis)) * y_axis
        x_norm = np.linalg.norm(x_projected)
        if x_norm <= eps:
            continue

        x_stable[idx] = x_projected / x_norm
        z_stable[idx] = _normalize_vectors(np.cross(x_stable[idx : idx + 1], y_hat[idx : idx + 1]))[0]

    return x_stable, z_stable


def _lock_axis_signs(
    x_hat: np.ndarray,
    y_hat: np.ndarray,
    z_hat: np.ndarray,
    stable_idx: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lock X/Z directions, then flip Y only to preserve a right-handed LCS."""
    del stable_idx

    raw_forward = np.array([1.0, 0.0, 0.0])
    raw_down = np.array([0.0, 1.0, 0.0])
    x_locked = _lock_axis_to_reference(x_hat, raw_forward)
    z_locked = _lock_axis_to_reference(z_hat, raw_down)
    y_locked = y_hat.copy()

    rotation = np.stack([x_locked, y_locked, z_locked], axis=2)
    det = np.linalg.det(rotation)
    y_locked[np.isfinite(det) & (det < 0.0)] *= -1.0
    return x_locked, y_locked, z_locked


def _lock_axis_to_reference(axis: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Flip each frame of one axis so it points toward a reference direction."""
    locked = axis.copy()
    dots = locked @ reference
    locked[np.isfinite(dots) & (dots < 0.0)] *= -1.0
    return locked


def _project_reference_onto_bat_normal_plane(
    reference: np.ndarray,
    y_hat: np.ndarray,
) -> np.ndarray:
    """Project a field reference into each frame's plane perpendicular to Y."""
    ref = np.asarray(reference, dtype=float)
    return ref - np.sum(y_hat * ref, axis=1, keepdims=True) * y_hat


def _is_left_handed_hitter(handedness: object) -> bool:
    """Return True for left-handed hitter labels."""
    if handedness is None or pd.isna(handedness):
        return False
    label = str(handedness).strip().upper()
    return label in {"L", "LH", "LHH", "LEFT", "LEFTY", "LEFT-HANDED"}


def construct_bat_80_lcs(
    top: np.ndarray,
    knob: np.ndarray,
    handedness: object = None,
    axis_backfill_indices: Sequence[int | None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct the BAT_80 local coordinate system.

    Parameters
    ----------
    top, knob:
        Arrays of shape ``(n_frames, 3)`` in raw/source Kinatrax coordinates.
    handedness:
        Optional hitter side from the batting report. Right-handed hitters use
        the base configuration. Left-handed hitters flip Z before X is derived.
    axis_backfill_indices:
        Optional ordered frame indices used to backfill early Z-axis candidates.
        The pipeline passes BALL_MIN first and BAT_STOP second when available.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``bat_80`` has shape ``(n_frames, 3)``. ``r_bat80`` has shape
        ``(n_frames, 3, 3)`` with columns ``[x_hat, y_hat, z_hat]``.

    Notes
    -----
    Basic BAT_80 motion frame:
    ``Y = normalize(TOP - BAT_80)``, ``temp_i = BAT_80_(i-1) - BAT_80_i``,
    ``Z = normalize(cross(Y, temp))``, and ``X = normalize(cross(Y, Z))``.
    """
    top_xyz = np.asarray(top, dtype=float)
    knob_xyz = np.asarray(knob, dtype=float)

    # --- Input validation -------------------------------------------------
    if top_xyz.ndim != 2 or top_xyz.shape[1] != 3:
        raise ValueError("top must have shape (n_frames, 3).")
    if knob_xyz.shape != top_xyz.shape:
        raise ValueError("knob must have the same shape as top.")

    # --- BAT_80 origin and Y axis (along the shaft, toward TOP) -----------
    bat_80 = knob_xyz + SWEET_SPOTS["K80"] * (top_xyz - knob_xyz)
    y_hat = _normalize_vectors(top_xyz - bat_80)

    # --- Backward-difference frame velocity -------------------------------
    # First frame mirrors the second so n=1 inputs still produce a valid LCS.
    bat_temp = np.zeros_like(bat_80)
    if len(bat_80) > 1:
        bat_temp[1:] = bat_80[:-1] - bat_80[1:]
        bat_temp[0] = bat_temp[1]

    # --- Z axis from -(Y x dBAT), backfilled, with fallback for degenerate frames
    z_candidates = _backfill_axis_candidates(
        -np.cross(y_hat, bat_temp),
        anchor_indices=axis_backfill_indices,
    )
    z_candidates = _project_reference_onto_bat_normal_plane(z_candidates, y_hat)
    degenerate = np.linalg.norm(z_candidates, axis=1) <= 1e-8
    if degenerate.any():
        z_candidates[degenerate] = _fallback_axis_candidates(y_hat[degenerate])
    z_hat = _normalize_vectors(z_candidates)

    # --- X axis and handedness --------------------------------------------
    # X = -normalize(Y x Z) is invariant to the handedness flip on Z because
    # cross(y, -z) = -cross(y, z), which cancels the explicit RHH sign flip
    # in the original two-branch implementation. Computing X first lets us
    # express handedness as a single conditional on Z only.
    x_hat = -_normalize_vectors(np.cross(y_hat, z_hat))
    if _is_left_handed_hitter(handedness):
        z_hat = -z_hat

    r_bat80 = np.stack([x_hat, y_hat, z_hat], axis=2)
    return bat_80, r_bat80


def _bat_80_lcs(
    knob: np.ndarray,
    top: np.ndarray,
    handedness: object = None,
    axis_backfill_indices: Sequence[int | None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build BAT_80-origin local axes for internal ``(3, n_frames)`` inputs."""
    bat_80, r_bat80 = construct_bat_80_lcs(
        top.T,
        knob.T,
        handedness=handedness,
        axis_backfill_indices=axis_backfill_indices,
    )
    return bat_80.T, r_bat80


def _r80_unit_vector_counts(rotation: np.ndarray) -> tuple[int, int, float, float]:
    """Count R_80 axis vectors whose norms round to 1."""
    norms = np.linalg.norm(rotation, axis=1)
    finite_norms = norms[np.isfinite(norms)]
    if finite_norms.size == 0:
        return 0, int(norms.size), np.nan, np.nan

    rounded_to_unit = np.round(finite_norms, R80_UNIT_VECTOR_DECIMALS) == 1.0
    return (
        int(rounded_to_unit.sum()),
        int(norms.size),
        float(finite_norms.min()),
        float(finite_norms.max()),
    )


def _r80_orthogonal_pair_counts(rotation: np.ndarray) -> tuple[int, int, float]:
    """Count R_80 axis pairs whose dot products round to 0."""
    x_hat = rotation[:, :, 0]
    y_hat = rotation[:, :, 1]
    z_hat = rotation[:, :, 2]
    dots = np.column_stack(
        [
            np.einsum("ij,ij->i", x_hat, y_hat),
            np.einsum("ij,ij->i", y_hat, z_hat),
            np.einsum("ij,ij->i", x_hat, z_hat),
        ]
    )
    finite_dots = dots[np.isfinite(dots)]
    if finite_dots.size == 0:
        return 0, int(dots.size), np.nan

    rounded_to_orthogonal = (
        np.round(finite_dots, R80_ORTHOGONAL_DECIMALS) == 0.0
    )
    return (
        int(rounded_to_orthogonal.sum()),
        int(dots.size),
        float(np.max(np.abs(finite_dots))),
    )


def _log_r80_unit_vector_check(
    confirmed: int,
    total: int,
    norm_min: float,
    norm_max: float,
) -> None:
    """Log whether the internal R_80 matrix axes are unit vectors."""
    if total == 0:
        logger.warning("R_80 matrix unit-vector check: no unit vectors were evaluated.")
        return

    log_fn = logger.info if confirmed == total else logger.warning
    log_fn(
        "%s/%s unit vectors confirmed for R_80 matrix "
        "(axis norms round to 1 at %s decimals; min=%.6g, max=%.6g).",
        f"{confirmed:,}",
        f"{total:,}",
        R80_UNIT_VECTOR_DECIMALS,
        norm_min,
        norm_max,
    )


def _log_r80_orthogonality_check(
    confirmed: int,
    total: int,
    max_abs_dot: float,
) -> None:
    """Log whether the internal R_80 matrix axes are mutually orthogonal."""
    if total == 0:
        logger.warning("R_80 matrix orthogonality check: no axis pairs were evaluated.")
        return

    log_fn = logger.info if confirmed == total else logger.warning
    log_fn(
        "%s/%s axis pairs confirmed orthogonal for R_80 matrix "
        "(dot products round to 0 at %s decimals; max_abs_dot=%.6g).",
        f"{confirmed:,}",
        f"{total:,}",
        R80_ORTHOGONAL_DECIMALS,
        max_abs_dot,
    )


def _global_to_bat_local(
    points_xyz: np.ndarray,
    origin_xyz: np.ndarray,
    r_bat: np.ndarray,
) -> np.ndarray:
    """Transform global points into a bat-local coordinate frame."""
    points = points_xyz.T
    origin = origin_xyz.T
    r_bat_t = np.transpose(r_bat, (0, 2, 1))
    return np.einsum("fij,fj->fi", r_bat_t, points - origin).T


def _minimum_index(distance: np.ndarray) -> int | None:
    """Return the frame index of the minimum distance, or None if all values are NaN."""
    if distance.size == 0:
        return None
    if np.all(np.isnan(distance)):
        return None
    return int(np.nanargmin(distance))


def _minimum_index_in_window(
    distance: np.ndarray,
    start_idx: int,
    end_idx: int,
) -> int | None:
    """Return the minimum finite index inside an inclusive frame-index window.

    If the window contains no valid frames, returns None. The pipeline does
    NOT fall back to the full frame range, because the BAT_80 LCS construction
    depends on BALL_MIN / BAT_STOP as backfill anchors and the argmin window
    requires DOWNSWING / BAT_STOP. Without those events, the local frame is
    not stably defined and the t-min has no biomechanical meaning. NaN
    outputs in that case are intentional and correct.
    """
    if distance.size == 0:
        return None

    lo, hi = sorted((int(start_idx), int(end_idx)))
    lo = max(lo, 0)
    hi = min(hi, distance.size - 1)
    if lo > hi:
        return None

    window = distance[lo:hi + 1]
    if window.size == 0 or np.all(np.isnan(window)):
        return None
    return int(lo + np.nanargmin(window))


# --------------------------
# Plotting / legacy helper API
# --------------------------
def _plot_md_validation(
    distances: np.ndarray,
    t_min: int,
    *,
    frame: np.ndarray | None = None,
    pitch_id: str | int | None = None,
    out_dir: str | Path = VALIDATION_PLOT_ROOT,
    ball_z: np.ndarray | None = None,
    target_z: np.ndarray | None = None,
    events: dict[str, float] | None = None,
    ball_start_frame: float | None = None,
) -> None:
    distances = np.asarray(distances)
    n_frames = distances.shape[0]
    x = np.asarray(frame) if frame is not None else np.arange(n_frames)
    if x.shape[0] != n_frames:
        raise ValueError(f"frame length {x.shape[0]} does not match distances length {n_frames}")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    label = "unknown_mlbam_id" if pitch_id is None else str(pitch_id)
    fname = out_path / f"{label}_md_validation.png"

    plt.figure(figsize=(8, 4))
    if ball_z is not None and target_z is not None:
        ball_z = np.asarray(ball_z)
        target_z = np.asarray(target_z)
        if ball_z.shape[0] != n_frames or target_z.shape[0] != n_frames:
            raise ValueError("ball_z/target_z length must match distances length")
        plt.plot(x, target_z, label="Bat Sweet Spot Z")
        plt.plot(x, ball_z, label="Ball Z")
        plt.plot(x, target_z - ball_z, label="Bat Z - Ball Z")
        plt.plot(x, np.zeros_like(x), label="Zero Line")
    else:
        plt.plot(x, distances, label="Miss Distance (m)")

    if events is not None:
        for event_name in EVENT_COLS:
            if event_name in events:
                plt.axvline(
                    x=events[event_name],
                    linestyle="--",
                    linewidth=1,
                    color=EVENT_COLORS.get(event_name, "k"),
                    label=event_name,
                )

    plt.axvline(
        x=x[t_min],
        color=EVENT_COLORS["BatBall_Min"],
        linestyle="-.",
        linewidth=1.5,
        label=f"t_min (d={float(distances[t_min]):.3f} m)",
    )

    if ball_start_frame is not None:
        plt.xlim(left=max(ball_start_frame - 100, float(np.min(x))))

    plt.xlabel("Frame")
    plt.ylabel("Position / Distance (m)")
    plt.title(f"Miss Distance Validation - ID={label}")
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize=8)
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


def _plot_md_validation_3d(
    *,
    ball_x: np.ndarray,
    ball_y: np.ndarray,
    ball_z: np.ndarray,
    target_x: np.ndarray,
    target_y: np.ndarray,
    target_z: np.ndarray,
    t_min: int,
    pitch_id: str | int | None = None,
    out_dir: str | Path = VALIDATION_PLOT_ROOT,
    events: dict[str, float] | None = None,
) -> None:
    ball_x = np.asarray(ball_x, dtype=float)
    ball_y = np.asarray(ball_y, dtype=float)
    ball_z = np.asarray(ball_z, dtype=float)
    target_x = np.asarray(target_x, dtype=float)
    target_y = np.asarray(target_y, dtype=float)
    target_z = np.asarray(target_z, dtype=float)

    n_frames = ball_x.shape[0]
    if any(arr.shape[0] != n_frames for arr in [ball_y, ball_z, target_x, target_y, target_z]):
        raise ValueError("All coordinate arrays must have the same length.")
    if not 0 <= t_min < n_frames:
        raise ValueError(f"t_min={t_min} is out of bounds for length {n_frames}.")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    label = "unknown_mlbam_id" if pitch_id is None else str(pitch_id)
    fname = out_path / f"{label}_md_validation_3d.html"

    event_traces = []
    if events is not None:
        for event_name, frame_idx in events.items():
            try:
                idx = int(frame_idx)
            except (TypeError, ValueError):
                continue
            if 0 <= idx < n_frames:
                event_traces.append(
                    go.Scatter3d(
                        x=[ball_x[idx]],
                        y=[ball_y[idx]],
                        z=[ball_z[idx]],
                        mode="markers",
                        name=event_name,
                        marker=dict(size=4, symbol="diamond"),
                    )
                )

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=ball_x,
                y=ball_y,
                z=ball_z,
                mode="lines",
                name="Ball",
                line=dict(width=4),
            ),
            go.Scatter3d(
                x=target_x,
                y=target_y,
                z=target_z,
                mode="lines",
                name="Bat Sweet Spot",
                line=dict(width=4),
            ),
            go.Scatter3d(
                x=[ball_x[t_min], target_x[t_min]],
                y=[ball_y[t_min], target_y[t_min]],
                z=[ball_z[t_min], target_z[t_min]],
                mode="markers+lines",
                name="Closest global miss",
                marker=dict(size=6),
                line=dict(width=2, dash="dash"),
            ),
            *event_traces,
        ]
    )
    fig.update_layout(
        title=f"3D Miss Distance Validation - ID={label}",
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    fig.write_html(str(fname), include_plotlyjs="cdn")


def _closest_distance_and_vector(
    ball: np.ndarray,
    target: np.ndarray,
    *,
    pitch_id: str | int | None = None,
    frame: np.ndarray | None = None,
    save_plot: bool = False,
    out_dir: str | Path = VALIDATION_PLOT_ROOT,
    events: dict[str, float] | None = None,
    ball_start_frame: float | None = None,
) -> tuple[int, float, np.ndarray]:
    """Return index, global XYZ Euclidean distance, and BALL-minus-target vector."""
    if ball.shape != target.shape:
        raise ValueError(f"Ball and target shapes do not match: {ball.shape} vs {target.shape}")
    if ball.shape[0] != 3:
        raise ValueError(f"Expected 3 x N arrays but got {ball.shape}")

    miss_all = ball - target
    distances = np.linalg.norm(miss_all, axis=0)
    t_min = _minimum_index(distances)

    if save_plot:
        _plot_md_validation(
            distances,
            t_min,
            frame=frame,
            pitch_id=pitch_id,
            out_dir=out_dir,
            ball_z=ball[2, :],
            target_z=target[2, :],
            events=events,
            ball_start_frame=ball_start_frame,
        )
        _plot_md_validation_3d(
            ball_x=ball[0, :],
            ball_y=ball[1, :],
            ball_z=ball[2, :],
            target_x=target[0, :],
            target_y=target[1, :],
            target_z=target[2, :],
            t_min=t_min,
            pitch_id=pitch_id,
            out_dir=out_dir,
            events=events,
        )

    return t_min, float(distances[t_min]), miss_all[:, t_min].copy()


def _axial_band_check(
    knob_t: np.ndarray,
    ball_t: np.ndarray,
    ss_top_t: np.ndarray,
    ss_bottom_t: np.ndarray,
) -> bool:
    axis = ss_top_t - knob_t
    norm = np.linalg.norm(axis)
    if norm == 0.0:
        return False
    unit_axis = axis / norm
    projected = float(np.dot(ball_t - knob_t, unit_axis))
    top_radius = np.linalg.norm(ss_top_t - knob_t)
    bottom_radius = np.linalg.norm(ss_bottom_t - knob_t)
    lo, hi = sorted((bottom_radius, top_radius))
    eps = 1e-9
    return lo - eps <= projected <= hi + eps


# --------------------------
# Outcome logic
# --------------------------
def _norm_upper(df_pitch: pd.DataFrame, col: str) -> pd.Series:
    if col not in df_pitch.columns:
        return pd.Series([], dtype=str)
    return df_pitch[col].astype(str).str.strip().str.upper()


def _flag_is_set(df_pitch: pd.DataFrame, col: str) -> bool:
    """Check whether a boolean flag column is set for this pitch.

    Flag columns in BATTING_REPORTS store the column name as the value
    when true (e.g. SWING column = 'SWING') and NaN when false.
    """
    if col not in df_pitch.columns:
        return False
    vals = df_pitch[col].dropna()
    return not vals.empty


def _infer_handedness_from_report(df_pitch: pd.DataFrame) -> str | float:
    """Infer batter side from explicit report R/L flags or foot positions."""
    if _flag_is_set(df_pitch, "R"):
        return "R"
    if _flag_is_set(df_pitch, "L"):
        return "L"
    if {"LEFTFOOT_TX", "RIGHTFOOT_TX"}.issubset(df_pitch.columns):
        left_back_foot = df_pitch["LEFTFOOT_TX"].iloc[0]
        right_back_foot = df_pitch["RIGHTFOOT_TX"].iloc[0]
        if right_back_foot < 0:
            return "R"
        if left_back_foot > 0:
            return "L"
    return np.nan


def _infer_outcome_with_take_skip(df_pitch: pd.DataFrame) -> tuple[str, bool]:
    """Classify a pitch as hit, miss, check_swing, or unknown.

    Flag columns (TAKE, SWING, BALL_CONTACT, MISS, CHECK_SWING) are
    independent boolean indicators — a non-NaN value means the flag is set.
    """
    if _flag_is_set(df_pitch, "TAKE"):
        return "TAKE", True

    has_contact = _flag_is_set(df_pitch, "BALL_CONTACT")
    has_check_swing = _flag_is_set(df_pitch, "CHECK_SWING")
    has_swing = _flag_is_set(df_pitch, "SWING")
    has_miss = _flag_is_set(df_pitch, "MISS")

    if has_contact:
        return "hit", False
    if has_check_swing:
        return "CHECK_SWING", False
    if has_miss or (has_swing and not has_contact):
        return "miss", False
    return np.nan, False


def _fill_derived_report_tags(meta_pitch: dict[str, object], outcome: object) -> None:
    """Fill report-style tags when the raw report stores the state implicitly."""
    if outcome == "miss" and pd.isna(meta_pitch.get("MISS", np.nan)):
        meta_pitch["MISS"] = "MISS"


def _frame_vector(df_pitch: pd.DataFrame, n_frames: int) -> np.ndarray:
    if "FRAME" in df_pitch.columns:
        return df_pitch["FRAME"].to_numpy()
    return np.arange(n_frames, dtype=int)


def _event_times(df_pitch: pd.DataFrame) -> dict[str, float]:
    events: dict[str, float] = {}
    for event_name in EVENT_COLS:
        if event_name in df_pitch.columns:
            value = first_non_null(df_pitch[event_name])
            if pd.notna(value):
                events[event_name] = value
    return events


def _frame_index_for_value(frame_vec: np.ndarray, value: object) -> int | None:
    """Return the frame index matching an event value, or None if unavailable."""
    if pd.isna(value):
        return None
    frames = pd.to_numeric(pd.Series(frame_vec), errors="coerce").to_numpy(float)
    if frames.size == 0:
        return None
    try:
        event_value = float(value)
    except (TypeError, ValueError):
        return None

    exact = np.where(np.isclose(frames, event_value, equal_nan=False))[0]
    if exact.size:
        return int(exact[0])

    positional_frame = int(round(event_value))
    if 1 <= positional_frame <= frames.size:
        return positional_frame - 1
    if 0 <= positional_frame < frames.size:
        return positional_frame
    return None


def _window_start_index(
    df_pitch: pd.DataFrame,
    frame_vec: np.ndarray,
) -> int:
    """Use DOWNSWING/DS when present, otherwise START_DATA, otherwise frame 0."""
    for col in ("DOWNSWING", "DS", "START_DATA"):
        if col not in df_pitch.columns:
            continue
        idx = _frame_index_for_value(frame_vec, first_non_null(df_pitch[col]))
        if idx is not None:
            return idx
    return 0


def _window_end_index(
    df_pitch: pd.DataFrame,
    frame_vec: np.ndarray,
) -> int:
    """Use BAT_STOP when present, otherwise the final frame."""
    if "BAT_STOP" in df_pitch.columns:
        idx = _frame_index_for_value(frame_vec, first_non_null(df_pitch["BAT_STOP"]))
        if idx is not None:
            return idx
    return max(len(frame_vec) - 1, 0)


def _axis_backfill_indices(df_pitch: pd.DataFrame, frame_vec: np.ndarray) -> list[int | None]:
    """Return ordered R80 backfill anchors: BALL_MIN first, then BAT_STOP."""
    indices: list[int | None] = []
    for col in ("BALL_MIN", "BAT_STOP"):
        if col not in df_pitch.columns:
            continue
        indices.append(_frame_index_for_value(frame_vec, first_non_null(df_pitch[col])))
    return indices


def _add_start_data_positions(
    row: dict[str, object],
    df_pitch: pd.DataFrame,
    frame_vec: np.ndarray,
) -> None:
    idx = (
        _frame_index_for_value(frame_vec, first_non_null(df_pitch["START_DATA"]))
        if "START_DATA" in df_pitch.columns
        else None
    )
    for out_col, source_col in FOOT_ANKLE_START_SOURCES.items():
        if idx is None or source_col not in df_pitch.columns:
            row[out_col] = np.nan
        else:
            row[out_col] = float(df_pitch[source_col].iloc[idx])


def _signed_peak_components(vector_xyz: np.ndarray, start_idx: int, end_idx: int) -> dict[str, float]:
    lo, hi = sorted((start_idx, end_idx))
    window = vector_xyz[:, lo:hi + 1]
    peaks: dict[str, float] = {}
    for axis_idx, axis in enumerate(AXES):
        values = window[axis_idx]
        finite = np.isfinite(values)
        if not finite.any():
            peaks[axis] = np.nan
            continue
        finite_values = values[finite]
        peaks[axis] = float(finite_values[np.nanargmax(np.abs(finite_values))])
    return peaks


def _max_speed(speed: np.ndarray, start_idx: int, end_idx: int) -> float:
    lo, hi = sorted((start_idx, end_idx))
    values = speed[lo:hi + 1]
    finite = values[np.isfinite(values)]
    return float(np.nanmax(finite)) if finite.size else np.nan


def _add_window_max_metrics(
    row: dict[str, object],
    *,
    spot: str,
    space: str,
    start_idx: int,
    end_idx: int | None,
    velocity: np.ndarray,
    speed: np.ndarray,
) -> None:
    for axis in AXES:
        row[f"MAX_MISS_VELOCITY_{spot}_{space}_{axis}"] = np.nan
    row[f"MAX_MISS_SPEED_{spot}_{space}"] = np.nan

    if end_idx is None:
        return

    for axis, value in _signed_peak_components(velocity, start_idx, end_idx).items():
        row[f"MAX_MISS_VELOCITY_{spot}_{space}_{axis}"] = value
    row[f"MAX_MISS_SPEED_{spot}_{space}"] = _max_speed(
        speed,
        start_idx,
        end_idx,
    )


def _add_local_miss_direction_flags(
    row: dict[str, object],
    miss_vector_local: np.ndarray,
    idx: int | None,
) -> None:
    """Classify the K80 local miss vector at T_MIN_LOCAL."""
    for col in ("CAPPED", "JAMMED", "OVER", "UNDER"):
        row[col] = np.nan

    if idx is None:
        return

    y_value = float(miss_vector_local[1, idx])
    z_value = float(miss_vector_local[2, idx])
    if not np.isfinite(y_value) or not np.isfinite(z_value):
        return

    row["CAPPED"] = 1 if y_value > 0.0 else 0
    row["JAMMED"] = 1 if y_value < 0.0 else 0
    row["OVER"] = 1 if z_value < 0.0 else 0
    row["UNDER"] = 1 if z_value > 0.0 else 0


def _axis_values(vector_xyz: np.ndarray, idx: int) -> dict[str, float]:
    return {
        "X": float(vector_xyz[0, idx]),
        "Y": float(vector_xyz[1, idx]),
        "Z": float(vector_xyz[2, idx]),
    }


def _metric_keys_for(spot: str, space: str) -> list[str]:
    return OUTPUT_SCHEMA.metric_keys(spot, space)


def _add_metrics_for_space(
    row: dict[str, object],
    *,
    spot: str,
    space: str,
    idx: int | None,
    frame_value: object,
    distance: np.ndarray,
    miss_vector: np.ndarray,
    velocity: np.ndarray,
    speed: np.ndarray,
    ball_position: np.ndarray,
    bat_knob_position: np.ndarray,
    bat_top_position: np.ndarray,
    sweet_spot_position: np.ndarray,
) -> None:
    for key in _metric_keys_for(spot, space):
        row[key] = np.nan

    if idx is None:
        return

    row[f"T_MIN_{space}_{spot}"] = frame_value
    row[f"MISSED_DISTANCE_{space}_{spot}"] = float(distance[idx])

    for axis, value in _axis_values(miss_vector, idx).items():
        row[f"MISS_VECTOR_{space}_{spot}_{axis}"] = value
    for axis, value in _axis_values(velocity, idx).items():
        row[f"MISS_VECTOR_VELOCITY_{space}_{spot}_{axis}"] = value

    row[f"MISSED_DISTANCE_{space}_{spot}_SPEED_AT_TMIN"] = float(speed[idx])

    if space == "GLOBAL":
        ball_prefix = f"BALL_AT_TMIN_{spot}"
        knob_prefix = f"BAT_KNOB_AT_TMIN_{spot}"
        top_prefix = f"BAT_TOP_AT_TMIN_{spot}"
    else:
        ball_prefix = f"BALL_IN_BAT_AT_TMIN_{spot}"
        top_prefix = f"BAT_TOP_IN_BAT_AT_TMIN_{spot}"

    for axis, value in _axis_values(ball_position, idx).items():
        row[f"{ball_prefix}_{axis}"] = value
    for axis, value in _axis_values(bat_top_position, idx).items():
        row[f"{top_prefix}_{axis}"] = value
    if space == "GLOBAL":
        for axis, value in _axis_values(bat_knob_position, idx).items():
            row[f"{knob_prefix}_{axis}"] = value
        ss_prefix = f"SS_{spot}_AT_TMIN"
        for axis, value in _axis_values(sweet_spot_position, idx).items():
            row[f"{ss_prefix}_{axis}"] = value


def _add_time_series_spot_columns(
    ts_data: dict[str, object],
    *,
    spot: str,
    ss_global: np.ndarray,
    miss_global: np.ndarray,
    miss_local: np.ndarray,
    distance_global: np.ndarray,
    distance_local: np.ndarray,
    velocity_global: np.ndarray,
    velocity_local: np.ndarray,
    speed_global: np.ndarray,
    speed_local: np.ndarray,
) -> None:
    for axis_idx, axis in enumerate(("X", "Y", "Z")):
        ts_data[f"SS_{spot}_{axis}"] = ss_global[axis_idx]
        ts_data[f"MISS_VECTOR_GLOBAL_{spot}_{axis}"] = miss_global[axis_idx]
        ts_data[f"MISS_VECTOR_LOCAL_{spot}_{axis}"] = miss_local[axis_idx]
        ts_data[f"MISS_VECTOR_VELOCITY_GLOBAL_{spot}_{axis}"] = velocity_global[axis_idx]
        ts_data[f"MISS_VECTOR_VELOCITY_LOCAL_{spot}_{axis}"] = velocity_local[axis_idx]

    ts_data[f"MISSED_DISTANCE_GLOBAL_{spot}"] = distance_global
    ts_data[f"MISSED_DISTANCE_LOCAL_{spot}"] = distance_local
    ts_data[f"MISSED_DISTANCE_GLOBAL_{spot}_SPEED"] = speed_global
    ts_data[f"MISSED_DISTANCE_LOCAL_{spot}_SPEED"] = speed_local


def _existing(columns: pd.Index, desired: Sequence[str]) -> list[str]:
    return [column for column in desired if column in columns]


def _time_series_column_order(
    df: pd.DataFrame,
    group_id_cols: Sequence[str],
) -> list[str]:
    desired = OUTPUT_SCHEMA.time_series_columns(group_id_cols, FOOT_ANKLE_COLS)
    ordered = _existing(df.columns, desired)
    ordered += [column for column in df.columns if column not in ordered]
    return ordered


def _reorder_metrics_columns(
    df: pd.DataFrame,
    output_id_cols: Sequence[str],
) -> pd.DataFrame:
    """Reorder metrics columns into the analysis-facing schema."""
    desired = OUTPUT_SCHEMA.discrete_front_columns(FOOT_ANKLE_START_SOURCES)
    front = _existing(df.columns, desired)
    remaining = [column for column in df.columns if column not in front]
    return df[front + remaining]


# --------------------------
# Public API
# --------------------------
def compute_discrete_and_time_series(
    df: pd.DataFrame,
    *,
    group_id_cols: Sequence[str] = DEFAULT_GROUP_ID_COLS,
    output_id_cols: Sequence[str] | None = None,
    save_validation_plots: bool = True,
    validation_plot_format: str = "both",
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if validation_plot_format not in {"none", "png", "html", "both"}:
        raise ValueError(
            "validation_plot_format must be one of: none, png, html, both"
        )

    group_id_cols = tuple(group_id_cols)
    output_id_cols = tuple(output_id_cols or group_id_cols)
    _validate_schema(df, group_id_cols)

    sort_cols = [col for col in (*group_id_cols, "FRAME") if col in df.columns]
    df_work = df.sort_values(sort_cols).copy() if sort_cols else df.copy()

    metrics_rows: list[dict[str, object]] = []
    ts_chunks: list[pd.DataFrame] = []
    processed_count = 0
    outcome_counts = {
        "hit": 0,
        "miss": 0,
        "CHECK_SWING": 0,
        "no_report": 0,
        "TAKE_skipped": 0,
        "empty_frames_skipped": 0,
    }
    r80_unit_vectors_confirmed = 0
    r80_unit_vectors_total = 0
    r80_norm_min = np.inf
    r80_norm_max = -np.inf
    r80_orthogonal_pairs_confirmed = 0
    r80_orthogonal_pairs_total = 0
    r80_max_abs_dot = 0.0

    for group_values, df_pitch in df_work.groupby(list(group_id_cols), dropna=False):
        group_key = _group_key_map(group_id_cols, group_values)
        output_ids = _first_id_values(df_pitch, output_id_cols, group_key)
        plot_id = _plot_label(output_ids, group_key)

        outcome, skip = _infer_outcome_with_take_skip(df_pitch)
        if skip:
            outcome_counts["TAKE_skipped"] += 1
            continue
        if df_pitch.empty:
            outcome_counts["empty_frames_skipped"] += 1
            continue

        left_back_foot = df_pitch["LEFTFOOT_TX"].iloc[0]
        right_back_foot = df_pitch["RIGHTFOOT_TX"].iloc[0]
        handedness = _infer_handedness_from_report(df_pitch)
        back_foot_x_pos = left_back_foot if handedness == "L" else right_back_foot

        ball_global = _to_vec3(df_pitch, "CENTER_TX", "CENTER_TY", "CENTER_TZ")
        top_global = _to_vec3(df_pitch, "TOP_TX", "TOP_TY", "TOP_TZ")
        knob_global = _to_vec3(df_pitch, "KNOB_TX", "KNOB_TY", "KNOB_TZ")
        n_frames = ball_global.shape[1]
        frame_vec = _frame_vector(df_pitch, n_frames)
        axis_backfill_indices = _axis_backfill_indices(df_pitch, frame_vec)

        if n_frames == 0:
            outcome_counts["empty_frames_skipped"] += 1
            continue

        processed_count += 1
        count_key = outcome if pd.notna(outcome) else "no_report"
        outcome_counts[count_key] = outcome_counts.get(count_key, 0) + 1

        ss_positions = _sweet_spot_positions(knob_global, top_global)
        ss_global = {
            "K80": ss_positions["ss_k80"],
        }

        bat_80_global, r_bat80 = _bat_80_lcs(
            knob_global,
            top_global,
            handedness=handedness,
            axis_backfill_indices=axis_backfill_indices,
        )
        confirmed, total, norm_min, norm_max = _r80_unit_vector_counts(r_bat80)
        r80_unit_vectors_confirmed += confirmed
        r80_unit_vectors_total += total
        if np.isfinite(norm_min):
            r80_norm_min = min(r80_norm_min, norm_min)
        if np.isfinite(norm_max):
            r80_norm_max = max(r80_norm_max, norm_max)
        orthogonal_confirmed, orthogonal_total, max_abs_dot = (
            _r80_orthogonal_pair_counts(r_bat80)
        )
        r80_orthogonal_pairs_confirmed += orthogonal_confirmed
        r80_orthogonal_pairs_total += orthogonal_total
        if np.isfinite(max_abs_dot):
            r80_max_abs_dot = max(r80_max_abs_dot, max_abs_dot)

        ball_local = _global_to_bat_local(ball_global, bat_80_global, r_bat80)
        knob_local = _global_to_bat_local(knob_global, bat_80_global, r_bat80)
        top_local = _global_to_bat_local(top_global, bat_80_global, r_bat80)
        ss_local = {
            spot: _global_to_bat_local(position, bat_80_global, r_bat80)
            for spot, position in ss_global.items()
        }

        miss_global = {spot: ball_global - position for spot, position in ss_global.items()}
        miss_local = {"K80": ball_local}
        distance_global = {
            spot: np.linalg.norm(vector, axis=0)
            for spot, vector in miss_global.items()
        }
        # Local missed distance uses the full local 3D miss-vector norm.
        # Because R_80 is orthonormal, this should match the global K80
        # distance at the same frame while preserving local X/Y/Z context.
        distance_local = {
            spot: _local_selection_distance(vector)
            for spot, vector in miss_local.items()
        }
        local_selection_metric = {
            spot: _local_selection_distance(vector)
            for spot, vector in miss_local.items()
        }

        velocity_global = {
            spot: _filtered_vector_velocity(vector)
            for spot, vector in miss_global.items()
        }
        velocity_local = {
            spot: _filtered_vector_velocity(vector)
            for spot, vector in miss_local.items()
        }
        speed_global = {
            spot: _speed_from_velocity(vector)
            for spot, vector in velocity_global.items()
        }
        speed_local = {
            spot: _speed_from_velocity(vector)
            for spot, vector in velocity_local.items()
        }

        event_times = _event_times(df_pitch)
        downswing_start_idx = _window_start_index(df_pitch, frame_vec)
        bat_stop_idx = _window_end_index(df_pitch, frame_vec)
        ball_start_frame = event_times.get("BALL_START")
        if save_validation_plots and validation_plot_format != "none":
            for spot in SWEET_SPOTS:
                idx = _minimum_index_in_window(
                    distance_global[spot],
                    downswing_start_idx,
                    bat_stop_idx,
                )
                if idx is None:
                    logger.warning(
                        "Skipping validation plot for %s/%s: no valid frames from downswing to bat stop.",
                        plot_id,
                        spot,
                    )
                    continue
                if validation_plot_format in {"png", "both"}:
                    _plot_md_validation(
                        distance_global[spot],
                        idx,
                        frame=frame_vec,
                        pitch_id=plot_id,
                        out_dir=VALIDATION_PLOT_ROOT / spot,
                        ball_z=ball_global[2, :],
                        target_z=ss_global[spot][2, :],
                        events=event_times,
                        ball_start_frame=ball_start_frame,
                    )
                if validation_plot_format in {"html", "both"}:
                    _plot_md_validation_3d(
                        ball_x=ball_global[0, :],
                        ball_y=ball_global[1, :],
                        ball_z=ball_global[2, :],
                        target_x=ss_global[spot][0, :],
                        target_y=ss_global[spot][1, :],
                        target_z=ss_global[spot][2, :],
                        t_min=idx,
                        pitch_id=plot_id,
                        out_dir=VALIDATION_PLOT_ROOT / spot,
                        events=event_times,
                    )

        meta_pitch = first_non_null_map(df_pitch, META_DESIRED_METRICS)
        pitch_ts_full = first_non_null(df_pitch.get("PITCH_TIMESTAMP"))
        session_date, pitch_time = split_timestamp(pitch_ts_full)
        if "SESSION_DATE" not in meta_pitch and pd.notna(session_date):
            meta_pitch["SESSION_DATE"] = session_date
        if "PITCH_TIMESTAMP" not in meta_pitch and pd.notna(pitch_time):
            meta_pitch["PITCH_TIMESTAMP"] = pitch_time

        jersey = meta_pitch.get("PLAYER_JERSEY_NUMBER", np.nan)
        gcs_path = meta_pitch.get("GCS_PATH", np.nan)
        meta_pitch["BATTER_NAME"] = parse_batter_name(gcs_path, jersey)
        _fill_derived_report_tags(meta_pitch, outcome)

        row: dict[str, object] = {
            **output_ids,
            "OUTCOME": outcome,
        }
        _add_start_data_positions(row, df_pitch, frame_vec)

        for spot in SWEET_SPOTS:
            global_idx = _minimum_index_in_window(
                distance_global[spot],
                downswing_start_idx,
                bat_stop_idx,
            )
            local_idx = _minimum_index_in_window(
                local_selection_metric[spot],
                downswing_start_idx,
                bat_stop_idx,
            )

            if global_idx is not None:
                in_band = _axial_band_check(
                    knob_global[:, global_idx],
                    ball_global[:, global_idx],
                    ss_positions["ss_top"][:, global_idx],
                    ss_positions["ss_bottom"][:, global_idx],
                )
                row[f"IN_BAND_{spot}"] = (1 if in_band else 0) if outcome == "hit" else np.nan
            else:
                logger.warning(
                    "%s/%s has no valid global distance from downswing to bat stop; discrete outputs will be NaN.",
                    plot_id,
                    spot,
                )
                row[f"IN_BAND_{spot}"] = np.nan

            _add_metrics_for_space(
                row,
                spot=spot,
                space="GLOBAL",
                idx=global_idx,
                frame_value=frame_vec[global_idx] if global_idx is not None else np.nan,
                distance=distance_global[spot],
                miss_vector=miss_global[spot],
                velocity=velocity_global[spot],
                speed=speed_global[spot],
                ball_position=ball_global,
                bat_knob_position=knob_global,
                bat_top_position=top_global,
                sweet_spot_position=ss_global[spot],
            )
            _add_window_max_metrics(
                row,
                spot=spot,
                space="GLOBAL",
                start_idx=downswing_start_idx,
                end_idx=global_idx,
                velocity=velocity_global[spot],
                speed=speed_global[spot],
            )
            _add_metrics_for_space(
                row,
                spot=spot,
                space="LOCAL",
                idx=local_idx,
                frame_value=frame_vec[local_idx] if local_idx is not None else np.nan,
                distance=distance_local[spot],
                miss_vector=miss_local[spot],
                velocity=velocity_local[spot],
                speed=speed_local[spot],
                ball_position=ball_local,
                bat_knob_position=knob_local,
                bat_top_position=top_local,
                sweet_spot_position=ss_local[spot],
            )
            _add_local_miss_direction_flags(
                row,
                miss_local[spot],
                local_idx,
            )
            _add_window_max_metrics(
                row,
                spot=spot,
                space="LOCAL",
                start_idx=downswing_start_idx,
                end_idx=local_idx,
                velocity=velocity_local[spot],
                speed=speed_local[spot],
            )

        for key, value in meta_pitch.items():
            if key in row:
                continue
            if key in LEGACY_INTERNAL_ID_COLS and key not in output_id_cols:
                continue
            row[key] = value

        metrics_rows.append(row)

        ts_ids = {col: output_ids[col] for col in group_id_cols if col in output_ids}
        ts_data: dict[str, object] = {
            **ts_ids,
            "FRAME": frame_vec,
            "BALL_X": ball_global[0],
            "BALL_Y": ball_global[1],
            "BALL_Z": ball_global[2],
            "BAT_KNOB_X": knob_global[0],
            "BAT_KNOB_Y": knob_global[1],
            "BAT_KNOB_Z": knob_global[2],
            "BAT_TOP_X": top_global[0],
            "BAT_TOP_Y": top_global[1],
            "BAT_TOP_Z": top_global[2],
            "BALL_IN_BAT_X": ball_local[0],
            "BALL_IN_BAT_Y": ball_local[1],
            "BALL_IN_BAT_Z": ball_local[2],
        }
        for col in FOOT_ANKLE_COLS:
            if col in df_pitch.columns:
                ts_data[col] = df_pitch[col].to_numpy(float)
        for spot in SWEET_SPOTS:
            _add_time_series_spot_columns(
                ts_data,
                spot=spot,
                ss_global=ss_global[spot],
                miss_global=miss_global[spot],
                miss_local=miss_local[spot],
                distance_global=distance_global[spot],
                distance_local=distance_local[spot],
                velocity_global=velocity_global[spot],
                velocity_local=velocity_local[spot],
                speed_global=speed_global[spot],
                speed_local=speed_local[spot],
            )

        ts_chunks.append(pd.DataFrame(ts_data))

    metrics_df = pd.DataFrame.from_records(metrics_rows)
    metrics_df = _reorder_metrics_columns(metrics_df, output_id_cols)
    time_series = (
        pd.concat(ts_chunks, ignore_index=True)
        if ts_chunks
        else pd.DataFrame(columns=[*group_id_cols, "FRAME"])
    )

    metrics_df = dedupe_columns(metrics_df)
    time_series = dedupe_columns(time_series)

    drop_internal_cols = [
        col for col in LEGACY_INTERNAL_ID_COLS if col not in set(output_id_cols)
    ]
    metrics_df = metrics_df.drop(columns=drop_internal_cols, errors="ignore")
    time_series = time_series.drop(columns=drop_internal_cols, errors="ignore")
    time_series = time_series[_time_series_column_order(time_series, group_id_cols)]

    if metrics_df.shape[0] != processed_count:
        raise AssertionError(
            f"Per-pitch rows mismatch: expected processed {processed_count}, got {metrics_df.shape[0]}"
        )

    distance_cols = [
        column for column in metrics_df.columns if column.startswith("MISSED_DISTANCE_")
    ]
    if distance_cols and (metrics_df[distance_cols] < 0).to_numpy().any():
        raise AssertionError("Negative miss distance detected.")

    _log_r80_unit_vector_check(
        r80_unit_vectors_confirmed,
        r80_unit_vectors_total,
        r80_norm_min,
        r80_norm_max,
    )
    _log_r80_orthogonality_check(
        r80_orthogonal_pairs_confirmed,
        r80_orthogonal_pairs_total,
        r80_max_abs_dot,
    )

    return metrics_df, time_series, outcome_counts
