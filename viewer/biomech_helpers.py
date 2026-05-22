"""Vendored biomechanics helpers for the hitting viewer.

This module is intentionally self-contained: it replaces the viewer's
former dependency on the project-wide ``biomech_functions`` package so
the viewer can ship with only its own folder.

Provenance
----------
Functions and constants here were lifted from
``biomech_functions/functions.py`` (May 2026 snapshot) and pruned to the
narrow set the viewer actually calls:

    * ``_sweet_spot_positions``
    * ``_bat_80_lcs`` (wraps ``construct_bat_80_lcs``)
    * ``_closest_distance_and_vector``
    * ``_infer_outcome_with_take_skip``

The only behavioural change from the original is in
``_closest_distance_and_vector``: the ``save_plot=True`` branch (which
wrote matplotlib + plotly validation figures via internal helpers
``_plot_md_validation`` / ``_plot_md_validation_3d``) has been removed.
The viewer only ever calls it with ``save_plot=False`` (see
``hitting_viewer_app.py`` line ~1709). To make this contract explicit
and prevent silent regressions, ``save_plot=True`` now raises
``NotImplementedError`` rather than silently doing nothing.

If the figure outputs are ever needed again, restore them from the
project's pre-cleanup ``biomech_functions/functions.py`` at the same
git tag this folder ships under.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Sweet-spot fractions along the knob->top bat axis.
SWEET_SPOTS: dict[str, float] = {"K80": 0.80}
BAND_LO_FRACTION: float = 0.79
BAND_HI_FRACTION: float = 0.85


# ---------------------------------------------------------------------------
# Vector helpers
# ---------------------------------------------------------------------------
def _normalize_vectors(vectors: np.ndarray, *, eps: float = 1e-8) -> np.ndarray:
    """Row-wise unit-normalize an ``(N, 3)`` array; near-zero rows return zeros."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = np.zeros_like(vectors, dtype=float)
    valid = norms.squeeze(axis=1) > eps
    normalized[valid] = vectors[valid] / norms[valid]
    return normalized


def _fallback_axis_candidates(y_hat: np.ndarray) -> np.ndarray:
    """Deterministic perpendicular axis candidates when bat motion is degenerate."""
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

    # Prefer caller-supplied anchors (e.g. BALL_MIN, BAT_STOP) when valid.
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

    # Otherwise fall back to forward-fill from first valid frame.
    first_valid = valid_indices[0]
    filled[:first_valid] = filled[first_valid]

    last_valid = first_valid
    for idx in range(first_valid + 1, len(filled)):
        if norms[idx] > eps:
            last_valid = idx
        else:
            filled[idx] = filled[last_valid]
    return filled


def _project_reference_onto_bat_normal_plane(
    reference: np.ndarray,
    y_hat: np.ndarray,
) -> np.ndarray:
    """Project a field reference into each frame's plane perpendicular to Y."""
    ref = np.asarray(reference, dtype=float)
    return ref - np.sum(y_hat * ref, axis=1, keepdims=True) * y_hat


def _is_left_handed_hitter(handedness: object) -> bool:
    """Return True for left-handed hitter labels (``L``, ``LH``, ``LHH``, ...)."""
    if handedness is None or pd.isna(handedness):
        return False
    label = str(handedness).strip().upper()
    return label in {"L", "LH", "LHH", "LEFT", "LEFTY", "LEFT-HANDED"}


def _minimum_index(distance: np.ndarray) -> int | None:
    """Return the frame index of the minimum distance, or None if all NaN."""
    if distance.size == 0:
        return None
    if np.all(np.isnan(distance)):
        return None
    return int(np.nanargmin(distance))


# ---------------------------------------------------------------------------
# Outcome / flag helpers
# ---------------------------------------------------------------------------
def _flag_is_set(df_pitch: pd.DataFrame, col: str) -> bool:
    """Whether a boolean flag column is set for this pitch.

    BATTING_REPORTS stores the column name as the value when true
    (e.g. ``SWING`` column = ``'SWING'``) and NaN when false, so a
    non-empty ``dropna()`` means the flag is set.
    """
    if col not in df_pitch.columns:
        return False
    return not df_pitch[col].dropna().empty


def _infer_outcome_with_take_skip(df_pitch: pd.DataFrame) -> tuple[str, bool]:
    """Classify a pitch as hit, miss, check_swing, take, or unknown.

    Flag columns (TAKE, SWING, BALL_CONTACT, MISS, CHECK_SWING) are
    independent boolean indicators — a non-NaN value means the flag is
    set. Returns ``(outcome_label, skip_flag)``.
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


# ---------------------------------------------------------------------------
# Sweet-spot geometry
# ---------------------------------------------------------------------------
def _sweet_spot_positions(knob: np.ndarray, top: np.ndarray) -> dict[str, np.ndarray]:
    """Return K80 sweet-spot and contact-band loci along the knob->top axis."""
    bat = top - knob
    return {
        "ss_k80": knob + SWEET_SPOTS["K80"] * bat,
        "ss_top": knob + BAND_HI_FRACTION * bat,
        "ss_bottom": knob + BAND_LO_FRACTION * bat,
    }


# ---------------------------------------------------------------------------
# BAT_80 local coordinate system
# ---------------------------------------------------------------------------
def construct_bat_80_lcs(
    top: np.ndarray,
    knob: np.ndarray,
    handedness: object = None,
    axis_backfill_indices: Sequence[int | None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct the BAT_80 local coordinate system.

    Parameters
    ----------
    top, knob:
        Arrays of shape ``(n_frames, 3)`` in raw/source Kinatrax coordinates.
    handedness:
        Optional hitter side. Right-handed hitters use the base
        configuration; left-handed hitters flip Z.
    axis_backfill_indices:
        Optional ordered frame indices used to backfill early Z-axis
        candidates. The pipeline passes BALL_MIN first and BAT_STOP
        second when available.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``bat_80`` has shape ``(n_frames, 3)``; ``r_bat80`` has shape
        ``(n_frames, 3, 3)`` with columns ``[x_hat, y_hat, z_hat]``.
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

    # --- Z axis from -(Y x dBAT), backfilled, with degenerate fallback ----
    z_candidates = _backfill_axis_candidates(
        -np.cross(y_hat, bat_temp),
        anchor_indices=axis_backfill_indices,
    )
    z_candidates = _project_reference_onto_bat_normal_plane(z_candidates, y_hat)
    degenerate = np.linalg.norm(z_candidates, axis=1) <= 1e-8
    if degenerate.any():
        z_candidates[degenerate] = _fallback_axis_candidates(y_hat[degenerate])
    z_hat = _normalize_vectors(z_candidates)

    # --- X axis (handedness-invariant) and handedness flip on Z -----------
    # X = -normalize(Y x Z) is invariant under the handedness flip on Z
    # because cross(y, -z) = -cross(y, z), which cancels the explicit
    # right-handed-hitter sign flip from the original two-branch impl.
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
    """Build BAT_80-origin local axes for internal ``(3, n_frames)`` inputs.

    Thin wrapper around :func:`construct_bat_80_lcs` that transposes
    between the viewer's ``(3, n_frames)`` convention and the LCS
    builder's ``(n_frames, 3)`` convention.
    """
    bat_80, r_bat80 = construct_bat_80_lcs(
        top.T,
        knob.T,
        handedness=handedness,
        axis_backfill_indices=axis_backfill_indices,
    )
    return bat_80.T, r_bat80


# ---------------------------------------------------------------------------
# Closest-distance computation
# ---------------------------------------------------------------------------
def _closest_distance_and_vector(
    ball: np.ndarray,
    target: np.ndarray,
    *,
    pitch_id: str | int | None = None,
    frame: np.ndarray | None = None,
    save_plot: bool = False,
    out_dir: object = None,
    events: dict[str, float] | None = None,
    ball_start_frame: float | None = None,
) -> tuple[int, float, np.ndarray]:
    """Return ``(t_min, distance_at_t_min, ball_minus_target_at_t_min)``.

    Inputs are ``(3, N)`` arrays of synchronized ball and target
    positions in the same frame. The validation-plot branch from the
    original pipeline has been removed; passing ``save_plot=True``
    raises ``NotImplementedError`` to fail loudly rather than silently.
    """
    # Unused kwargs are kept in the signature for call-site compatibility.
    del pitch_id, frame, out_dir, events, ball_start_frame

    if save_plot:
        raise NotImplementedError(
            "save_plot=True is not supported in the vendored viewer "
            "helper. Re-vendor _plot_md_validation* from "
            "biomech_functions/functions.py if validation figures are "
            "needed again."
        )

    if ball.shape != target.shape:
        raise ValueError(
            f"Ball and target shapes do not match: {ball.shape} vs {target.shape}"
        )
    if ball.shape[0] != 3:
        raise ValueError(f"Expected 3 x N arrays but got {ball.shape}")

    miss_all = ball - target
    distances = np.linalg.norm(miss_all, axis=0)
    t_min = _minimum_index(distances)
    if t_min is None:
        raise ValueError("No finite distance values to take a minimum over.")

    return t_min, float(distances[t_min]), miss_all[:, t_min].copy()


__all__ = [
    "SWEET_SPOTS",
    "BAND_LO_FRACTION",
    "BAND_HI_FRACTION",
    "construct_bat_80_lcs",
    "_bat_80_lcs",
    "_closest_distance_and_vector",
    "_infer_outcome_with_take_skip",
    "_sweet_spot_positions",
]
