# hitting_viewer_app.py
# -------------------------------------------------------------
# Flask application for visualizing hitting data from KT Ocean CSV
# Features: MLBAM_GUID/Game search, frame-by-frame playback, 3D visualization
# of batter skeleton, bat, and ball positions
# -------------------------------------------------------------

import os
import glob
import argparse
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file
from pathlib import Path
import traceback
import json
import sys

VIEWER_FPS = 300.0
VIEWER_DT = 1.0 / VIEWER_FPS
VIEWER_LOWPASS_CUTOFF_HZ = 30.0
VIEWER_LOWPASS_ORDER = 4

# ---------------------------------------------------------------------------
# Data directory — always resolved relative to this script so the app works
# regardless of the working directory it is launched from.
#
# The viewer lives at <project>/viewer/, but the project's data/ folder
# lives at <project>/data/. Resolve one level up from the script dir so
# the viewer keeps working after the May-2026 cleanup that moved it into
# its own folder.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
DATA_DIR = os.path.join(_PROJECT_ROOT, "data")

# ---------------------------------------------------------------------------
# Data discovery helpers
# ---------------------------------------------------------------------------

def discover_game_dirs(data_dir: str = DATA_DIR) -> list[str]:
    """
    Return a sorted list of game-ID subdirectory paths under data_dir.
    Each valid subdirectory must contain motion_sequence.csv or .csv.gz.
    """
    game_dirs = []
    if not os.path.isdir(data_dir):
        return game_dirs
    for entry in sorted(os.scandir(data_dir), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        # A valid game directory must have a motion_sequence file
        has_motion = (
            os.path.exists(os.path.join(entry.path, "motion_sequence.csv"))
            or os.path.exists(os.path.join(entry.path, "motion_sequence.csv.gz"))
        )
        if has_motion:
            game_dirs.append(entry.path)
    return game_dirs


def motion_sequence_path(game_dir: str) -> str | None:
    """
    Return the motion_sequence file path for a game directory,
    preferring the uncompressed CSV over .gz.
    Returns None if neither exists.
    """
    plain = os.path.join(game_dir, "motion_sequence.csv")
    gz    = os.path.join(game_dir, "motion_sequence.csv.gz")
    if os.path.exists(plain):
        return plain
    if os.path.exists(gz):
        return gz
    return None


def load_game_csv(game_dir: str) -> pd.DataFrame:
    """
    Load motion_sequence.csv (or .gz) from a single game directory
    and tag each row with the game ID derived from the directory name.
    Also merges hitting_report.csv columns when present (reuses the
    existing merge_nearby_hitting_report helper after loading).
    """
    path = motion_sequence_path(game_dir)
    if path is None:
        raise FileNotFoundError(f"No motion_sequence CSV found in: {game_dir}")
    game_id = os.path.basename(game_dir)
    df = pd.read_csv(path, low_memory=False)
    # Inject game ID so rows are traceable when multiple games are combined
    if "MLBAM_GAME_ID" not in df.columns:
        df.insert(0, "MLBAM_GAME_ID", game_id)
    return df, path


def build_combined_csv(game_dirs: list[str], out_path: str) -> str:
    """
    Concatenate motion_sequence data from all supplied game directories,
    write the result to out_path, and return out_path.
    Skips any game directory that fails to load (logs a warning).
    """
    frames = []
    for gdir in game_dirs:
        try:
            df, src = load_game_csv(gdir)
            print(f"  ✅ Loaded {len(df):,} rows from {os.path.basename(gdir)}")
            frames.append(df)
        except Exception as exc:
            print(f"  ⚠️  Skipping {os.path.basename(gdir)}: {exc}")

    if not frames:
        raise RuntimeError("No game data could be loaded from any game directory.")

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(out_path, index=False)
    print(f"  💾 Combined CSV written → {out_path}  ({len(combined):,} rows)")
    return out_path


def resolve_csv_path(arg: str | None, data_dir: str = DATA_DIR) -> str | None:
    """
    Resolve the user-supplied argument to a concrete CSV path.

    Resolution order:
      1. None or omitted  → return None (lazy mode: game loaded on demand via UI).
      2. Explicit path to an existing file → use directly (backward compat).
      3. A numeric game-ID string (e.g. '823871') → data/<id>/motion_sequence.csv.
      4. 'all' → concat all discovered games into data/combined_motion_sequence.csv.

    Lazy mode (None) is the default when the app is launched without arguments.
    The UI's Game ID dropdown will trigger /api/select-game/<id> to load on demand.
    """
    # --- Case 1: no argument — start empty, let the UI pick the game ---
    if not arg:
        print("📂 Lazy mode: no game pre-loaded — select a game from the UI dropdown")
        return None

    # --- Case 2: direct file path ---
    if os.path.isfile(arg):
        return arg

    # --- Case 3: numeric game ID ---
    if arg.isdigit():
        game_dir = os.path.join(data_dir, arg)
        path = motion_sequence_path(game_dir)
        if path is None:
            raise FileNotFoundError(
                f"Game '{arg}' not found. Expected: {game_dir}/motion_sequence.csv[.gz]"
            )
        print(f"📂 Single game mode: {arg}")
        return path

    # --- Case 4: explicit 'all' — concat every available game ---
    if arg.lower() == "all":
        game_dirs = discover_game_dirs(data_dir)
        if not game_dirs:
            raise FileNotFoundError(
                f"No game directories found under: {data_dir}\n"
                "Each game needs a subdirectory containing motion_sequence.csv"
            )
        game_ids = [os.path.basename(d) for d in game_dirs]
        print(f"📂 Multi-game mode: loading {len(game_dirs)} games → {game_ids}")
        out_path = os.path.join(data_dir, "combined_motion_sequence.csv")
        return build_combined_csv(game_dirs, out_path)

    raise ValueError(f"Unrecognised game argument: '{arg}'")


# Local vendored helpers (formerly biomech_functions.functions).
# See viewer/biomech_helpers.py for the provenance note. Keeping the
# script dir on sys.path so this works whether the app is launched as
# `python viewer/hitting_viewer_app.py` from the project root or from
# inside viewer/.
sys.path.insert(0, _SCRIPT_DIR)
from biomech_helpers import (
    _bat_80_lcs,
    _closest_distance_and_vector,
    _infer_outcome_with_take_skip,
    _sweet_spot_positions,
)

# -----------------------------
# Helper functions for data sanitization
# -----------------------------
def sanitize_for_json(obj):
    """
    Recursively sanitize numpy types and other non-JSON-serializable objects.
    Converts numpy scalars, arrays, and pandas types to native Python types.
    """
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(item) for item in obj]
    elif pd.isna(obj):
        return None
    return obj

# -----------------------------
# Coordinate remapping helper
# -----------------------------
def remap_coords(tx, ty, tz):
    """
    Remap coordinates from data format to visualization format.
    Data format: TX, TY, TZ
    Visualization format: X=Z, Y=X, Z=-Y
    
    Args:
        tx: Original X coordinate from data
        ty: Original Y coordinate from data
        tz: Original Z coordinate from data
    
    Returns:
        Tuple of (x, y, z) for visualization
    """
    # Map: X (viz) = Z (data), Y (viz) = X (data), Z (viz) = -Y (data)
    return tz, tx, -ty


def coord_columns(columns, raw_prefix, output_prefix=None):
    """Return source-coordinate columns for raw motion or pipeline time-series data."""
    raw_cols = (f"{raw_prefix}_TX", f"{raw_prefix}_TY", f"{raw_prefix}_TZ")
    if all(col in columns for col in raw_cols):
        return raw_cols

    if output_prefix is not None:
        output_cols = (
            f"{output_prefix}_X",
            f"{output_prefix}_Y",
            f"{output_prefix}_Z",
        )
        if all(col in columns for col in output_cols):
            return output_cols

    return None


def row_point(row, cols):
    """Extract one XYZ point from a one-row DataFrame."""
    return np.array([float(row[col].iloc[0]) for col in cols], dtype=float)


def infer_handedness(pitch_df):
    """Return hitter handedness from report flags or foot positions when available."""
    if "R" in pitch_df.columns and pitch_df["R"].notna().any():
        if pitch_df["R"].dropna().astype(str).str.strip().str.upper().eq("R").any():
            return "R"
    if "L" in pitch_df.columns and pitch_df["L"].notna().any():
        if pitch_df["L"].dropna().astype(str).str.strip().str.upper().eq("L").any():
            return "L"
    if {"LEFTFOOT_TX", "RIGHTFOOT_TX"}.issubset(pitch_df.columns):
        left_back_foot = pitch_df["LEFTFOOT_TX"].iloc[0]
        right_back_foot = pitch_df["RIGHTFOOT_TX"].iloc[0]
        if pd.notna(right_back_foot) and right_back_foot < 0:
            return "R"
        if pd.notna(left_back_foot) and left_back_foot > 0:
            return "L"
    return None


def infer_swing_status(pitch_df):
    """Return SWING/NO_SWING/UNKNOWN for one pitch."""
    if "SWING" in pitch_df.columns and pitch_df["SWING"].notna().any():
        swing_values = pitch_df["SWING"].dropna().astype(str).str.strip().str.upper()
        if swing_values.isin({"SWING", "TRUE", "1", "YES", "Y"}).any():
            return "SWING"
        if swing_values.isin({"NO_SWING", "FALSE", "0", "NO", "N"}).any():
            return "NO_SWING"
    if "TAKE" in pitch_df.columns and pitch_df["TAKE"].notna().any():
        take_values = pitch_df["TAKE"].dropna().astype(str).str.strip().str.upper()
        if take_values.isin({"TAKE", "TRUE", "1", "YES", "Y"}).any():
            return "NO_SWING"
    return "UNKNOWN"


def event_frame_index(pitch_df, event_col):
    """Return pitch-local index for an event frame value when available."""
    if event_col not in pitch_df.columns or "FRAME" not in pitch_df.columns:
        return None

    values = pitch_df[event_col].dropna()
    if values.empty:
        return None

    frames = pd.to_numeric(pitch_df["FRAME"], errors="coerce").to_numpy(float)
    if frames.size == 0:
        return None

    try:
        event_value = float(values.iloc[0])
    except (TypeError, ValueError):
        return None

    exact = np.where(np.isclose(frames, event_value, equal_nan=False))[0]
    if exact.size:
        return int(exact[0])

    positional_idx = int(round(event_value))
    if 0 <= positional_idx < frames.size:
        return positional_idx
    return None


def event_frame_indices(pitch_df, event_cols):
    """Return ordered pitch-local event indices, skipping unavailable events."""
    return [
        idx
        for idx in (event_frame_index(pitch_df, event_col) for event_col in event_cols)
        if idx is not None
    ]


def pitch_label(pitch_id, pitch_df):
    """Return UI label with handedness and swing status before the identifier."""
    handedness = infer_handedness(pitch_df) or "?"
    if handedness not in {"R", "L"}:
        handedness = handedness[:1] if handedness else "?"
    return f"{handedness} | {infer_swing_status(pitch_df)} | {pitch_id}"


def merge_nearby_hitting_report(df, csv_path, id_col, game_col=None):
    """Attach lightweight report columns for viewer labels when a sibling report exists."""
    report_path = Path(csv_path).with_name("hitting_report.csv")
    if not report_path.exists() or Path(csv_path).resolve() == report_path.resolve():
        return df

    report_header = pd.read_csv(report_path, nrows=0).columns.tolist()
    if id_col not in report_header:
        return df

    merge_keys = [id_col]
    if game_col is not None and game_col in df.columns and game_col in report_header:
        merge_keys.append(game_col)

    label_cols = [
        col
        for col in ("SWING", "TAKE", "BAD", "R", "L", "BALL_MIN", "BAT_STOP")
        if col in report_header and col not in df.columns
    ]
    if not label_cols:
        return df

    report = pd.read_csv(
        report_path,
        usecols=merge_keys + label_cols,
        low_memory=False,
    ).drop_duplicates(subset=merge_keys)

    before_cols = len(df.columns)
    merged = df.merge(report, on=merge_keys, how="left")
    print(
        f"📊 Merged {len(merged.columns) - before_cols} viewer label columns "
        f"from {report_path.name}"
    )
    return merged


def remap_point(point_xyz):
    """Remap a source XYZ point into viewer coordinates."""
    return remap_coords(point_xyz[0], point_xyz[1], point_xyz[2])


def projected_trace_coords(start_viz, end_viz, view):
    """Project a 3D line segment into the requested 2D viewer plane."""
    if view == 'Front':
        return [start_viz[0], end_viz[0]], [start_viz[2], end_viz[2]]
    if view == 'Side':
        return [start_viz[1], end_viz[1]], [start_viz[2], end_viz[2]]
    if view == 'Top':
        return [start_viz[0], end_viz[0]], [start_viz[1], end_viz[1]]
    raise ValueError(f"Unsupported 2D view: {view}")

def compute_axis_ranges(df, joints=None):
    """
    Compute axis ranges for 3D visualization based on available position data.
    Returns (x_range, y_range, z_range) tuples for Plotly.
    Note: Coordinates are remapped (X=Z, Y=X, Z=-Y) before computing ranges.
    """
    x_vals, y_vals, z_vals = [], [], []
    
    # Collect ball position and remap
    ball_cols = coord_columns(df.columns, 'CENTER', 'BALL')
    if ball_cols is not None:
        for _, row in df[list(ball_cols)].dropna().iterrows():
            tx, ty, tz = (row[col] for col in ball_cols)
            x, y, z = remap_coords(tx, ty, tz)
            x_vals.append(x)
            y_vals.append(y)
            z_vals.append(z)
    
    # Collect bat positions and remap
    for raw_prefix, output_prefix in [('TOP', 'BAT_TOP'), ('KNOB', 'BAT_KNOB')]:
        cols = coord_columns(df.columns, raw_prefix, output_prefix)
        if cols is not None:
            for _, row in df[list(cols)].dropna().iterrows():
                tx, ty, tz = (row[col] for col in cols)
                x, y, z = remap_coords(tx, ty, tz)
                x_vals.append(x)
                y_vals.append(y)
                z_vals.append(z)
    
    # Collect skeleton joint positions if available and remap
    if joints:
        for joint in joints:
            cols = [f"{joint}_TX", f"{joint}_TY", f"{joint}_TZ"]
            if all(col in df.columns for col in cols):
                for _, row in df[cols].dropna().iterrows():
                    tx, ty, tz = row[f"{joint}_TX"], row[f"{joint}_TY"], row[f"{joint}_TZ"]
                    x, y, z = remap_coords(tx, ty, tz)
                    x_vals.append(x)
                    y_vals.append(y)
                    z_vals.append(z)
    
    # Default ranges if no data found
    if not x_vals:
        x_vals = [-2, 2]
    if not y_vals:
        y_vals = [-2, 2]
    if not z_vals:
        z_vals = [0, 2]
    
    x_range = (min(x_vals) - 0.5, max(x_vals) + 0.5)
    y_range = (min(y_vals) - 0.5, max(y_vals) + 0.5)
    z_range = (max(0, min(z_vals) - 0.5), max(z_vals) + 0.5)
    
    return x_range, y_range, z_range

# -----------------------------
# Home plate visualization
# -----------------------------
def build_home_plate_mesh():
    """
    Create home plate as pentagon at z=0.01 (just above ground level for visibility).
    Returns numpy array of 3D coordinates.
    """
    # Home plate dimensions: 17" wide at back, comes to point
    # At z=0.01 to sit visibly on ground plane
    plate = np.array([
        [-0.708, 0.0, 0.01],   # back left corner
        [ 0.708, 0.0, 0.01],   # back right corner
        [ 0.708, 0.0, -0.354], # right side
        [ 0.0,   0.0, -0.708], # point (front)
        [-0.708, 0.0, -0.354], # left side
        [-0.708, 0.0, 0.01]    # close the loop
    ])
    return plate

def build_ground_plane(size=20):
    """
    Create a ground plane mesh at z=0 to represent dirt/field.
    Returns arrays for x, y, z coordinates suitable for surface plot.
    """
    x = np.linspace(-size/2, size/2, 2)
    y = np.linspace(-size/2, size/2, 2)
    xx, yy = np.meshgrid(x, y)
    zz = np.zeros_like(xx)  # Ground at z=0
    return xx, yy, zz

# -----------------------------
# Skeleton joint definitions for batter
# -----------------------------
# Key joints to visualize (simplified skeleton)
BATTER_JOINTS = [
    'HIPS', 'SPINE', 'NECK', 'HEAD',
    'LEFTUPPERARM', 'LEFTFOREARM', 'LEFTHAND',
    'RIGHTUPPERARM', 'RIGHTFOREARM', 'RIGHTHAND',
    'LEFTTHIGH', 'LEFTSHIN', 'LEFTANKLE', 'LEFTFOOT',
    'RIGHTTHIGH', 'RIGHTSHIN', 'RIGHTANKLE', 'RIGHTFOOT'
]

# Connections between joints (parent-child relationships)
BATTER_CONNECTIONS = [
    # Spine chain
    ('HIPS', 'SPINE'),
    ('SPINE', 'NECK'),
    ('NECK', 'HEAD'),
    # Left arm
    ('SPINE', 'LEFTUPPERARM'),
    ('LEFTUPPERARM', 'LEFTFOREARM'),
    ('LEFTFOREARM', 'LEFTHAND'),
    # Right arm
    ('SPINE', 'RIGHTUPPERARM'),
    ('RIGHTUPPERARM', 'RIGHTFOREARM'),
    ('RIGHTFOREARM', 'RIGHTHAND'),
    # Left leg
    ('HIPS', 'LEFTTHIGH'),
    ('LEFTTHIGH', 'LEFTSHIN'),
    ('LEFTSHIN', 'LEFTANKLE'),
    ('LEFTANKLE', 'LEFTFOOT'),
    # Right leg
    ('HIPS', 'RIGHTTHIGH'),
    ('RIGHTTHIGH', 'RIGHTSHIN'),
    ('RIGHTSHIN', 'RIGHTANKLE'),
    ('RIGHTANKLE', 'RIGHTFOOT'),
]

def available_joints(df):
    """
    Return list of joints that have complete X/Y/Z position data in the DataFrame.
    """
    joints = []
    for joint in BATTER_JOINTS:
        x_col = f"{joint}_TX"
        y_col = f"{joint}_TY"
        z_col = f"{joint}_TZ"
        if all(col in df.columns for col in [x_col, y_col, z_col]):
            # Check if at least one row has non-null values
            if df[[x_col, y_col, z_col]].notna().any(axis=1).any():
                joints.append(joint)
    return joints

# -----------------------------
# Plot generation function
# -----------------------------
def figure_payload_for_frame(
    df,
    df_indexed,
    frame,
    pitch_id=None,
    game_id=None,
    show_plate=True,
    show_lcs=True,
    view='3D',
):
    """
    Generate Plotly figure payload for a specific frame.
    
    Args:
        df: Full DataFrame with hitting data
        df_indexed: DataFrame indexed by MLBAM_GUID or legacy PITCH_ID for fast lookups
        frame: Frame number to display
        pitch_id: Optional pitch identifier to filter data
        game_id: Optional MLBAM_GAME_ID to disambiguate/filter data
        show_plate: Whether to show home plate
        show_lcs: Whether to show BAT_80 local coordinate axes
        view: View type - '3D', 'Front', 'Side', or 'Top'
    
    Returns:
        Dictionary with 'data' and 'layout' for Plotly (JSON-serializable)
    """
    import plotly.graph_objs as go
    
    # Filter to specific pitch if provided (use indexed DataFrame for speed)
    if pitch_id is not None:
        pitch_df = df_indexed.loc[[pitch_id]].copy() if pitch_id in df_indexed.index else pd.DataFrame()
        if len(pitch_df) == 0:
            print(f"Warning: No data found for pitch identifier {pitch_id}")
            return {"data": [], "layout": {}}
    else:
        pitch_df = df.copy()

    if game_id is not None and "MLBAM_GAME_ID" in pitch_df.columns:
        pitch_df = pitch_df[pitch_df["MLBAM_GAME_ID"].astype(str) == str(game_id)]
        if len(pitch_df) == 0:
            print(f"Warning: No data found for game identifier {game_id}")
            return {"data": [], "layout": {}}
    
    # Ensure FRAME column is numeric and sort
    pitch_df['FRAME'] = pd.to_numeric(pitch_df['FRAME'], errors='coerce')
    pitch_df = pitch_df.sort_values('FRAME').reset_index(drop=True)
    
    # Find the closest frame to requested frame
    frames = pitch_df["FRAME"]
    if frames.isna().all() or len(pitch_df) == 0:
        print("Warning: No valid FRAME values found")
        return {
            "data": [],
            "layout": {
                "annotations": [{
                    "text": "No data available for this frame",
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.5,
                    "y": 0.5,
                    "showarrow": False,
                    "font": {"size": 16, "color": "red"}
                }]
            }
        }
    
    # Get closest frame
    row_idx = 0
    valid_frames = frames.dropna()
    if len(valid_frames) == 0:
        row = pitch_df.iloc[[0]].copy()
    else:
        idx = (valid_frames - frame).abs().idxmin()
        row = pitch_df.loc[[idx]].copy()
        row_idx = int(idx)
        actual_frame = row['FRAME'].iloc[0]
        # Removed print statement for performance (too many during playback)
    
    # Get available joints
    joints = available_joints(pitch_df)
    # Removed print statement for performance (too many during playback)
    
    # Collect skeleton segments (connections between joints)
    x_lines, y_lines, z_lines = [], [], []
    for a, b in BATTER_CONNECTIONS:
        if a in joints and b in joints:
            try:
                # Extract original coordinates from data
                ax_data = float(row[f"{a}_TX"].iloc[0])
                ay_data = float(row[f"{a}_TY"].iloc[0])
                az_data = float(row[f"{a}_TZ"].iloc[0])
                bx_data = float(row[f"{b}_TX"].iloc[0])
                by_data = float(row[f"{b}_TY"].iloc[0])
                bz_data = float(row[f"{b}_TZ"].iloc[0])
                
                # Skip if any value is NaN
                if any(np.isnan([ax_data, ay_data, az_data, bx_data, by_data, bz_data])):
                    continue
                
                # Remap coordinates: X=Z, Y=X, Z=-Y
                ax, ay, az = remap_coords(ax_data, ay_data, az_data)
                bx, by, bz = remap_coords(bx_data, by_data, bz_data)
                
                x_lines += [ax, bx, None]
                y_lines += [ay, by, None]
                z_lines += [az, bz, None]
            except Exception as e:
                print(f"Warning: Could not process connection {a}-{b}: {e}")
                continue
    
    # Joint points
    jx, jy, jz, labels = [], [], [], []
    for j in joints:
        try:
            # Extract original coordinates from data
            tx_data = float(row[f"{j}_TX"].iloc[0])
            ty_data = float(row[f"{j}_TY"].iloc[0])
            tz_data = float(row[f"{j}_TZ"].iloc[0])
            
            if not any(np.isnan([tx_data, ty_data, tz_data])):
                # Remap coordinates: X=Z, Y=X, Z=-Y
                x_val, y_val, z_val = remap_coords(tx_data, ty_data, tz_data)
                jx.append(x_val)
                jy.append(y_val)
                jz.append(z_val)
                labels.append(j)
        except Exception as e:
            continue
    
    # Get bat positions (knob and top)
    bat_knob_x, bat_knob_y, bat_knob_z = None, None, None
    bat_top_x, bat_top_y, bat_top_z = None, None, None
    knob_cols = coord_columns(pitch_df.columns, 'KNOB', 'BAT_KNOB')
    top_cols = coord_columns(pitch_df.columns, 'TOP', 'BAT_TOP')
    try:
        if knob_cols is not None:
            # Extract original coordinates from data
            knob_point = row_point(row, knob_cols)
            if not np.isnan(knob_point).any():
                # Remap coordinates: X=Z, Y=X, Z=-Y
                bat_knob_x, bat_knob_y, bat_knob_z = remap_point(knob_point)
        
        if top_cols is not None:
            # Extract original coordinates from data
            top_point = row_point(row, top_cols)
            if not np.isnan(top_point).any():
                # Remap coordinates: X=Z, Y=X, Z=-Y
                bat_top_x, bat_top_y, bat_top_z = remap_point(top_point)
    except Exception as e:
        print(f"Warning: Could not get bat positions: {e}")
    
    # Get ball position
    ball_x, ball_y, ball_z = None, None, None
    ball_cols = coord_columns(pitch_df.columns, 'CENTER', 'BALL')
    try:
        if ball_cols is not None:
            # Extract original coordinates from data
            ball_point = row_point(row, ball_cols)
            if not np.isnan(ball_point).any():
                # Remap coordinates: X=Z, Y=X, Z=-Y
                ball_x, ball_y, ball_z = remap_point(ball_point)
                # Removed print statement for performance (too many during playback)
    except Exception as e:
        print(f"Warning: Could not get ball position: {e}")
    
    # Calculate and visualize sweet spot position (K80 at 80%)
    ss_k80_x, ss_k80_y, ss_k80_z = None, None, None
    try:
        if (bat_knob_x is not None and bat_top_x is not None and 
            knob_cols is not None and top_cols is not None):
            # Extract original coordinates for sweet spot calculation
            knob_point = row_point(row, knob_cols)
            top_point = row_point(row, top_cols)
            
            if not np.isnan([*knob_point, *top_point]).any():
                # Calculate sweet spots in original coordinate system
                knob_vec = knob_point.reshape(3, 1)
                top_vec = top_point.reshape(3, 1)
                
                # Calculate sweet spot positions
                ss_positions = _sweet_spot_positions(knob_vec, top_vec)
                
                # Extract K80 (80%) sweet spot
                ss_k80_orig = ss_positions["ss_k80"][:, 0]  # (3,)
                
                # Remap coordinates for visualization
                ss_k80_x, ss_k80_y, ss_k80_z = remap_coords(
                    ss_k80_orig[0], ss_k80_orig[1], ss_k80_orig[2]
                )
    except Exception as e:
        print(f"Warning: Could not calculate sweet spot positions: {e}")

    # Calculate BAT_80 local coordinate axes for visual inspection.
    lcs_axes = []
    try:
        if show_lcs and knob_cols is not None and top_cols is not None:
            knob_series = pitch_df[list(knob_cols)].to_numpy(dtype=float).T
            top_series = pitch_df[list(top_cols)].to_numpy(dtype=float).T
            bat_80_origin, r_bat80 = _bat_80_lcs(
                knob_series,
                top_series,
                handedness=infer_handedness(pitch_df),
                axis_backfill_indices=event_frame_indices(pitch_df, ("BALL_MIN", "BAT_STOP")),
            )
            lcs_idx = max(0, min(row_idx, bat_80_origin.shape[1] - 1))
            origin = bat_80_origin[:, lcs_idx]

            if not np.isnan(origin).any():
                axis_length = 0.30
                axis_specs = [
                    ("R80 X", r_bat80[lcs_idx, :, 0], "red"),
                    ("R80 Y", r_bat80[lcs_idx, :, 1], "green"),
                    ("R80 Z", r_bat80[lcs_idx, :, 2], "blue"),
                ]
                for name, axis_vector, color in axis_specs:
                    if np.isnan(axis_vector).any():
                        continue
                    endpoint = origin + axis_length * axis_vector
                    lcs_axes.append(
                        {
                            "name": name,
                            "color": color,
                            "start": remap_point(origin),
                            "end": remap_point(endpoint),
                        }
                    )
    except Exception as e:
        print(f"Warning: Could not calculate BAT_80 LCS axes: {e}")
    
    # Compute axis ranges (use only current pitch data for speed)
    xr, yr, zr = compute_axis_ranges(pitch_df, joints)
    xr = [float(xr[0]), float(xr[1])]
    yr = [float(yr[0]), float(yr[1])]
    zr = [float(zr[0]), float(zr[1])]
    
    traces = []
    
    # Check if 2D view requested
    is_2d = view in ['Front', 'Side', 'Top']
    
    if not is_2d:
        # 3D view - add ground plane
        ground_x, ground_y, ground_z = build_ground_plane(size=20)
        traces.append(go.Surface(
            x=ground_x.tolist(),
            y=ground_y.tolist(),
            z=ground_z.tolist(),
            colorscale=[[0, 'rgb(139, 90, 43)'], [1, 'rgb(139, 90, 43)']],
            showscale=False,
            name="Field",
            opacity=0.3,
            hoverinfo="skip"
        ))
    
    # Home plate (if enabled)
    if show_plate:
        hp = build_home_plate_mesh()
        if is_2d:
            if view == 'Front':  # X-Z plane
                traces.append(go.Scatter(
                    x=hp[:,0].tolist(), y=hp[:,2].tolist(),
                    mode="lines", line=dict(color="white", width=3),
                    name="Home Plate", showlegend=True
                ))
            elif view == 'Side':  # Y-Z plane
                traces.append(go.Scatter(
                    x=hp[:,1].tolist(), y=hp[:,2].tolist(),
                    mode="lines", line=dict(color="white", width=3),
                    name="Home Plate", showlegend=True
                ))
            elif view == 'Top':  # X-Y plane
                traces.append(go.Scatter(
                    x=hp[:,0].tolist(), y=hp[:,1].tolist(),
                    mode="lines+markers", line=dict(color="white", width=3),
                    marker=dict(size=8, color="white"),
                    name="Home Plate", showlegend=True, fill="toself", fillcolor="rgba(255,255,255,0.3)"
                ))
        else:
            # 3D view
            traces.append(go.Scatter3d(
                x=hp[:,0].tolist(),
                y=hp[:,1].tolist(),
                z=hp[:,2].tolist(),
                mode="lines",
                line=dict(color="white", width=8),
                name="Home Plate",
                showlegend=True,
                hoverinfo="name"
            ))
    
    # Skeleton lines
    if x_lines:
        if is_2d:
            if view == 'Front':  # X-Z plane
                traces.append(go.Scatter(
                    x=x_lines, y=z_lines,
                    mode="lines", line=dict(color="royalblue", width=4),
                    name="Skeleton", showlegend=True, hoverinfo="skip"
                ))
            elif view == 'Side':  # Y-Z plane
                traces.append(go.Scatter(
                    x=y_lines, y=z_lines,
                    mode="lines", line=dict(color="royalblue", width=4),
                    name="Skeleton", showlegend=True, hoverinfo="skip"
                ))
            elif view == 'Top':  # X-Y plane
                traces.append(go.Scatter(
                    x=x_lines, y=y_lines,
                    mode="lines", line=dict(color="royalblue", width=4),
                    name="Skeleton", showlegend=True, hoverinfo="skip"
                ))
        else:
            # 3D view
            traces.append(go.Scatter3d(
                x=x_lines, y=y_lines, z=z_lines,
                mode="lines",
                line=dict(color="royalblue", width=6),
                name="Skeleton",
                showlegend=True,
                hoverinfo="skip"
            ))
    
    # Joints
    if jx:
        if is_2d:
            if view == 'Front':
                plot_x, plot_y = jx, jz
            elif view == 'Side':
                plot_x, plot_y = jy, jz
            elif view == 'Top':
                plot_x, plot_y = jx, jy
            
            traces.append(go.Scatter(
                x=plot_x, y=plot_y,
                mode="markers",
                marker=dict(size=6, color="royalblue", line=dict(width=1, color="darkblue"), opacity=1.0),
                text=labels,
                name="Joints", showlegend=True,
                hovertemplate="%{text}<br>%{x:.2f}, %{y:.2f}<extra></extra>"
            ))
        else:
            # 3D view
            traces.append(go.Scatter3d(
                x=jx, y=jy, z=jz,
                mode="markers",
                marker=dict(size=5, color="royalblue", line=dict(width=1, color="darkblue"), opacity=1.0),
                text=labels,
                name="Joints",
                showlegend=True,
                hovertemplate="%{text}<br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>"
            ))
    
    # Bat (draw line from knob to top)
    if bat_knob_x is not None and bat_top_x is not None:
        if is_2d:
            if view == 'Front':
                bat_x = [bat_knob_x, bat_top_x]
                bat_y = [bat_knob_z, bat_top_z]
            elif view == 'Side':
                bat_x = [bat_knob_y, bat_top_y]
                bat_y = [bat_knob_z, bat_top_z]
            elif view == 'Top':
                bat_x = [bat_knob_x, bat_top_x]
                bat_y = [bat_knob_y, bat_top_y]
            
            traces.append(go.Scatter(
                x=bat_x, y=bat_y,
                mode="lines+markers",
                line=dict(color="brown", width=8),
                marker=dict(size=10, color="brown"),
                name="Bat", showlegend=True,
                hovertemplate="Bat<br>%{x:.2f}, %{y:.2f}<extra></extra>"
            ))
        else:
            # 3D view
            traces.append(go.Scatter3d(
                x=[bat_knob_x, bat_top_x],
                y=[bat_knob_y, bat_top_y],
                z=[bat_knob_z, bat_top_z],
                mode="lines+markers",
                line=dict(color="brown", width=10),
                marker=dict(size=8, color="brown"),
                name="Bat",
                showlegend=True,
                hovertemplate="Bat<br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>"
            ))
    
    # Ball position
    if ball_x is not None:
        if is_2d:
            if view == 'Front':
                ball_plot_x, ball_plot_y = ball_x, ball_z
            elif view == 'Side':
                ball_plot_x, ball_plot_y = ball_y, ball_z
            elif view == 'Top':
                ball_plot_x, ball_plot_y = ball_x, ball_y
            
            traces.append(go.Scatter(
                x=[ball_plot_x], y=[ball_plot_y],
                mode="markers",
                marker=dict(size=14, color="red", symbol="circle",
                           line=dict(width=2, color="darkred"), opacity=0.9),
                name="Ball", showlegend=True,
                hovertemplate="Ball<br>%{x:.2f}, %{y:.2f}<extra></extra>"
            ))
        else:
            # 3D view
            traces.append(go.Scatter3d(
                x=[ball_x], y=[ball_y], z=[ball_z],
                mode="markers",
                marker=dict(size=8, color="red", symbol="circle",
                           line=dict(width=2, color="darkred"), opacity=0.9),
                name="Ball",
                showlegend=True,
                hovertemplate="Ball<br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>"
            ))
    
    # Sweet spot position (K80 at 80%)
    if ss_k80_x is not None:
        if is_2d:
            if view == 'Front':
                ss_k80_plot_x, ss_k80_plot_y = ss_k80_x, ss_k80_z
            elif view == 'Side':
                ss_k80_plot_x, ss_k80_plot_y = ss_k80_y, ss_k80_z
            elif view == 'Top':
                ss_k80_plot_x, ss_k80_plot_y = ss_k80_x, ss_k80_y
            
            # K80 sweet spot (80%)
            traces.append(go.Scatter(
                x=[ss_k80_plot_x], y=[ss_k80_plot_y],
                mode="markers",
                marker=dict(size=16, color="orange", symbol="diamond",
                           line=dict(width=2, color="darkorange"), opacity=0.9),
                name="Sweet Spot K80 (80%)", showlegend=True,
                hovertemplate="Sweet Spot K80 (80%%)<br>%{x:.3f}, %{y:.3f}<extra></extra>"
            ))
        else:
            # 3D view
            # K80 sweet spot (80%)
            traces.append(go.Scatter3d(
                x=[ss_k80_x], y=[ss_k80_y], z=[ss_k80_z],
                mode="markers",
                marker=dict(size=12, color="orange", symbol="diamond",
                           line=dict(width=2, color="darkorange"), opacity=0.9),
                name="Sweet Spot K80 (80%)",
                showlegend=True,
                hovertemplate="Sweet Spot K80 (80%%)<br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>"
            ))

    # BAT_80 local coordinate system axes (X=red, Y=green, Z=blue)
    for axis in lcs_axes:
        start = axis["start"]
        end = axis["end"]
        if is_2d:
            line_x, line_y = projected_trace_coords(start, end, view)
            traces.append(go.Scatter(
                x=line_x,
                y=line_y,
                mode="lines+markers",
                line=dict(color=axis["color"], width=5),
                marker=dict(size=[6, 10], color=axis["color"]),
                name=axis["name"],
                showlegend=True,
                hovertemplate=f"{axis['name']}<br>%{{x:.3f}}, %{{y:.3f}}<extra></extra>",
            ))
        else:
            traces.append(go.Scatter3d(
                x=[start[0], end[0]],
                y=[start[1], end[1]],
                z=[start[2], end[2]],
                mode="lines+markers",
                line=dict(color=axis["color"], width=8),
                marker=dict(size=[4, 7], color=axis["color"]),
                name=axis["name"],
                showlegend=True,
                hovertemplate=(
                    f"{axis['name']}<br>"
                    "X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>"
                ),
            ))
    
    # Create layout based on view type
    if is_2d:
        if view == 'Front':
            xlabel, ylabel = "X (m)", "Z (m)"
            xrange = [-2, 2]
            yrange = [0, 2]
        elif view == 'Side':
            xlabel, ylabel = "Y (m)", "Z (m)"
            xrange = [2, -2]
            yrange = [0, 2]
        elif view == 'Top':
            xlabel, ylabel = "X (m)", "Y (m)"
            xrange = [-2, 2]
            yrange = [2, -2]
        
        layout = dict(
            xaxis=dict(title=xlabel, range=xrange, showgrid=True),
            yaxis=dict(title=ylabel, range=yrange, showgrid=True),
            plot_bgcolor="rgb(245, 245, 245)",
            margin=dict(l=50, r=50, t=30, b=50),
            showlegend=True
        )
    else:
        # 3D layout
        layout = dict(
            scene=dict(
                xaxis=dict(title="X (m)", range=xr, showspikes=False, showgrid=True),
                yaxis=dict(title="Y (m)", range=[yr[1], yr[0]], showspikes=False, showgrid=True),
                zaxis=dict(title="Z (m)", range=[0, zr[1]], showspikes=False, showgrid=True),
                aspectmode="manual",
                aspectratio=dict(x=1, y=1, z=0.75),
                camera=dict(
                    eye=dict(x=1.5, y=-1.5, z=1.0),
                    projection=dict(type='perspective')
                ),
                bgcolor="rgb(230, 240, 255)"
            ),
            margin=dict(l=0, r=0, t=10, b=0),
            showlegend=True
        )
    
    # Convert to JSON-serializable dict
    fig_temp = go.Figure(data=traces, layout=layout)
    fig_dict = fig_temp.to_dict()
    fig_dict = sanitize_for_json(fig_dict)
    
    return {"data": fig_dict['data'], "layout": fig_dict['layout']}


def _finite_point(values):
    """Return a finite XYZ point or None."""
    point = np.array(values, dtype=float)
    if point.shape != (3,) or not np.isfinite(point).all():
        return None
    return point


def _row_point_or_none(row, cols):
    """Extract one finite XYZ point from a row Series."""
    if cols is None:
        return None
    try:
        return _finite_point([row[col] for col in cols])
    except Exception:
        return None


def _point_trace(point):
    """Return 3D trace coordinate arrays for one source-coordinate point."""
    if point is None:
        return {"x": [], "y": [], "z": []}
    x, y, z = remap_point(point)
    return {"x": [x], "y": [y], "z": [z]}


def _line_trace(start, end):
    """Return 3D trace coordinate arrays for one source-coordinate line segment."""
    if start is None or end is None:
        return {"x": [], "y": [], "z": []}
    sx, sy, sz = remap_point(start)
    ex, ey, ez = remap_point(end)
    return {"x": [sx, ex], "y": [sy, ey], "z": [sz, ez]}


def _json_number_series(values):
    """Return finite floats and JSON nulls for a numeric sequence."""
    arr = np.asarray(values, dtype=float)
    return [float(value) if np.isfinite(value) else None for value in arr]


def _json_frame_series(values):
    """Return frame values as compact JSON numbers."""
    frames = []
    for value in np.asarray(values, dtype=float):
        if not np.isfinite(value):
            frames.append(None)
        elif float(value).is_integer():
            frames.append(int(value))
        else:
            frames.append(float(value))
    return frames


def _finite_xy_series(frames, values):
    """Return matching frame/value arrays with only finite plotted points."""
    frame_arr = np.asarray(frames, dtype=float)
    value_arr = np.asarray(values, dtype=float)
    finite = np.isfinite(frame_arr) & np.isfinite(value_arr)
    return {
        "frames": _json_frame_series(frame_arr[finite]),
        "values": _json_number_series(value_arr[finite]),
    }


def _valid_frame_runs(valid):
    """Return half-open ranges for contiguous valid frames."""
    runs = []
    start_idx = None
    for idx, is_valid in enumerate(valid):
        if is_valid and start_idx is None:
            start_idx = idx
        elif not is_valid and start_idx is not None:
            runs.append((start_idx, idx))
            start_idx = None
    if start_idx is not None:
        runs.append((start_idx, len(valid)))
    return runs


def _fixed_lowpass_filter(data):
    """Apply the same fixed 30 Hz zero-phase Butterworth used by the pipeline."""
    arr = np.asarray(data, dtype=float)
    if arr.size == 0:
        return arr.copy()

    try:
        from scipy.signal import butter, filtfilt
    except Exception:
        return arr.copy()

    nyquist = VIEWER_FPS / 2.0
    normalized_cutoff = VIEWER_LOWPASS_CUTOFF_HZ / nyquist
    b, a = butter(VIEWER_LOWPASS_ORDER, normalized_cutoff, btype="low", analog=False)
    padlen = 3 * (max(len(a), len(b)) - 1)
    if arr.shape[0] <= padlen:
        return arr.copy()
    return filtfilt(b, a, arr, axis=0)


def _vector_derivative(vector_xyz):
    """Differentiate a 3 x N vector series in source coordinates."""
    if vector_xyz.shape[1] <= 1:
        return np.zeros_like(vector_xyz, dtype=float)
    return np.gradient(vector_xyz, VIEWER_DT, axis=1)


def _filtered_vector_velocity(vector_xyz):
    """Differentiate and filter XYZ velocity, preserving gaps in ball tracking."""
    finite_frames = np.isfinite(vector_xyz).all(axis=0)
    if finite_frames.all():
        return _fixed_lowpass_filter(_vector_derivative(vector_xyz).T).T

    filtered = np.full_like(vector_xyz, np.nan, dtype=float)
    for start_idx, end_idx in _valid_frame_runs(finite_frames):
        segment = vector_xyz[:, start_idx:end_idx]
        filtered[:, start_idx:end_idx] = _fixed_lowpass_filter(
            _vector_derivative(segment).T
        ).T
    return filtered


def build_global_miss_metrics_series(pitch_df):
    """Build chart-ready global K80 missed-distance and miss-speed series."""
    if pitch_df.empty or "FRAME" not in pitch_df.columns:
        return {"frames": [], "missed_distance": [], "miss_speed": []}

    ball_cols = coord_columns(pitch_df.columns, "CENTER", "BALL")
    knob_cols = coord_columns(pitch_df.columns, "KNOB", "BAT_KNOB")
    top_cols = coord_columns(pitch_df.columns, "TOP", "BAT_TOP")
    if ball_cols is None or knob_cols is None or top_cols is None:
        return {"frames": [], "missed_distance": [], "miss_speed": []}

    frames = pd.to_numeric(pitch_df["FRAME"], errors="coerce").to_numpy(dtype=float)
    ball = pitch_df[list(ball_cols)].to_numpy(dtype=float).T
    knob = pitch_df[list(knob_cols)].to_numpy(dtype=float).T
    top = pitch_df[list(top_cols)].to_numpy(dtype=float).T

    # Kinatrax writes ball coordinates as all-zero when the ball is absent.
    # Treat those as missing so the diagnostic chart does not invent a ball.
    zero_ball = np.isclose(ball, 0.0).all(axis=0)
    ball[:, zero_ball] = np.nan

    ss_k80 = _sweet_spot_positions(knob, top)["ss_k80"]
    miss_global = ball - ss_k80
    finite_miss = np.isfinite(miss_global).all(axis=0)

    missed_distance = np.full(miss_global.shape[1], np.nan, dtype=float)
    missed_distance[finite_miss] = np.linalg.norm(miss_global[:, finite_miss], axis=0)

    velocity_global = _filtered_vector_velocity(miss_global)
    miss_speed = np.full(miss_global.shape[1], np.nan, dtype=float)
    finite_velocity = np.isfinite(velocity_global).all(axis=0)
    miss_speed[finite_velocity] = np.linalg.norm(
        velocity_global[:, finite_velocity],
        axis=0,
    )

    min_frame = None
    min_distance = None
    if np.isfinite(missed_distance).any():
        min_idx = int(np.nanargmin(missed_distance))
        min_frame_value = frames[min_idx]
        min_frame = int(min_frame_value) if float(min_frame_value).is_integer() else float(min_frame_value)
        min_distance = float(missed_distance[min_idx])

    return {
        "frames": _json_frame_series(frames),
        "missed_distance": _json_number_series(missed_distance),
        "missed_distance_visible": _finite_xy_series(frames, missed_distance),
        "miss_speed": _json_number_series(miss_speed),
        "miss_speed_visible": _finite_xy_series(frames, miss_speed),
        "min_frame": min_frame,
        "min_distance": min_distance,
        "speed_filter": {
            "type": "butterworth",
            "order": VIEWER_LOWPASS_ORDER,
            "cutoff_hz": VIEWER_LOWPASS_CUTOFF_HZ,
            "zero_phase": True,
        },
    }


def build_pitch_frame_cache(pitch_df, *, show_lcs=True):
    """
    Build a compact per-frame 3D payload for browser-side scrubbing/playback.

    The normal /api/frame path still builds full Plotly figures. This helper keeps
    only dynamic trace coordinates so the browser can call Plotly.restyle() while
    dragging the slider instead of asking Flask to rebuild the whole figure.
    """
    if pitch_df.empty:
        return {"frames": [], "frames_data": []}

    pitch_df = pitch_df.copy()
    pitch_df["FRAME"] = pd.to_numeric(pitch_df["FRAME"], errors="coerce")
    pitch_df = pitch_df.dropna(subset=["FRAME"]).sort_values("FRAME").reset_index(drop=True)
    if pitch_df.empty:
        return {"frames": [], "frames_data": []}

    joints = available_joints(pitch_df)
    ball_cols = coord_columns(pitch_df.columns, "CENTER", "BALL")
    knob_cols = coord_columns(pitch_df.columns, "KNOB", "BAT_KNOB")
    top_cols = coord_columns(pitch_df.columns, "TOP", "BAT_TOP")

    ss_k80 = None
    bat_80_origin = None
    r_bat80 = None
    if knob_cols is not None and top_cols is not None:
        try:
            knob_series = pitch_df[list(knob_cols)].to_numpy(dtype=float).T
            top_series = pitch_df[list(top_cols)].to_numpy(dtype=float).T
            ss_k80 = _sweet_spot_positions(knob_series, top_series)["ss_k80"].T
            if show_lcs:
                bat_80_origin, r_bat80 = _bat_80_lcs(
                    knob_series,
                    top_series,
                    handedness=infer_handedness(pitch_df),
                    axis_backfill_indices=event_frame_indices(pitch_df, ("BALL_MIN", "BAT_STOP")),
                )
        except Exception as exc:
            print(f"Warning: Could not precompute BAT_80 viewer data: {exc}")

    frames = []
    frames_data = []
    axis_length = 0.30

    for idx, row in pitch_df.iterrows():
        frame_value = float(row["FRAME"])
        frames.append(int(frame_value) if frame_value.is_integer() else frame_value)

        skeleton_x, skeleton_y, skeleton_z = [], [], []
        for a, b in BATTER_CONNECTIONS:
            if a not in joints or b not in joints:
                continue
            start = _row_point_or_none(row, (f"{a}_TX", f"{a}_TY", f"{a}_TZ"))
            end = _row_point_or_none(row, (f"{b}_TX", f"{b}_TY", f"{b}_TZ"))
            if start is None or end is None:
                continue
            sx, sy, sz = remap_point(start)
            ex, ey, ez = remap_point(end)
            skeleton_x += [sx, ex, None]
            skeleton_y += [sy, ey, None]
            skeleton_z += [sz, ez, None]

        joint_x, joint_y, joint_z, labels = [], [], [], []
        for joint in joints:
            point = _row_point_or_none(row, (f"{joint}_TX", f"{joint}_TY", f"{joint}_TZ"))
            if point is None:
                continue
            x_val, y_val, z_val = remap_point(point)
            joint_x.append(x_val)
            joint_y.append(y_val)
            joint_z.append(z_val)
            labels.append(joint)

        knob_point = _row_point_or_none(row, knob_cols)
        top_point = _row_point_or_none(row, top_cols)
        ball_point = _row_point_or_none(row, ball_cols)

        sweet_spot = None
        if ss_k80 is not None and idx < len(ss_k80):
            sweet_spot = _finite_point(ss_k80[idx])

        lcs_axes = {
            "r80_x": {"x": [], "y": [], "z": []},
            "r80_y": {"x": [], "y": [], "z": []},
            "r80_z": {"x": [], "y": [], "z": []},
        }
        if (
            show_lcs
            and bat_80_origin is not None
            and r_bat80 is not None
            and idx < bat_80_origin.shape[1]
            and idx < r_bat80.shape[0]
        ):
            origin = _finite_point(bat_80_origin[:, idx])
            if origin is not None:
                for key, axis_idx in (("r80_x", 0), ("r80_y", 1), ("r80_z", 2)):
                    axis = _finite_point(r_bat80[idx, :, axis_idx])
                    if axis is not None:
                        lcs_axes[key] = _line_trace(origin, origin + axis_length * axis)

        frames_data.append(
            {
                "frame": frames[-1],
                "skeleton": {"x": skeleton_x, "y": skeleton_y, "z": skeleton_z},
                "joints": {"x": joint_x, "y": joint_y, "z": joint_z, "text": labels},
                "bat": _line_trace(knob_point, top_point),
                "ball": _point_trace(ball_point),
                "sweet_spot": _point_trace(sweet_spot),
                **lcs_axes,
            }
        )

    return {
        "frames": frames,
        "frames_data": frames_data,
        "metrics": build_global_miss_metrics_series(pitch_df),
        "count": len(frames_data),
    }

# -----------------------------
# Flask app
# -----------------------------
def create_app(csv_path: str):
    """
    Create and configure the Flask application.

    csv_path may be:
      - An absolute or relative file path  → load that CSV at startup.
      - None (default)                     → start empty; data is loaded on
                                             demand when the user picks a game
                                             from the UI dropdown (/api/select-game).
    """
    app = Flask(__name__)

    # Seed app.config with empty state so all routes have safe defaults
    # even before a game is selected.
    _reset_app_data(app)

    if csv_path is None:
        # Lazy mode — no data loaded yet; UI will trigger /api/select-game
        print("📂 App started in lazy mode. Select a game from the UI to load data.")
        _register_routes(app)
        return app

    # Resolve relative paths against the canonical data directory
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(DATA_DIR, csv_path)

    app.config["CSV_PATH"] = csv_path

    # Load and validate data
    try:
        print(f"📁 Loading CSV: {csv_path}")
        
        # Define columns we actually need (much faster than loading all columns)
        # First, read just the header to get column names
        print("📊 Reading column headers...")
        header_df = pd.read_csv(csv_path, nrows=0)
        all_columns = header_df.columns.tolist()
        
        # Prefer canonical MLBAM GUIDs when pipeline outputs include them.
        id_col = 'MLBAM_GUID' if 'MLBAM_GUID' in all_columns else 'PITCH_ID'
        game_col = 'MLBAM_GAME_ID' if 'MLBAM_GAME_ID' in all_columns else None

        frame_source_col = None
        for candidate in ('FRAME', 'FRAME_NUMBER', 'TIMESTAMP'):
            if candidate in all_columns:
                frame_source_col = candidate
                break
        if frame_source_col is None:
            raise ValueError("CSV must contain FRAME, FRAME_NUMBER, or TIMESTAMP")

        # Identify required columns
        required_cols = [id_col, frame_source_col]
        position_cols = []
        
        # Ball position columns: raw motion or pipeline time-series output.
        ball_cols = coord_columns(all_columns, 'CENTER', 'BALL')
        if ball_cols is not None:
            position_cols.extend(ball_cols)
        
        # Bat position columns: raw motion or pipeline time-series output.
        for raw_prefix, output_prefix in [('TOP', 'BAT_TOP'), ('KNOB', 'BAT_KNOB')]:
            cols = coord_columns(all_columns, raw_prefix, output_prefix)
            if cols is not None:
                position_cols.extend(cols)
        
        # Skeleton joint columns (only load joints we visualize)
        joint_cols = []
        for joint in BATTER_JOINTS:
            for coord in ['TX', 'TY', 'TZ']:
                col = f"{joint}_{coord}"
                if col in all_columns:
                    joint_cols.append(col)
        
        # Outcome columns (needed for outcome determination)
        outcome_cols = []
        for col in ['TAKE', 'BAD', 'R', 'L', 'SWING', 'BALL_MIN', 'BAT_STOP']:
            if col in all_columns:
                outcome_cols.append(col)
        
        # Combine all needed columns
        needed_cols = required_cols + position_cols + joint_cols + outcome_cols
        if game_col is not None:
            needed_cols.append(game_col)
        needed_cols = [col for col in needed_cols if col in all_columns]
        
        print(f"📊 Loading {len(needed_cols)} of {len(all_columns)} columns...")
        print(f"📊 Skipping {len(all_columns) - len(needed_cols)} unnecessary columns")
        
        # Load only needed columns - MUCH faster!
        df = pd.read_csv(csv_path, usecols=needed_cols, low_memory=False)
        df = merge_nearby_hitting_report(df, csv_path, id_col, game_col=game_col)
        
        print(f"✅ Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
        
        # Ensure FRAME column exists and is numeric
        if 'FRAME' not in df.columns:
            if frame_source_col == 'TIMESTAMP':
                df['FRAME'] = (
                    df.sort_values([id_col, frame_source_col])
                    .groupby(id_col, dropna=False)
                    .cumcount()
                    .astype(int)
                )
            else:
                df['FRAME'] = df[frame_source_col]

        if 'FRAME' not in df.columns:
            raise ValueError("CSV must contain a FRAME column")
        
        df['FRAME'] = pd.to_numeric(df['FRAME'], errors='coerce')
        df = df.dropna(subset=['FRAME'])
        
        # Filter out TAKE pitches (skip pitches where TAKE column equals "TAKE")
        if 'TAKE' in df.columns:
            print("📊 Filtering out TAKE pitches...")
            initial_count = len(df)
            initial_pitches = df[id_col].nunique()
            
            # Normalize TAKE column to uppercase string for comparison
            df['TAKE'] = df['TAKE'].astype(str).str.strip().str.upper()
            
            # Get list of pitch identifiers that have TAKE = "TAKE"
            take_pitch_ids = df[df['TAKE'] == 'TAKE'][id_col].unique()
            
            # Remove all rows for TAKE pitches
            df = df[df[id_col].isin(take_pitch_ids) == False]
            
            filtered_count = len(df)
            filtered_pitches = df[id_col].nunique()
            
            print(f"📊 Removed {initial_count - filtered_count} rows from {len(take_pitch_ids)} TAKE pitches")
            print(f"📊 Remaining: {filtered_pitches} pitches ({initial_pitches - len(take_pitch_ids)} removed)")
        else:
            print("📊 No TAKE column found - skipping TAKE filtering")
        
        # Sort and index by canonical pitch identifier for faster lookups
        print("📊 Sorting and indexing data...")
        df = df.sort_values([id_col, 'FRAME']).reset_index(drop=True)
        
        # Create index for faster pitch lookups
        df_indexed = df.set_index(id_col, drop=False)
        
        # Get unique canonical pitch IDs efficiently
        pitch_ids = sorted(df[id_col].unique().tolist())
        print(f"⚾ Found {len(pitch_ids)} unique {id_col} values")
        pitch_labels = {
            pid: pitch_label(pid, group)
            for pid, group in df.groupby(id_col, dropna=False)
        }
        
        # Get frame ranges per pitch efficiently using groupby
        print("📊 Computing frame ranges per pitch...")
        frame_ranges = df.groupby(id_col)['FRAME'].agg(['min', 'max']).to_dict('index')
        pitch_ranges = {pid: {'min': int(r['min']), 'max': int(r['max'])} 
                       for pid, r in frame_ranges.items()}
        
        # Compute global frame range across ALL pitches (for unrestricted frame access)
        global_frame_min = int(df['FRAME'].min())
        global_frame_max = int(df['FRAME'].max())
        print(f"📊 Global frame range: {global_frame_min} to {global_frame_max}")
        
        print("✅ Data loading complete!")
        
        # Build cascading game → GUID map for the two-dropdown UI.
        # Each key is a game ID string; value is an ordered list of
        # (guid, display_label) tuples for that game's trials.
        game_guid_map = {}
        if game_col and game_col in df.columns:
            for gid, group in df.groupby(game_col, sort=True):
                gid_str = str(gid)
                guids = sorted(group[id_col].unique().tolist())
                game_guid_map[gid_str] = [
                    (guid, pitch_labels.get(guid, guid)) for guid in guids
                ]
        else:
            # No game column — treat all trials as one synthetic game
            game_guid_map["all"] = [
                (guid, pitch_labels.get(guid, guid)) for guid in pitch_ids
            ]
        print(f"🗂  Game→GUID map built: {len(game_guid_map)} games, "
              f"{sum(len(v) for v in game_guid_map.values())} total trials")

        # Store indexed DataFrame and frame ranges in app config for use in routes
        app.config['DF'] = df
        app.config['DF_INDEXED'] = df_indexed
        app.config['GLOBAL_FRAME_MIN'] = global_frame_min
        app.config['GLOBAL_FRAME_MAX'] = global_frame_max
        app.config['PITCH_IDS'] = pitch_ids
        app.config['PITCH_LABELS'] = pitch_labels
        app.config['PITCH_RANGES'] = pitch_ranges
        app.config['ID_COL'] = id_col
        app.config['GAME_COL'] = game_col
        app.config['GAME_GUID_MAP'] = game_guid_map
        app.config['PITCH_DATA_CACHE'] = {}

    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        traceback.print_exc()
        raise

    _register_routes(app)
    return app


# ---------------------------------------------------------------------------
# _reset_app_data  — initialise / clear all data keys in app.config
# ---------------------------------------------------------------------------
def _reset_app_data(app):
    """Set safe empty defaults for every data key so routes never KeyError."""
    app.config.update({
        'CSV_PATH':        None,
        'DF':              pd.DataFrame(),
        'DF_INDEXED':      pd.DataFrame(),
        'GLOBAL_FRAME_MIN': 0,
        'GLOBAL_FRAME_MAX': 0,
        'PITCH_IDS':       [],
        'PITCH_LABELS':    {},
        'PITCH_RANGES':    {},
        'ID_COL':          'MLBAM_GUID',
        'GAME_COL':        'MLBAM_GAME_ID',
        'GAME_GUID_MAP':   {},
        'PITCH_DATA_CACHE': {},
    })


# ---------------------------------------------------------------------------
# load_game_into_app  — load one game's CSV and update app.config in-place
# ---------------------------------------------------------------------------
def load_game_into_app(app, game_id: str) -> dict:
    """
    Load motion_sequence.csv for game_id, process it exactly as create_app does,
    and hot-swap all data keys in app.config.

    Returns a summary dict suitable for sending back as JSON to the frontend.
    Raises FileNotFoundError / ValueError on bad input.
    """
    game_dir = os.path.join(DATA_DIR, str(game_id))
    csv_path = motion_sequence_path(game_dir)
    if csv_path is None:
        raise FileNotFoundError(
            f"No motion_sequence CSV found for game '{game_id}' in {game_dir}"
        )

    print(f"📁 Loading game {game_id}: {csv_path}")

    header_df = pd.read_csv(csv_path, nrows=0)
    all_columns = header_df.columns.tolist()

    id_col   = 'MLBAM_GUID'   if 'MLBAM_GUID'   in all_columns else 'PITCH_ID'
    game_col = 'MLBAM_GAME_ID' if 'MLBAM_GAME_ID' in all_columns else None

    frame_source_col = next(
        (c for c in ('FRAME', 'FRAME_NUMBER', 'TIMESTAMP') if c in all_columns),
        None
    )
    if frame_source_col is None:
        raise ValueError("CSV must contain FRAME, FRAME_NUMBER, or TIMESTAMP")

    required_cols  = [id_col, frame_source_col]
    position_cols  = []
    ball_cols_     = coord_columns(all_columns, 'CENTER', 'BALL')
    if ball_cols_:
        position_cols.extend(ball_cols_)
    for raw_pfx, out_pfx in [('TOP', 'BAT_TOP'), ('KNOB', 'BAT_KNOB')]:
        c = coord_columns(all_columns, raw_pfx, out_pfx)
        if c:
            position_cols.extend(c)
    joint_cols = [
        f"{j}_{ax}" for j in BATTER_JOINTS
        for ax in ('TX', 'TY', 'TZ')
        if f"{j}_{ax}" in all_columns
    ]
    outcome_cols = [c for c in ('TAKE', 'BAD', 'R', 'L', 'SWING', 'BALL_MIN', 'BAT_STOP')
                    if c in all_columns]

    needed_cols = list(dict.fromkeys(
        required_cols + position_cols + joint_cols + outcome_cols
        + ([game_col] if game_col else [])
    ))
    needed_cols = [c for c in needed_cols if c in all_columns]

    df = pd.read_csv(csv_path, usecols=needed_cols, low_memory=False)
    df = merge_nearby_hitting_report(df, csv_path, id_col, game_col=game_col)

    # FRAME normalisation
    if 'FRAME' not in df.columns:
        df['FRAME'] = (
            df.sort_values([id_col, frame_source_col])
            .groupby(id_col, dropna=False).cumcount().astype(int)
            if frame_source_col == 'TIMESTAMP'
            else df[frame_source_col]
        )
    df['FRAME'] = pd.to_numeric(df['FRAME'], errors='coerce')
    df = df.dropna(subset=['FRAME'])

    # Filter TAKE pitches
    if 'TAKE' in df.columns:
        df['TAKE'] = df['TAKE'].astype(str).str.strip().str.upper()
        take_ids = df[df['TAKE'] == 'TAKE'][id_col].unique()
        df = df[~df[id_col].isin(take_ids)]

    df = df.sort_values([id_col, 'FRAME']).reset_index(drop=True)
    df_indexed = df.set_index(id_col, drop=False)

    pitch_ids = sorted(df[id_col].unique().tolist())
    pitch_labels = {
        pid: pitch_label(pid, grp)
        for pid, grp in df.groupby(id_col, dropna=False)
    }
    frame_ranges  = df.groupby(id_col)['FRAME'].agg(['min', 'max']).to_dict('index')
    pitch_ranges  = {pid: {'min': int(r['min']), 'max': int(r['max'])}
                     for pid, r in frame_ranges.items()}
    global_frame_min = int(df['FRAME'].min())
    global_frame_max = int(df['FRAME'].max())

    # Build game→GUID map (single game only)
    guids_for_game = [(g, pitch_labels.get(g, g)) for g in pitch_ids]
    game_guid_map  = {str(game_id): guids_for_game}

    # Hot-swap all data in app.config
    app.config.update({
        'CSV_PATH':         csv_path,
        'DF':               df,
        'DF_INDEXED':       df_indexed,
        'GLOBAL_FRAME_MIN': global_frame_min,
        'GLOBAL_FRAME_MAX': global_frame_max,
        'PITCH_IDS':        pitch_ids,
        'PITCH_LABELS':     pitch_labels,
        'PITCH_RANGES':     pitch_ranges,
        'ID_COL':           id_col,
        'GAME_COL':         game_col,
        'GAME_GUID_MAP':    game_guid_map,
        'PITCH_DATA_CACHE': {},
    })

    print(f"✅ Game {game_id} loaded: {len(pitch_ids)} trials, "
          f"frames {global_frame_min}–{global_frame_max}")

    return {
        "game_id":         str(game_id),
        "trial_count":     len(pitch_ids),
        "global_frame_min": global_frame_min,
        "global_frame_max": global_frame_max,
        "guids": [{"guid": g, "label": lbl} for g, lbl in guids_for_game],
    }


# ---------------------------------------------------------------------------
# _register_routes — define all Flask routes (called once by create_app)
# ---------------------------------------------------------------------------
def _register_routes(app):
    """Attach all URL routes to the Flask app instance."""

    @app.route("/api/games")
    def api_games():
        """
        Return all available game IDs discovered from the data directory.
        Reads from the filesystem so it works in lazy mode (before any game
        is selected) as well as when a game is already loaded.
        Response: [{id}]
        """
        game_dirs = discover_game_dirs(DATA_DIR)
        return jsonify([
            {"id": os.path.basename(d)}
            for d in game_dirs
        ])

    @app.route("/api/guids/<game_id>")
    def api_guids(game_id: str):
        """
        Return all GUIDs (trials) for the currently-loaded game.
        Only valid after /api/select-game/<id> has been called.
        Response: [{guid, label}]
        """
        game_map = app.config['GAME_GUID_MAP']
        guids = game_map.get(str(game_id), [])
        if not guids:
            return jsonify({"error": f"Game '{game_id}' not found or not yet loaded"}), 404
        return jsonify([{"guid": g, "label": lbl} for g, lbl in guids])

    @app.route("/api/select-game/<game_id>", methods=["POST"])
    def api_select_game(game_id: str):
        """
        Load a single game's motion_sequence CSV into memory, replacing any
        previously loaded game. Called by the frontend when the user picks a
        game from the Game ID dropdown.

        Response: {game_id, trial_count, guids: [{guid, label}]}
        """
        try:
            summary = load_game_into_app(app, game_id)
            return jsonify(summary)
        except FileNotFoundError as exc:
            return jsonify({"error": str(exc)}), 404
        except Exception as exc:
            traceback.print_exc()
            return jsonify({"error": str(exc)}), 500

    def requested_pitch_identifier() -> str | None:
        """Accept the new guid parameter while keeping pitch_id compatibility."""
        return request.args.get("guid", type=str) or request.args.get("pitch_id", type=str)

    def requested_game_identifier() -> str | None:
        """Return optional MLBAM_GAME_ID filter."""
        return request.args.get("game_id", type=str)
    
    @app.route("/")
    def index():
        """Main page with hitting viewer."""
        pitch_ids    = app.config['PITCH_IDS']
        pitch_labels = app.config['PITCH_LABELS']
        pitch_ranges = app.config['PITCH_RANGES']

        # In lazy mode no game has been loaded yet — serve an empty shell.
        # The user will pick a game from the dropdown which triggers
        # /api/select-game and then populates the GUID dropdown.
        if not pitch_ids:
            return render_template(
                "hitting_viewer.html",
                plot_payload={"data": [], "layout": {}},
                pitch_ids=[],
                pitch_labels={},
                default_pitch_id=None,
                default_frame=0,
                global_frame_min=0,
                global_frame_max=0,
            )

        # A game is already loaded — generate the initial plot for the first trial.
        df        = app.config['DF']
        df_indexed = app.config['DF_INDEXED']
        default_pitch_id = pitch_ids[0]
        default_frame    = pitch_ranges[default_pitch_id]['min']

        try:
            payload = figure_payload_for_frame(df, df_indexed, default_frame, pitch_id=default_pitch_id)
            print(f"✅ Generated initial plot payload with {len(payload['data'])} traces")
        except Exception as e:
            print(f"❌ Error generating initial plot: {e}")
            traceback.print_exc()
            payload = {"data": [], "layout": {}}

        return render_template(
            "hitting_viewer.html",
            plot_payload=payload,
            pitch_ids=pitch_ids,
            pitch_labels=pitch_labels,
            default_pitch_id=default_pitch_id,
            default_frame=default_frame,
            global_frame_min=app.config['GLOBAL_FRAME_MIN'],
            global_frame_max=app.config['GLOBAL_FRAME_MAX'],
        )
    
    def calculate_missed_distance(pitch_df, frame_idx=None):
        """
        Calculate missed distance for a pitch.
        
        Args:
            pitch_df: DataFrame filtered to a specific pitch
            frame_idx: Optional frame index to calculate distance at (if None, calculates minimum)
        
        Returns:
            dict with missed distance info: {
                'current_frame_dist': distance at current frame (m),
                'min_distance': minimum distance for pitch (m),
                'min_distance_frame': frame where minimum occurs,
                'miss_vec': miss vector at minimum (x, y, z),
                'outcome': outcome string (hit, miss, CHECK_SWING, TAKE, unknown)
            }
        """
        try:
            # Calculate outcome first
            outcome = "unknown"
            try:
                # Check if outcome columns exist
                has_take = 'TAKE' in pitch_df.columns
                has_bad = 'BAD' in pitch_df.columns
                has_r = 'R' in pitch_df.columns
                
                if not (has_take or has_bad or has_r):
                    print(f"Warning: No outcome columns (TAKE, BAD, R) found in pitch data")
                    outcome = "unknown"
                else:
                    outcome, skip = _infer_outcome_with_take_skip(pitch_df)
                    # Normalize outcome for display
                    if outcome == "hit":
                        outcome = "Contact"
                    elif outcome == "miss":
                        outcome = "Swinging Strike"
                    elif outcome == "CHECK_SWING":
                        outcome = "Check Swing"
                    elif outcome == "TAKE":
                        outcome = "Take"
                    # else keep as "unknown"
            except Exception as e:
                print(f"Warning: Could not determine outcome: {e}")
                traceback.print_exc()
                outcome = "unknown"
            # Check for required columns
            ball_cols = coord_columns(pitch_df.columns, 'CENTER', 'BALL')
            knob_cols = coord_columns(pitch_df.columns, 'KNOB', 'BAT_KNOB')
            top_cols = coord_columns(pitch_df.columns, 'TOP', 'BAT_TOP')
            if ball_cols is None or knob_cols is None or top_cols is None:
                return None
            
            # Extract trajectories (before coordinate remapping)
            ball = pitch_df[list(ball_cols)].to_numpy(dtype=float).T  # (3, N)
            knob = pitch_df[list(knob_cols)].to_numpy(dtype=float).T
            top = pitch_df[list(top_cols)].to_numpy(dtype=float).T
            
            # Calculate sweet spot positions (K80 at 80% of bat length)
            ss_positions = _sweet_spot_positions(knob, top)
            ss_k80 = ss_positions["ss_k80"]  # (3, N) - sweet spot at 80% bat length
            
            # Calculate missed distance using the function
            t_min, min_dist, miss_vec = _closest_distance_and_vector(
                ball, ss_k80, pitch_id=None, frame=None, save_plot=False
            )
            
            # Get current frame distance if frame_idx provided
            current_frame_dist = None
            if frame_idx is not None:
                # Find closest frame in pitch_df
                frames = pitch_df['FRAME'].values
                if len(frames) > 0:
                    frame_idx_in_array = np.argmin(np.abs(frames - frame_idx))
                    if frame_idx_in_array < ball.shape[1]:
                        miss_vec_current = ball[:, frame_idx_in_array] - ss_k80[:, frame_idx_in_array]
                        current_frame_dist = float(np.linalg.norm(miss_vec_current))
            
            # Get frame number where minimum occurs
            min_distance_frame = None
            if t_min < len(pitch_df):
                min_distance_frame = int(pitch_df.iloc[t_min]['FRAME'])
            
            return {
                'current_frame_dist': current_frame_dist,
                'min_distance': float(min_dist),
                'min_distance_frame': min_distance_frame,
                'miss_vec': {
                    'x': float(miss_vec[0]),
                    'y': float(miss_vec[1]),
                    'z': float(miss_vec[2])
                },
                'outcome': outcome
            }
        except Exception as e:
            print(f"Warning: Could not calculate missed distance: {e}")
            traceback.print_exc()
            return None
    
    @app.get("/api/frame")
    def api_frame():
        """API endpoint to get data for a specific frame."""
        try:
            # Get DataFrames from app config
            df = app.config['DF']
            df_indexed = app.config['DF_INDEXED']
            
            frame = request.args.get("frame", type=float)
            pitch_id = requested_pitch_identifier()
            view = request.args.get("view", default="3D", type=str)
            show_plate = request.args.get("show_plate", default="true", type=str).lower() == "true"
            show_lcs = request.args.get("show_lcs", default="true", type=str).lower() == "true"
            include_md = request.args.get("include_md", default="true", type=str).lower() == "true"
            
            if frame is None:
                return jsonify({"error": "frame parameter required"}), 400
            
            game_id = requested_game_identifier()
            payload = figure_payload_for_frame(
                df,
                df_indexed,
                frame,
                pitch_id=pitch_id,
                game_id=game_id,
                show_plate=show_plate,
                show_lcs=show_lcs,
                view=view,
            )
            
            # Calculate missed distance if pitch_id is provided
            missed_distance_info = None
            if pitch_id and include_md:
                try:
                    pitch_df = df_indexed.loc[[pitch_id]].copy() if pitch_id in df_indexed.index else pd.DataFrame()
                    if game_id is not None and "MLBAM_GAME_ID" in pitch_df.columns:
                        pitch_df = pitch_df[pitch_df["MLBAM_GAME_ID"].astype(str) == str(game_id)]
                    if len(pitch_df) > 0:
                        missed_distance_info = calculate_missed_distance(pitch_df, frame_idx=frame)
                except Exception as e:
                    print(f"Warning: Failed to calculate missed distance: {e}")
            
            payload = sanitize_for_json(payload)
            if missed_distance_info:
                payload['missed_distance'] = missed_distance_info
            
            return jsonify(payload)
        except Exception as e:
            print(f"Error in /api/frame: {e}")
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500

    @app.get("/api/pitch-data")
    def api_pitch_data():
        """Return compact dynamic trace data for one pitch for smooth playback."""
        try:
            df_indexed = app.config['DF_INDEXED']
            pitch_id = requested_pitch_identifier()
            game_id = requested_game_identifier()
            show_lcs = request.args.get("show_lcs", default="true", type=str).lower() == "true"

            if pitch_id is None:
                return jsonify({"error": "guid parameter required"}), 400
            if pitch_id not in df_indexed.index:
                return jsonify({"error": f"Pitch identifier {pitch_id} not found"}), 404

            cache = app.config['PITCH_DATA_CACHE']
            cache_key = (str(pitch_id), str(game_id or ""), bool(show_lcs))
            if cache_key in cache:
                return jsonify(cache[cache_key])

            pitch_df = df_indexed.loc[[pitch_id]].copy()
            if game_id is not None and "MLBAM_GAME_ID" in pitch_df.columns:
                pitch_df = pitch_df[pitch_df["MLBAM_GAME_ID"].astype(str) == str(game_id)]
                if pitch_df.empty:
                    return jsonify({"error": f"Pitch identifier {pitch_id} not found for game_id {game_id}"}), 404

            payload = build_pitch_frame_cache(pitch_df, show_lcs=show_lcs)
            payload = sanitize_for_json(payload)
            cache[cache_key] = payload
            return jsonify(payload)
        except Exception as e:
            print(f"Error in /api/pitch-data: {e}")
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500
    
    @app.get("/api/pitch-frames")
    def api_pitch_frames():
        """Return min/max frames for a given MLBAM_GUID and optional game_id."""
        try:
            pitch_ranges = app.config['PITCH_RANGES']
            df_indexed = app.config['DF_INDEXED']
            pitch_id = requested_pitch_identifier()
            game_id = requested_game_identifier()
            if pitch_id is None:
                return jsonify({"error": "guid parameter required"}), 400

            if pitch_id not in pitch_ranges:
                return jsonify({"error": f"Pitch identifier {pitch_id} not found"}), 404

            if game_id is not None and "MLBAM_GAME_ID" in df_indexed.columns:
                pitch_df = df_indexed.loc[[pitch_id]].copy() if pitch_id in df_indexed.index else pd.DataFrame()
                pitch_df = pitch_df[pitch_df["MLBAM_GAME_ID"].astype(str) == str(game_id)]
                if pitch_df.empty:
                    return jsonify({"error": f"Pitch identifier {pitch_id} not found for game_id {game_id}"}), 404
                return jsonify(
                    {
                        "min": int(pitch_df["FRAME"].min()),
                        "max": int(pitch_df["FRAME"].max()),
                    }
                )

            return jsonify(pitch_ranges[pitch_id])
        except Exception as e:
            print(f"Error in /api/pitch-frames: {e}")
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500
    
    @app.get("/api/search-pitch")
    def api_search_pitch():
        """Search for MLBAM_GUIDs matching a query string."""
        try:
            df = app.config['DF']
            id_col = app.config['ID_COL']
            pitch_labels = app.config.get('PITCH_LABELS', {})
            query = request.args.get("q", type=str, default="")
            game_id = requested_game_identifier()
            if game_id is not None and "MLBAM_GAME_ID" in df.columns:
                df = df[df["MLBAM_GAME_ID"].astype(str) == str(game_id)]
            pitch_ids = sorted(df[id_col].dropna().unique().tolist())

            def pitch_option(pid):
                return {"id": pid, "label": pitch_labels.get(pid, str(pid))}

            if not query:
                selected = pitch_ids[:50]
                return jsonify(
                    {
                        "pitch_ids": selected,
                        "pitch_options": [pitch_option(pid) for pid in selected],
                    }
                )
            
            # Simple substring search (case-insensitive)
            query_lower = query.lower()
            matches = [
                pid
                for pid in pitch_ids
                if query_lower in str(pid).lower()
                or query_lower in pitch_labels.get(pid, "").lower()
            ][:50]
            return jsonify(
                {
                    "pitch_ids": matches,
                    "pitch_options": [pitch_option(pid) for pid in matches],
                }
            )
        except Exception as e:
            print(f"Error in /api/search-pitch: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.get("/api/export-animation")
    def api_export_animation():
        """Export animation as standalone HTML file."""
        try:
            from hitting_viewer_export import generate_animation_html
            import tempfile
            
            pitch_ranges = app.config['PITCH_RANGES']
            pitch_id = requested_pitch_identifier()
            view = request.args.get("view", default="3D", type=str)
            
            if pitch_id is None:
                return jsonify({"error": "guid parameter required"}), 400
            
            if pitch_id not in pitch_ranges:
                return jsonify({"error": f"Pitch identifier {pitch_id} not found"}), 404
            
            # Get DataFrames from app config
            df = app.config['DF']
            df_indexed = app.config['DF_INDEXED']
            
            # Create temporary file for HTML export
            temp_dir = tempfile.gettempdir()
            output_filename = f"hitting_animation_{pitch_id}_{view}.html"
            output_path = os.path.join(temp_dir, output_filename)
            
            # Generate animation HTML
            print(f"📹 Exporting animation for pitch identifier {pitch_id}...")
            generate_animation_html(
                df, 
                df_indexed, 
                pitch_id, 
                output_path, 
                view=view,
                figure_payload_func=figure_payload_for_frame
            )
            
            # Send file for download
            return send_file(
                output_path,
                mimetype='text/html',
                as_attachment=True,
                download_name=output_filename
            )
        except Exception as e:
            print(f"Error in /api/export-animation: {e}")
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500
    
    return app

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Flask Hitting Data Viewer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Data directory: {DATA_DIR}

Usage examples:
  python hitting_viewer_app.py                  # load all discovered games
  python hitting_viewer_app.py 823871           # load single game by ID
  python hitting_viewer_app.py all              # explicit all-games mode
  python hitting_viewer_app.py /path/to/my.csv  # direct CSV path (legacy)
        """,
    )
    parser.add_argument(
        "game",
        nargs="?",
        default=None,
        help=(
            "Game ID (e.g. 823871), 'all', or a direct path to a CSV file. "
            "Omit to auto-load all discovered games under data/."
        ),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    args = parser.parse_args()

    print("🚀 Starting Hitting Data Viewer")
    print(f"📂 Data directory: {DATA_DIR}")

    # Discover available games and print a summary before loading
    available = discover_game_dirs(DATA_DIR)
    print(f"🗂  Available games: {[os.path.basename(d) for d in available]}")

    # Resolve the supplied argument to a concrete CSV path
    try:
        csv_path = resolve_csv_path(args.game, DATA_DIR)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"\n❌ {exc}")
        sys.exit(1)

    app = create_app(csv_path)

    print(f"🌐 Server starting at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)
