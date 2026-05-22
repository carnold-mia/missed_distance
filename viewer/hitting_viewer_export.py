# hitting_viewer_export.py
# -------------------------------------------------------------
# Export functionality for hitting viewer animations
# Generates standalone HTML files with Plotly animations
# -------------------------------------------------------------

import plotly.graph_objs as go
import plotly.offline as pyo
import numpy as np
import pandas as pd
from pathlib import Path
import json
import os
import sys

# Add current directory to path to avoid circular imports
# We'll import the function directly when needed


def generate_animation_html(df, df_indexed, pitch_id, output_path, view='3D', show_plate=True, figure_payload_func=None):
    """
    Generate a standalone HTML file with Plotly animation for a complete pitch.
    
    Args:
        df: Full DataFrame with hitting data
        df_indexed: DataFrame indexed by MLBAM_GUID or legacy PITCH_ID for fast lookups
        pitch_id: Pitch identifier to export
        output_path: Path where HTML file should be saved
        view: View type ('3D', 'Front', 'Side', 'Top')
        show_plate: Whether to show home plate
    
    Returns:
        Path to generated HTML file
    """
    # Filter to specific pitch
    pitch_df = df_indexed.loc[[pitch_id]].copy() if pitch_id in df_indexed.index else pd.DataFrame()
    if len(pitch_df) == 0:
        raise ValueError(f"No data found for pitch identifier {pitch_id}")
    
    # Ensure FRAME column is numeric and sort
    pitch_df['FRAME'] = pd.to_numeric(pitch_df['FRAME'], errors='coerce')
    pitch_df = pitch_df.sort_values('FRAME').reset_index(drop=True)
    pitch_df = pitch_df.dropna(subset=['FRAME'])
    
    if len(pitch_df) == 0:
        raise ValueError(f"No valid frames found for pitch identifier {pitch_id}")
    
    # Get frame range for this pitch
    frames = pitch_df['FRAME'].values
    frame_min = int(frames.min())
    frame_max = int(frames.max())
    
    print(f"📹 Generating animation for pitch identifier {pitch_id}")
    print(f"📊 Frame range: {frame_min} to {frame_max} ({len(frames)} frames)")
    
    # Import functions dynamically to avoid circular imports
    if figure_payload_func is None:
        # Import here to avoid circular import issues
        from hitting_viewer_app import figure_payload_for_frame, available_joints, compute_axis_ranges
        figure_payload_func = figure_payload_for_frame
        # Import helper functions
        _available_joints = available_joints
        _compute_axis_ranges = compute_axis_ranges
    else:
        # If function is passed, import helpers separately
        from hitting_viewer_app import available_joints as _available_joints, compute_axis_ranges as _compute_axis_ranges
    
    # Get available joints
    joints = _available_joints(pitch_df)
    
    # Compute axis ranges
    xr, yr, zr = _compute_axis_ranges(pitch_df, joints)
    xr = [float(xr[0]), float(xr[1])]
    yr = [float(yr[0]), float(yr[1])]
    zr = [float(zr[0]), float(zr[1])]
    
    # Check if 2D view
    is_2d = view in ['Front', 'Side', 'Top']
    
    # Create base figure with first frame
    first_frame_data = figure_payload_func(df, df_indexed, frame_min, pitch_id=pitch_id, show_plate=show_plate, view=view)
    
    # Create Plotly figure
    fig = go.Figure(data=first_frame_data['data'], layout=first_frame_data['layout'])
    
    # Generate frames for animation
    animation_frames = []
    print("🎬 Generating animation frames...")
    
    for idx, frame_num in enumerate(frames):
        if idx % 10 == 0:
            print(f"  Processing frame {idx+1}/{len(frames)} (frame {frame_num})")
        
        # Get data for this frame
        frame_data = figure_payload_func(df, df_indexed, frame_num, pitch_id=pitch_id, show_plate=show_plate, view=view)
        
        # Create frame trace data
        frame_traces = []
        for trace in frame_data['data']:
            frame_traces.append(trace)
        
        # Create frame
        animation_frames.append(
            go.Frame(
                data=frame_traces,
                name=str(frame_num)
            )
        )
    
    # Add frames to figure
    fig.frames = animation_frames
    
    # Add animation controls
    # Play/Pause buttons
    fig.update_layout(
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 50, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 30}
                            }
                        ]
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0}
                            }
                        ]
                    }
                ],
                "x": 0.1,
                "xanchor": "left",
                "y": 0,
                "yanchor": "top"
            }
        ],
        # Slider for frame navigation (separate from updatemenus)
        sliders=[
            {
                "active": 0,
                "steps": [
                    {
                        "args": [
                            [str(f)],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 0}
                            }
                        ],
                        "label": str(f),
                        "method": "animate"
                    }
                    for f in frames[:100]  # Limit slider steps for performance
                ],
                "x": 0.1,
                "y": 0,
                "len": 0.9,
                "xanchor": "left",
                "yanchor": "top",
                "pad": {"b": 10, "t": 50},
                "currentvalue": {
                    "visible": True,
                    "prefix": "Frame: ",
                    "xanchor": "right"
                }
            }
        ]
    )
    
    # Add title with pitch identifier
    fig.update_layout(
        title={
            "text": f"Hitting Animation - ID: {pitch_id}",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 20}
        },
        height=800
    )
    
    # Generate HTML
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"💾 Saving HTML to {output_path}")
    pyo.plot(fig, filename=str(output_path), auto_open=False, config={'displayModeBar': True})
    
    print(f"✅ Animation export complete!")
    return output_path
