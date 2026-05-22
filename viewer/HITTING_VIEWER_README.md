# Hitting Data Viewer

Interactive Flask web application for visualizing hitting biomechanics data from KT Ocean CSV files. Displays batter skeleton, bat, and ball positions in 3D with frame-by-frame playback controls.

## Features

* **MLBAM_GUID Selection**: Search and select specific pitches by `MLBAM_GUID`, with `MLBAM_GAME_ID` available for game context when present
* **3D Visualization**: Interactive 3D plot showing:
  * Batter skeleton (joints and connections)
  * Bat position (knob to top)
  * Ball position
  * Home plate reference
* **Multiple Views**: Switch between 3D, Front, Side, and Top views
* **Frame-by-Frame Playback**:
  * Play/pause controls
  * Step forward/backward
  * Adjustable playback speed (0.5x, 1x, 1.5x, 2x)
  * Frame slider for quick navigation
* **Real-time Updates**: Smooth frame transitions with camera position preservation

## Requirements

```bash
pip install flask pandas numpy plotly
```

## Usage

### Basic Usage

```bash
python hitting_viewer_app.py data/kt_ocean_multi_game.csv
```

### Advanced Options

```bash
python hitting_viewer_app.py data/kt_ocean_multi_game.csv --host 0.0.0.0 --port 8080 --debug
```

### Arguments

* `csv`: Path to KT Ocean CSV file (required)
* `--host`: Host address (default: 127.0.0.1)
* `--port`: Port number (default: 5000)
* `--debug`: Enable Flask debug mode

## Data Format

The CSV file must contain the following columns:

### Required Columns

* `MLBAM_GUID`: Canonical identifier for each pitch
* `MLBAM_GAME_ID`: Canonical game identifier when available
* `FRAME`: Frame number (numeric, sequential per pitch)

### Position Columns

* **Ball**: `CENTER_TX`, `CENTER_TY`, `CENTER_TZ`
* **Bat**: `TOP_TX`, `TOP_TY`, `TOP_TZ` (bat top) and `KNOB_TX`, `KNOB_TY`, `KNOB_TZ` (bat knob)

### Skeleton Joints (Optional)

The following joint positions are visualized if available:

* `HIPS_TX/Y/Z`, `SPINE_TX/Y/Z`, `NECK_TX/Y/Z`, `HEAD_TX/Y/Z`
* `LEFTUPPERARM_TX/Y/Z`, `LEFTFOREARM_TX/Y/Z`, `LEFTHAND_TX/Y/Z`
* `RIGHTUPPERARM_TX/Y/Z`, `RIGHTFOREARM_TX/Y/Z`, `RIGHTHAND_TX/Y/Z`
* `LEFTTHIGH_TX/Y/Z`, `LEFTSHIN_TX/Y/Z`, `LEFTANKLE_TX/Y/Z`, `LEFTFOOT_TX/Y/Z`
* `RIGHTTHIGH_TX/Y/Z`, `RIGHTSHIN_TX/Y/Z`, `RIGHTANKLE_TX/Y/Z`, `RIGHTFOOT_TX/Y/Z`

## User Interface

### Pitch Selection


1. **Search**: Type in the search box to filter `MLBAM_GUID` values
2. **Select**: Choose from dropdown list of all available pitch identifiers
3. Frame range automatically updates when a pitch is selected

### Visualization Controls

* **View Buttons**: Switch between Front, Side, Top, and 3D views
* **Frame Slider**: Drag to jump to any frame
* **Playback Controls**:
  * ⏮️ Rewind: Jump to start of pitch
  * ⏪ Previous: Step back one frame
  * ▶️ Play/Pause: Toggle playback
  * ⏩ Next: Step forward one frame
  * ⏭️ Forward: Jump to end of pitch
  * Speed: Adjust playback speed (0.5x - 2x)

### Keyboard Shortcuts

* `Space`: Play/pause
* `←`: Previous frame
* `→`: Next frame

### Display Options

* Show/Hide Home Plate
* Show/Hide Skeleton
* Show/Hide Bat

## API Endpoints

### `GET /api/frame`

Get visualization data for a specific frame.

**Parameters**:

* `frame` (required): Frame number
* `guid` (optional): `MLBAM_GUID` to filter data
* `game_id` (optional): `MLBAM_GAME_ID` context when present in the CSV
* `view` (optional): View type ('3D', 'Front', 'Side', 'Top')

**Response**: Plotly figure JSON

### `GET /api/pitch-frames`

Get frame range for a specific `MLBAM_GUID`.

**Parameters**:

* `guid` (required): `MLBAM_GUID`
* `game_id` (optional): `MLBAM_GAME_ID` context when present in the CSV

**Response**: `{"min": <int>, "max": <int>}`

### `GET /api/search-pitch`

Search for `MLBAM_GUID` values matching a query.

**Parameters**:

* `q` (optional): Search query string

**Response**: `{"pitch_ids": [<list of matching MLBAM_GUID values>]}`

## Architecture

### Data Flow


1. CSV loaded at startup and indexed by `MLBAM_GUID`
2. Frame ranges computed per pitch
3. Client requests frame data via API
4. Server filters data, extracts positions, generates Plotly figure
5. Client renders interactive 3D visualization

### Coordinate System

* **X**: Left/Right (negative = left, positive = right)
* **Y**: Forward/Backward (negative = backward, positive = forward)
* **Z**: Up/Down (0 = ground level, positive = upward)

### Visualization Components

* **Ground Plane**: Brown surface at z=0
* **Home Plate**: White pentagon outline
* **Skeleton**: Blue lines connecting joints, blue markers for joint positions
* **Bat**: Brown line from knob to top with markers
* **Ball**: Red sphere marker

## Troubleshooting

### No Data Displayed

* Verify CSV contains required columns (`MLBAM_GUID` plus `FRAME`)
* Check that position columns exist (`CENTER_TX/Y/Z`, `TOP_TX/Y/Z`, `KNOB_TX/Y/Z`)
* Ensure frame numbers are numeric and sequential

### Performance Issues

* Large CSV files (>100MB) may take time to load
* Consider filtering CSV to specific sessions/pitches before loading
* Use `--debug` flag to see detailed error messages

### Missing Skeleton

* Skeleton visualization requires joint position columns
* If joints are missing, only bat and ball will be displayed
* Check console for warnings about missing joint data

## Notes

* Data is loaded into memory at startup for fast frame access
* Camera position is preserved when switching frames in 3D view
* Playback automatically loops from end to start of pitch range
* Frame numbers are zero-indexed per pitch
* Legacy CSVs without `MLBAM_GUID` are still accepted internally by the viewer for archival inspection, but the active pipeline emits MLBAM identifiers.

## Example

```bash
# Start server
python hitting_viewer_app.py data/kt_ocean_multi_game.csv

# Open browser to:
# http://127.0.0.1:5000

# Search for a specific pitch:
# Type "2024_09_03_18_40" in search box

# Select pitch and play:
# Choose from dropdown → Click Play → Adjust speed as needed
```


---

*Last Updated: 2025-12-03* *Maintainer: Junior Integrated Performance Scientist* 
