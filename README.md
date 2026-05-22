# Missed Distance Pipeline

Hitting-only pipeline for pulling Kinatrax batting motion data, normalizing it to MLBAM identifiers, computing K80 missed-distance metrics, and writing discrete and frame-level CSV outputs.

The active pipeline is:

```text
Snowflake or cached CSV
  -> MLBAM hitting normalization
  -> optional hitting-report merge
  -> BAT_80 local coordinate system
  -> local/global K80 miss vectors, distances, velocities, and speeds
  -> results/discrete + results/time_series + outcome counts
```

For the short command-only version, see [QUICKSTART.md](QUICKSTART.md).

## Current Scope

This repo computes missed distance for hitting data only.

Current behavior:

- Metadata: `MLBAM_GAME_ID` + `MLBAM_GUID`.
  - `SESSION_ID` and `PITCH_ID` are retained internally for cache/report joins, but are not emitted in the final discrete or time-series outputs.
- K80 is treated as the sweet spot.
  - This is because all Kinatrax bat speed variables are derived relative to the 80% bat length
- Output format: CSV

## Repository Layout

```text
.
├── missed_distance.py              # CLI entry point
├── modules/
│   ├── data_service.py             # Snowflake queries, connector, empty-pull diagnostics
│   └── pipeline_normalization.py   # MLBAM/schema normalization
├── biomech_functions/
│   ├── functions.py                # BAT_80 LCS + missed-distance computation
│   ├── schema.py                   # Output column ordering
│   └── data_transformations.py     # Small metadata/column helpers
├── viewer/
│   ├── hitting_viewer_app.py       # Flask/Plotly viewer for cached game CSVs
│   └── HITTING_VIEWER_README.md
├── tests/
├── data/                           # Local raw cache, ignored by git
└── results/                        # Pipeline outputs, ignored by git
```

Generated local folders:

```text
data/<GAME_ID>/motion_sequence.csv
data/<GAME_ID>/hitting_report.csv

results/discrete/<RUN_LABEL>_discrete.csv
results/time_series/<RUN_LABEL>_time_series.csv
results/<RUN_LABEL>_outcome_counts.csv

fig_outputs/MLBAM_GAME_GUID_MD_VALIDATION/K80/   # only when plots are enabled
```

## Setup

Use the Python environment that has the project dependencies installed. The pipeline uses:

- `pandas`
- `numpy`
- `scipy`
- `matplotlib`
- `plotly`
- `snowflake-connector-python`
- `python-dotenv` optional, for loading `.env`
- `flask` for the viewer
- `pytest` for tests

Example install line if you are in a clean environment:

```bash
pip install pandas numpy scipy matplotlib plotly snowflake-connector-python python-dotenv flask pytest
```

For macOS/Codex runs, setting `MPLCONFIGDIR` avoids Matplotlib/font cache write warnings:

```bash
export MPLCONFIGDIR=/private/tmp
```

## Snowflake Configuration

Direct pulls read Snowflake config from environment variables. If `python-dotenv` is installed, `modules/data_service.py` will also load the nearest `.env` file.

Required:

```text
SNOWFLAKE_ACCOUNT
SNOWFLAKE_USER
SNOWFLAKE_WAREHOUSE
SNOWFLAKE_DATABASE
SNOWFLAKE_SCHEMA
```

Optional:

```text
SNOWFLAKE_ROLE
SNOWFLAKE_AUTHENTICATOR
SNOWFLAKE_PASSWORD
```

Default auth is:

```text
SNOWFLAKE_AUTHENTICATOR=externalbrowser
```

That opens the Okta/browser SSO flow. If cached data exists under `data/<GAME_ID>/`, the pipeline does not contact Snowflake unless `--force` is used.

## Data Pulls

The motion pull uses enriched batting motion data:

```sql
SELECT pps.*,
       pr.*
FROM   KINATRAX.BATTING_MOTION_SEQUENCE_BATTING AS pr
INNER JOIN KINATRAX.BATTING_PARAMETER_SET AS pps
  ON  pr.SESSION_ID = pps.SESSION_ID
 AND  pr.PITCH_ID   = pps.PITCH_ID
 AND  pr.TEAM_NAME  = pps.TEAM_NAME
WHERE ...
ORDER BY pps.MLBAM_GAME_ID ASC, pps.MLBAM_GUID ASC, pr.TIMESTAMP ASC
```

The hitting report pull uses:

```sql
SELECT pps.*,
       pr.*
FROM   KINATRAX.BATTING_REPORTS AS pr
INNER JOIN KINATRAX.BATTING_PARAMETER_SET AS pps
  ON  pr.SESSION_ID = pps.SESSION_ID
 AND  pr.PITCH_ID   = pps.PITCH_ID
 AND  pr.TEAM_NAME  = pps.TEAM_NAME
WHERE ...
ORDER BY pps.MLBAM_GAME_ID ASC, pps.MLBAM_GUID ASC, pps.SESSION_DATE ASC, pps.SESSION_ID ASC, pps.PITCH_ID ASC
```

Filters are parameterized by `MLBAM_GAME_ID`, `MLBAM_GUID`, or both.

## Running The Pipeline

Run the default game list from `missed_distance.py`:

```bash
MPLCONFIGDIR=/private/tmp python missed_distance.py --skip-validation-plots
```

Run one full game:

```bash
MPLCONFIGDIR=/private/tmp python missed_distance.py --game-id 822820 --skip-validation-plots
```

Run one trial within one game:

```bash
MPLCONFIGDIR=/private/tmp python missed_distance.py \
  --game-id 822820 \
  --guid 2597eb35-5407-3aee-9ea9-5131e21139ac \
  --skip-validation-plots
```

Run multiple games without loading them all at once:

```bash
MPLCONFIGDIR=/private/tmp python missed_distance.py \
  --game-id 824358 824599 822981 823384 \
  --game-chunk-size 1 \
  --continue-on-error \
  --skip-validation-plots
```

Run from an already enriched local CSV:

```bash
MPLCONFIGDIR=/private/tmp python missed_distance.py \
  --input-csv path/to/enriched_motion.csv \
  --skip-validation-plots
```

Force a fresh Snowflake pull even if cache exists:

```bash
MPLCONFIGDIR=/private/tmp python missed_distance.py \
  --game-id 822820 \
  --force \
  --skip-validation-plots
```

## Cache-First Behavior

The loader checks local disk before Snowflake:

1. `--game-id <GAME_ID>` checks `data/<GAME_ID>/motion_sequence.csv`.
2. `--game-id <GAME_ID> --guid <GUID>` loads the cached game and filters the GUID.
3. `--guid <GUID>` without a game scans `data/*/motion_sequence.csv` for that GUID.
4. If a cache hit is found, no Snowflake connector is loaded and no browser auth starts.
5. If cache is missing, or `--force` is set, the pipeline queries Snowflake and writes the raw game pull to `data/<GAME_ID>/`.

On a cache hit, the log should say:

```text
Using cached raw data from data/<GAME_ID>; skipping Snowflake query/cache write.
```

## Normalization

`modules/pipeline_normalization.py` standardizes column names and validates required fields.

Required canonical IDs:

```text
MLBAM_GUID
MLBAM_GAME_ID
MLBAM_PLAYER_ID
SESSION_DATE
```

Required internal keys:

```text
SESSION_ID
PITCH_ID
```

Required geometry:

```text
CENTER_TX/CENTER_TY/CENTER_TZ
TOP_TX/TOP_TY/TOP_TZ
KNOB_TX/KNOB_TY/KNOB_TZ
LEFTFOOT_TX
RIGHTFOOT_TX
```

Important normalization behavior:

- Column names are uppercased and whitespace/dashes become underscores.
- `FRAME` is generated per `MLBAM_GAME_ID` + `MLBAM_GUID` if missing.
- All-zero ball centers, where `CENTER_TX/CENTER_TY/CENTER_TZ == 0/0/0`, are converted to nulls. This prevents missing ball observations from being treated as a real ball at home plate.
- Rows with null canonical identifiers are kept, but a warning is logged.

## Hitting Report Merge

When `hitting_report.csv` is available, report-level columns are merged onto every motion frame with:

```text
SESSION_ID + PITCH_ID
```

This brings in tags/events such as:

```text
TAKE
SWING
MISS
BALL_CONTACT
CHECK_SWING
DOWNSWING
BALL_MIN
BAT_STOP
R
L
```

Take pitches are skipped from missed-distance computation and counted in outcome counts.

## BAT_80 Local Coordinate System

The local frame is centered at the K80 sweet spot:

```python
BAT_80 = KNOB + 0.80 * (TOP - KNOB)
```

Current LCS construction in `construct_bat_80_lcs()`:

```python
y_hat = normalize(TOP - BAT_80)

bat_temp[1:] = BAT_80[:-1] - BAT_80[1:]
bat_temp[0] = bat_temp[1]

z_candidates = -cross(y_hat, bat_temp)
z_candidates = backfill(z_candidates, anchors=[BALL_MIN, BAT_STOP])
z_candidates = project_onto_plane_perpendicular_to_y(z_candidates, y_hat)
z_hat = normalize(z_candidates)

x_hat = -normalize(cross(y_hat, z_hat))
if hitter_is_left_handed:
    z_hat = -z_hat

R_80 = [x_hat, y_hat, z_hat]
```

Axis intent:

- `Y` is along the bat shaft toward the top.
- `Z` is derived from BAT_80 motion and forced perpendicular to `Y`.
- `X` completes the local frame from `Y` and `Z`.

The pipeline logs two checks for the internal `R_80` matrix:

```text
unit vectors confirmed for R_80 matrix
axis pairs confirmed orthogonal for R_80 matrix
```

Those checks confirm that the axes have norm 1 and that `X/Y`, `Y/Z`, and `X/Z` dot products round to 0.

## Missed Distance Method

Global K80:

```python
SS_K80_GLOBAL = KNOB + 0.80 * (TOP - KNOB)
MISS_VECTOR_GLOBAL_K80 = BALL_GLOBAL - SS_K80_GLOBAL
MISSED_DISTANCE_GLOBAL_K80_BY_FRAME = norm(MISS_VECTOR_GLOBAL_K80)

T_MIN_GLOBAL_K80_IDX = argmin(
    MISSED_DISTANCE_GLOBAL_K80_BY_FRAME[downswing_idx:bat_stop_idx]
)
MISSED_DISTANCE_GLOBAL_K80 = MISSED_DISTANCE_GLOBAL_K80_BY_FRAME[T_MIN_GLOBAL_K80_IDX]
```

Local K80:

```python
BALL_IN_BAT = R_80.T @ (BALL_GLOBAL - BAT_80)
MISS_VECTOR_LOCAL_K80 = BALL_IN_BAT

# Local T-min selection uses the full local 3D miss vector.
LOCAL_3D_DISTANCE_K80_BY_FRAME = norm(MISS_VECTOR_LOCAL_K80)
T_MIN_LOCAL_K80_IDX = argmin(
    LOCAL_3D_DISTANCE_K80_BY_FRAME[downswing_idx:bat_stop_idx]
)

# Local missed distance is the full local 3D norm at that same frame.
MISSED_DISTANCE_LOCAL_K80 =
    LOCAL_3D_DISTANCE_K80_BY_FRAME[T_MIN_LOCAL_K80_IDX]
```

Because `BAT_80` is the local origin, the local K80 sweet spot is `(0, 0, 0)`. The local vector is recorded as full `X/Y/Z`, and the local distance is the full 3D norm of that vector. Since `R_80` is orthonormal, the local 3D distance should match the global K80 distance at the same frame while still preserving the local axis interpretation.

Current T-min selection:

- Search window starts at `DOWNSWING`, then `DS`, then `START_DATA`, then frame 0.
- Search window ends at `BAT_STOP`, otherwise the final frame.
- `T_MIN_LOCAL_K80` is selected from `sqrt(X^2 + Y^2 + Z^2)` in bat-local space.
- `T_MIN_GLOBAL_K80` is selected from the global K80 scalar distance.
- `MISSED_DISTANCE_LOCAL_K80` is computed after local T-min from `sqrt(X^2 + Y^2 + Z^2)`.
- `MISSED_DISTANCE_GLOBAL_K80` is computed at global T-min from `sqrt(X^2 + Y^2 + Z^2)`.

## Velocity, Speed, And Filtering

Velocity is computed from vector derivatives:

```python
velocity_xyz = derivative(miss_vector_xyz)
```

Then XYZ velocity components are filtered:

```text
filter type: 4th-order zero-phase Butterworth
sample rate: 300 Hz
cutoff: 30 Hz
```

Speed is then computed from the filtered velocity components:

```python
speed = norm(filtered_velocity_xyz)
```

This is applied to both:

```text
MISS_VECTOR_LOCAL_K80
MISS_VECTOR_GLOBAL_K80
```

For short series that cannot support `filtfilt` padding, the filter returns the input unchanged.

## Direction Flags

The discrete output includes four local miss direction flags after metadata:

```text
CAPPED
JAMMED
OVER
UNDER
```

They are based on `MISS_VECTOR_LOCAL_K80` at `T_MIN_LOCAL_K80`:

```text
CAPPED = 1 if local Y > 0 else 0
JAMMED = 1 if local Y < 0 else 0
OVER   = 1 if local Z < 0 else 0
UNDER  = 1 if local Z > 0 else 0
```

`CAPPED` and `JAMMED` cannot both be true. `OVER` and `UNDER` cannot both be true. One horizontal and one vertical flag can both be true.

## Output Files

### `results/discrete/<RUN_LABEL>_discrete.csv`

One row per processed `MLBAM_GAME_ID` + `MLBAM_GUID`.

Column groups:

```text
metadata
local miss direction flags
tags/outcome fields
events
START_DATA foot/ankle positions
BALL_MIN + BALL_MIN_DIST_MISS
T_MIN_LOCAL_K80 + T_MIN_GLOBAL_K80
local/global ball, bat, and sweet-spot positions at T-min
local/global miss vectors
local/global missed distances
max miss velocity components from downswing/start frame to each space's T-min frame
max miss speeds
velocity and speed at T-min
```

Key generated columns include:

```text
T_MIN_LOCAL_K80
T_MIN_GLOBAL_K80
BALL_IN_BAT_AT_TMIN_K80_X/Y/Z
BALL_AT_TMIN_K80_X/Y/Z
BAT_KNOB_AT_TMIN_K80_X/Y/Z
BAT_TOP_AT_TMIN_K80_X/Y/Z
SS_K80_AT_TMIN_X/Y/Z
MISS_VECTOR_LOCAL_K80_X/Y/Z
MISS_VECTOR_GLOBAL_K80_X/Y/Z
MISSED_DISTANCE_LOCAL_K80
MISSED_DISTANCE_GLOBAL_K80
MAX_MISS_VELOCITY_K80_LOCAL_X/Y/Z
MAX_MISS_VELOCITY_K80_GLOBAL_X/Y/Z
MAX_MISS_SPEED_K80_LOCAL
MAX_MISS_SPEED_K80_GLOBAL
```

### `results/time_series/<RUN_LABEL>_time_series.csv`

One row per frame per `MLBAM_GAME_ID` + `MLBAM_GUID`.

Current order:

```text
MLBAM_GAME_ID
MLBAM_GUID
FRAME
BALL_X/Y/Z
BALL_IN_BAT_X/Y/Z
BAT_KNOB_X/Y/Z
BAT_TOP_X/Y/Z
SS_K80_X/Y/Z
MISS_VECTOR_LOCAL_K80_X/Y/Z
MISS_VECTOR_GLOBAL_K80_X/Y/Z
MISSED_DISTANCE_LOCAL_K80
MISSED_DISTANCE_GLOBAL_K80
MISS_VECTOR_VELOCITY_LOCAL_K80_X/Y/Z
MISS_VECTOR_VELOCITY_GLOBAL_K80_X/Y/Z
MISSED_DISTANCE_LOCAL_K80_SPEED
MISSED_DISTANCE_GLOBAL_K80_SPEED
foot/ankle positions
```

### `results/<RUN_LABEL>_outcome_counts.csv`

Two-column count file:

```text
OUTCOME,COUNT
```

Typical outcomes:

```text
hit
miss
CHECK_SWING
no_report
TAKE_skipped
empty_frames_skipped
```

## Validation Plots

Validation plots are off by default in batch-style runs when you pass:

```bash
--skip-validation-plots
```

If enabled, plots are written under:

```text
fig_outputs/MLBAM_GAME_GUID_MD_VALIDATION/K80/
```

Use:

```bash
python missed_distance.py --game-id 822820 --validation-plot-format png
python missed_distance.py --game-id 822820 --validation-plot-format html
python missed_distance.py --game-id 822820 --validation-plot-format both
```

## Viewer

The viewer is useful for checking one cached game visually.

```bash
MPLCONFIGDIR=/private/tmp python viewer/hitting_viewer_app.py \
  data/822820/motion_sequence.csv \
  --port 5001
```

Open:

```text
http://127.0.0.1:5001
```

The viewer loads a CSV into memory at startup, lets you select a `MLBAM_GUID`, and displays the skeleton, bat, ball, K80 point, and `R80` axes.

More detail is in [viewer/HITTING_VIEWER_README.md](viewer/HITTING_VIEWER_README.md).

## Tests

Run the test suite:

```bash
MPLCONFIGDIR=/private/tmp pytest
```

Important coverage:

- Snowflake query construction and parameter binding
- cache-first CLI behavior
- normalization and required IDs/geometry
- all-zero ball centers becoming nulls
- fixed 30 Hz Butterworth behavior
- BAT_80 LCS unit/orthogonality behavior
- K80-only output schema snapshots
- no retired `K82`, `K67`, residual, or generated unit-suffix columns
- canonical output paths

## Troubleshooting

### Browser auth appears when I expected cache

Check that the cache exists:

```bash
ls data/<GAME_ID>/motion_sequence.csv
```

Then run without `--force`.

### A game returns zero rows from Snowflake

That means the joined batting motion query returned no rows for that game. In bulk mode, `NoMotionRowsError` games are skipped. For GUID-specific diagnostics, run with both `--game-id` and `--guid`; the pipeline logs parameter-set and joined-motion counts.

### The viewer axes look non-perpendicular

In 3D perspective, true 90-degree vectors can look skewed after projection onto a 2D screen. Check the terminal log for:

```text
axis pairs confirmed orthogonal for R_80 matrix
```

or inspect `dot(X, Y)`, `dot(Y, Z)`, and `dot(X, Z)` directly.

### Output has missing ball rows

All-zero ball centers are intentionally converted to nulls during normalization. This keeps absent ball observations from contaminating miss-distance calculations.

### Results folder has multiple games

That is expected. Each run label writes separate files:

```text
results/discrete/<label>_discrete.csv
results/time_series/<label>_time_series.csv
results/<label>_outcome_counts.csv
```

## Design Notes

- This is a clean-break K80 pipeline.
- The canonical public output identity is `MLBAM_GAME_ID` + `MLBAM_GUID`.
- Snowflake pulls are cached on disk because SSO is slow and full-game motion files are large.
- Bulk mode processes games one at a time to avoid loading multiple full games into memory.
- Generated metric columns do not include unit suffixes. Pulled source columns may keep units when they came from Kinatrax/report data, such as `MAX_BAT_SPEED_MPH`.
