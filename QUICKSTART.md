# Missed Distance Quickstart

Short version for running the pipeline.

## Run Defaults

Runs the default game list inside `missed_distance.py`:

```bash
MPLCONFIGDIR=/private/tmp python missed_distance.py --skip-validation-plots
```

## Run One Game

```bash
MPLCONFIGDIR=/private/tmp python missed_distance.py \
  --game-id 822820 \
  --skip-validation-plots
```

## Run One Trial

```bash
MPLCONFIGDIR=/private/tmp python missed_distance.py \
  --game-id 822820 \
  --guid 2597eb35-5407-3aee-9ea9-5131e21139ac \
  --skip-validation-plots
```

## Run Multiple Games

```bash
MPLCONFIGDIR=/private/tmp python missed_distance.py \
  --game-id 824358 824599 822981 823384 \
  --game-chunk-size 1 \
  --continue-on-error \
  --skip-validation-plots
```

## Cache Behavior

If this exists:

```text
data/<GAME_ID>/motion_sequence.csv
```

the pipeline uses it and does not hit Snowflake.

Use this only when you want a fresh Snowflake pull:

```bash
MPLCONFIGDIR=/private/tmp python missed_distance.py \
  --game-id 822820 \
  --force \
  --skip-validation-plots
```

## Outputs

```text
results/discrete/<RUN_LABEL>_discrete.csv
results/time_series/<RUN_LABEL>_time_series.csv
results/<RUN_LABEL>_outcome_counts.csv
```

Raw pulls/cache:

```text
data/<GAME_ID>/motion_sequence.csv
data/<GAME_ID>/hitting_report.csv
```

## Viewer

```bash
MPLCONFIGDIR=/private/tmp python viewer/hitting_viewer_app.py \
  data/822820/motion_sequence.csv \
  --port 5001
```

Open:

```text
http://127.0.0.1:5001
```

## Tests

```bash
MPLCONFIGDIR=/private/tmp pytest
```

## What This Pipeline Computes

- K80-only missed distance.
- Global miss vector: `BALL - SS_K80`.
- Local miss vector: `BALL_IN_BAT`, with BAT_80 as the local origin.
- Local/global distances, velocities, and speeds.
- Direction flags: `CAPPED`, `JAMMED`, `OVER`, `UNDER`.

For the full explanation, see [README.md](README.md).
