from __future__ import annotations

import argparse
import gc
import logging
import re
from pathlib import Path
from typing import Literal

import pandas as pd


RESULTS_DIR = Path("results")
DATA_DIR = Path("data")
GROUP_ID_COLS = ("MLBAM_GAME_ID", "MLBAM_GUID")
OUTPUT_ID_COLS = ("MLBAM_GAME_ID", "MLBAM_GUID", "MLBAM_PLAYER_ID", "SESSION_DATE")
DEFAULT_GAME_IDS = (
    "822981",
    "824519",
    "822817",
    "824521",
    "823953",
    "822983",
    "823710",
    "823632",
    "823869",
    "822820",
    "822980",
    "822734",
    "823549",
    "823462",

)
logger = logging.getLogger("missed_distance")
LoadSource = Literal["input_csv", "cache", "snowflake"]
LoadedRows = tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, LoadSource]
SOURCE_INPUT_CSV: LoadSource = "input_csv"
SOURCE_CACHE: LoadSource = "cache"
SOURCE_SNOWFLAKE: LoadSource = "snowflake"


class NoMotionRowsError(RuntimeError):
    """Raised when Snowflake returns no batting motion rows for a requested game."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the missed-distance pipeline with MLBAM GUID/Game identifiers."
    )
    parser.add_argument(
        "--guid",
        help="Optional MLBAM_GUID to fetch from Snowflake. Combine with --game-id to run one play.",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        help="Enriched hitting CSV containing MLBAM identifiers and motion rows.",
    )
    parser.add_argument(
        "--game-id",
        "--game-ids",
        dest="game_ids",
        nargs="+",
        help=(
            "One or more MLBAM_GAME_ID values to fetch from Snowflake. "
            "If --guid is omitted, each game runs as a full-game job."
        ),
    )
    parser.add_argument(
        "--game-chunk-size",
        type=int,
        default=1,
        help=(
            "Number of game IDs per bulk chunk. Games are still loaded, computed, "
            "written, and released one at a time to keep memory bounded."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory for raw Snowflake pulls, organized by game_id.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory for discrete/time_series folders and the outcome_counts CSV.",
    )
    parser.add_argument(
        "--skip-validation-plots",
        action="store_true",
        help="Skip per-pitch validation plot generation.",
    )
    parser.add_argument(
        "--validation-plot-format",
        choices=("none", "png", "html", "both"),
        default="none",
        help="Validation plot format to generate. Defaults to none for batch runs.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Bypass cached raw CSVs in --data-dir and re-query Snowflake.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="For bulk game runs, log a failed game and continue with the remaining IDs.",
    )
    args = parser.parse_args()
    if args.input_csv is None and args.guid is None and args.game_ids is None and DEFAULT_GAME_IDS:
        args.game_ids = list(DEFAULT_GAME_IDS)

    if args.input_csv is not None and (args.guid is not None or args.game_ids is not None):
        parser.error("--input-csv cannot be combined with --guid or --game-id.")
    if args.input_csv is None and args.guid is None and args.game_ids is None:
        parser.error("Provide --game-id, --guid, --input-csv, or set DEFAULT_GAME_IDS in missed_distance.py.")
    if args.guid is not None and args.game_ids is not None and len(args.game_ids) > 1:
        parser.error("--guid can only be combined with one --game-id.")
    if args.game_chunk_size < 1:
        parser.error("--game-chunk-size must be 1 or greater.")
    args.game_id = args.game_ids[0] if args.game_ids is not None and len(args.game_ids) == 1 else None
    return args


def load_hitting_rows(args: argparse.Namespace) -> LoadedRows:
    """Load batting motion + report data from CSV, cache, or Snowflake.

    Cache-first strategy (mirrors geometry-frustum-pipeline):
        1. Check data/{game_label}/ for existing CSV checkpoint
           files from a prior run.
        2. If found and --force is not set, load from disk (no Snowflake
           login required).
        3. Otherwise query Snowflake; raw results are persisted by the
           caller in _write_raw_snowflake_data for the next run.
    """
    if args.input_csv is not None:
        return _load_from_csv(args.input_csv)

    game_label = _safe_label(args.game_id) if args.game_id else None
    if game_label is not None:
        cached_motion, cached_report = _try_load_cache(args.data_dir, game_label, force=args.force)
        if cached_motion is not None:
            return _filter_cached_for_compute(cached_motion, cached_report, args, game_label)

        if args.force:
            logger.info("--force set; bypassing cached data for %s", args.data_dir / game_label)
        else:
            logger.info("No cached motion_sequence.csv found in %s; querying Snowflake.", args.data_dir / game_label)
    elif args.guid is not None and not args.force:
        cached_game_label = _find_cached_game_for_guid(args.data_dir, args.guid)
        if cached_game_label is not None:
            cached_motion, cached_report = _try_load_cache(args.data_dir, cached_game_label, force=False)
            if cached_motion is not None:
                return _filter_cached_for_compute(cached_motion, cached_report, args, cached_game_label)
        logger.info(
            "No cached motion_sequence.csv containing GUID %s found under %s; querying Snowflake.",
            args.guid,
            args.data_dir,
        )
    elif args.force:
        logger.info("--force set; bypassing cached data.")
    return _load_from_snowflake(args)


def _load_from_csv(input_csv: Path) -> LoadedRows:
    """Load a user-supplied enriched hitting CSV without touching Snowflake."""
    if not input_csv.exists():
        raise FileNotFoundError(f"Input file not found: {input_csv}")
    df = pd.read_csv(input_csv)
    return df, df, None, SOURCE_INPUT_CSV


def _filter_cached_for_compute(
    cached_motion: pd.DataFrame,
    cached_report: pd.DataFrame | None,
    args: argparse.Namespace,
    game_label: str | None,
) -> LoadedRows:
    """Return cached full-game data plus the requested compute subset."""
    logger.info("Loaded cached motion data from %s", args.data_dir / str(game_label))
    compute_df = cached_motion
    if args.guid is not None:
        compute_df = _filter_guid_rows(cached_motion, args.guid)
        if compute_df.empty:
            available = (
                sorted(cached_motion["MLBAM_GUID"].dropna().unique())
                if "MLBAM_GUID" in cached_motion.columns
                else []
            )
            logger.error("GUID %s not found in cached motion data.", args.guid)
            logger.error("Available GUIDs: %s", available)
            raise SystemExit(1)
    return compute_df, cached_motion, cached_report, SOURCE_CACHE


def _load_from_snowflake(args: argparse.Namespace) -> LoadedRows:
    """Load batting rows from Snowflake; keep imports lazy to avoid auth in CSV/cache paths."""
    from modules.data_service import (
        diagnose_batting_motion_pull,
        format_empty_pull_diagnostics,
        get_batting_hitting_report,
        get_batting_motion,
    )

    if args.game_id is not None:
        full_game_df = get_batting_motion(game_id=args.game_id)
        if full_game_df.empty:
            raise NoMotionRowsError(
                "No Snowflake batting motion rows were returned.\n"
                f"Game ID filter: {args.game_id}\n"
                "Likely issue: no joined batting motion rows exist for this game_id."
            )

        hitting_report = get_batting_hitting_report(game_id=args.game_id)
        compute_df = full_game_df
        if args.guid is not None:
            compute_df = _filter_guid_rows(full_game_df, args.guid)
            if compute_df.empty:
                diagnostics = diagnose_batting_motion_pull(args.guid, game_id=args.game_id)
                logger.error(format_empty_pull_diagnostics(diagnostics))
                raise SystemExit(1)

        return compute_df, full_game_df, hitting_report, SOURCE_SNOWFLAKE

    df = get_batting_motion(args.guid, game_id=args.game_id)
    if df.empty:
        diagnostics = diagnose_batting_motion_pull(args.guid, game_id=args.game_id)
        logger.error(format_empty_pull_diagnostics(diagnostics))
        raise SystemExit(1)

    inferred_game_id = _single_value(df, "MLBAM_GAME_ID")
    if inferred_game_id is None:
        hitting_report = get_batting_hitting_report(args.guid)
        return df, df, hitting_report, SOURCE_SNOWFLAKE

    full_game_df = get_batting_motion(game_id=inferred_game_id)
    if full_game_df.empty:
        hitting_report = get_batting_hitting_report(args.guid)
        return df, df, hitting_report, SOURCE_SNOWFLAKE

    hitting_report = get_batting_hitting_report(game_id=inferred_game_id)
    compute_df = _filter_guid_rows(full_game_df, args.guid)
    if compute_df.empty:
        compute_df = df
    return compute_df, full_game_df, hitting_report, SOURCE_SNOWFLAKE


def _try_load_cache(
    data_dir: Path,
    game_label: str | None,
    *,
    force: bool = False,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Load cached checkpoint files from a previous run.

    Checks for cached CSV files from a prior run.  Returns (motion_df,
    report_df) on cache hit, or (None, None) to fall through to Snowflake.
    """
    if game_label is None or force:
        return None, None

    cache_dir = data_dir / game_label

    motion_path = _resolve_checkpoint(cache_dir, "motion_sequence")
    if motion_path is None:
        return None, None

    try:
        motion_df = _read_checkpoint(motion_path)
        if motion_df.empty:
            logger.warning("Cached %s is empty; re-querying Snowflake.", motion_path.name)
            return None, None
    except Exception as exc:
        logger.warning("Failed to load %s (%s); re-querying Snowflake.", motion_path.name, exc)
        return None, None

    # Resolve report checkpoint (optional).
    report_df = None
    report_path = _resolve_checkpoint(cache_dir, "hitting_report")
    if report_path is not None:
        try:
            report_df = _read_checkpoint(report_path)
        except Exception as exc:
            logger.warning("Failed to load %s (%s); skipping report cache.", report_path.name, exc)

    return motion_df, report_df


def _find_cached_game_for_guid(data_dir: Path, guid: str) -> str | None:
    """Return the cached game folder containing guid, if one is already on disk."""
    if not data_dir.exists():
        return None

    target_guid = str(guid)
    for motion_path in sorted(data_dir.glob("*/motion_sequence.csv")):
        try:
            guid_values = pd.read_csv(motion_path, usecols=["MLBAM_GUID"])
        except ValueError:
            logger.warning("Cached %s has no MLBAM_GUID column; skipping.", motion_path)
            continue
        except Exception as exc:
            logger.warning("Failed to inspect cached %s (%s); skipping.", motion_path, exc)
            continue

        if guid_values["MLBAM_GUID"].astype(str).eq(target_guid).any():
            logger.info("Found GUID %s in cached game data %s", guid, motion_path.parent)
            return motion_path.parent.name
    return None


def _resolve_checkpoint(cache_dir: Path, stem: str) -> Path | None:
    """Return the cached CSV path if it exists."""
    path = cache_dir / f"{stem}.csv"
    return path if path.exists() else None


def _read_checkpoint(path: Path) -> pd.DataFrame:
    """Read a cached CSV checkpoint file."""
    return pd.read_csv(path)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    from biomech_functions.functions import compute_discrete_and_time_series
    from modules.pipeline_normalization import (
        CANONICAL_ID_COLUMNS,
        normalize_mlbam_hitting_data,
    )

    args = parse_args()
    if args.game_ids is not None and len(args.game_ids) > 1:
        _run_bulk_game_ids(
            args,
            compute_discrete_and_time_series=compute_discrete_and_time_series,
            normalize_mlbam_hitting_data=normalize_mlbam_hitting_data,
            canonical_id_columns=CANONICAL_ID_COLUMNS,
        )
        return

    try:
        _run_single_job(
            args,
            compute_discrete_and_time_series=compute_discrete_and_time_series,
            normalize_mlbam_hitting_data=normalize_mlbam_hitting_data,
            canonical_id_columns=CANONICAL_ID_COLUMNS,
        )
    except NoMotionRowsError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc


def _run_bulk_game_ids(
    args: argparse.Namespace,
    *,
    compute_discrete_and_time_series,
    normalize_mlbam_hitting_data,
    canonical_id_columns: tuple[str, ...],
) -> None:
    game_ids = list(args.game_ids or [])
    logger.info(
        "Bulk game run: %s game IDs, chunk size %s.",
        len(game_ids),
        args.game_chunk_size,
    )
    for chunk_idx, chunk in enumerate(_chunks(game_ids, args.game_chunk_size), start=1):
        total_chunks = (len(game_ids) + args.game_chunk_size - 1) // args.game_chunk_size
        logger.info(
            "Starting game chunk %s/%s: %s",
            chunk_idx,
            total_chunks,
            ", ".join(map(str, chunk)),
        )
        for game_id in chunk:
            run_args = argparse.Namespace(**vars(args))
            run_args.game_id = game_id
            run_args.game_ids = [game_id]
            logger.info("Starting bulk game_id=%s", game_id)
            try:
                _run_single_job(
                    run_args,
                    compute_discrete_and_time_series=compute_discrete_and_time_series,
                    normalize_mlbam_hitting_data=normalize_mlbam_hitting_data,
                    canonical_id_columns=canonical_id_columns,
                )
            except NoMotionRowsError as exc:
                logger.warning("Skipping game_id=%s: %s", game_id, exc)
            except SystemExit:
                if not args.continue_on_error:
                    raise
                logger.exception("Bulk game_id=%s failed with a controlled exit; continuing.", game_id)
            except Exception:
                if not args.continue_on_error:
                    raise
                logger.exception("Bulk game_id=%s failed; continuing.", game_id)
            gc.collect()
            logger.info("Finished bulk game_id=%s", game_id)


def _run_single_job(
    args: argparse.Namespace,
    *,
    compute_discrete_and_time_series,
    normalize_mlbam_hitting_data,
    canonical_id_columns: tuple[str, ...],
) -> None:
    logger.info("[1/5] Loading data (game_id=%s, guid=%s)", args.game_id, args.guid)
    raw_df, raw_motion_df, hitting_report, load_source = load_hitting_rows(args)
    logger.info("%s motion rows loaded", f"{len(raw_df):,}")

    run_label = _run_label(args, raw_df)
    if load_source == SOURCE_SNOWFLAKE:
        logger.info("[2/5] Caching raw Snowflake data to disk...")
        raw_paths = _write_raw_snowflake_data(
            raw_motion_df,
            hitting_report,
            data_dir=args.data_dir,
            game_label=_game_label(args, raw_df),
        )
    elif load_source == SOURCE_CACHE:
        logger.info(
            "[2/5] Using cached raw data from %s; skipping Snowflake query/cache write.",
            args.data_dir / _game_label(args, raw_df),
        )
        raw_paths = {}
    else:
        logger.info("[2/5] Using input CSV; skipping Snowflake query/cache write.")
        raw_paths = {}

    logger.info("[3/5] Normalizing MLBAM hitting data...")
    df = normalize_mlbam_hitting_data(raw_df)
    logger.info("%s rows after normalization", f"{len(df):,}")

    if hitting_report is not None and not hitting_report.empty:
        df = _merge_report_columns(df, hitting_report)

    logger.info("[4/5] Computing missed-distance discrete outputs and time series...")
    validation_plot_format = "none" if args.skip_validation_plots else args.validation_plot_format
    discrete_df, time_series, outcome_counts = compute_discrete_and_time_series(
        df,
        group_id_cols=GROUP_ID_COLS,
        output_id_cols=OUTPUT_ID_COLS,
        save_validation_plots=validation_plot_format != "none",
        validation_plot_format=validation_plot_format,
    )

    logger.info("[5/5] Writing output files...")
    discrete_dir = args.results_dir / "discrete"
    ts_dir = args.results_dir / "time_series"
    discrete_dir.mkdir(parents=True, exist_ok=True)
    ts_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    discrete_path = discrete_dir / f"{run_label}_discrete.csv"
    ts_path = ts_dir / f"{run_label}_time_series.csv"
    counts_path = args.results_dir / f"{run_label}_outcome_counts.csv"

    discrete_df.to_csv(discrete_path, index=False)
    time_series.to_csv(ts_path, index=False)
    _write_outcome_counts_csv(outcome_counts, counts_path)

    n_frames = len(df)
    n_pitches = df[list(GROUP_ID_COLS)].drop_duplicates().shape[0]
    logger.info("Missed-distance complete.")
    logger.info("Input: %s frames across %s MLBAM pitch IDs", f"{n_frames:,}", f"{n_pitches:,}")
    logger.info("Canonical IDs: %s", ", ".join(canonical_id_columns))
    for label, path in raw_paths.items():
        logger.info("Wrote raw %s: %s", label, path)
    logger.info("Wrote: %s", discrete_path)
    logger.info("Wrote: %s", ts_path)
    logger.info("Wrote: %s", counts_path)
    logger.info("Outcome counts: %s", outcome_counts)


def _chunks(values: list[str], size: int) -> list[list[str]]:
    """Split values into fixed-size chunks."""
    return [values[idx:idx + size] for idx in range(0, len(values), size)]


def _write_outcome_counts_csv(outcome_counts: dict[str, int], path: Path) -> None:
    """Write outcome counts as a simple two-column CSV."""
    pd.DataFrame(outcome_counts.items(), columns=["OUTCOME", "COUNT"]).to_csv(path, index=False)


def _merge_report_columns(
    motion_df: "pd.DataFrame",
    report_df: "pd.DataFrame",
) -> "pd.DataFrame":
    """Merge per-pitch report columns onto motion frames.

    The hitting report has one row per pitch with columns like TAKE, SWING,
    BAD, MISS, CHECK_SWING, BALL_CONTACT, and biomechanical report values.
    The motion DataFrame has many rows (frames) per pitch.  We left-join on
    (SESSION_ID, PITCH_ID) so every motion frame inherits the report-level
    classification needed by _infer_outcome_with_take_skip.
    """
    join_keys = ["SESSION_ID", "PITCH_ID"]
    for k in join_keys:
        if k not in motion_df.columns or k not in report_df.columns:
            logger.warning("Cannot merge report; missing join key %s", k)
            return motion_df

    # Only bring in columns that aren't already in the motion data
    overlap = set(motion_df.columns) & set(report_df.columns) - set(join_keys)
    report_cols = [c for c in report_df.columns if c not in overlap or c in join_keys]
    report_slim = report_df[report_cols].drop_duplicates(subset=join_keys)

    before = len(motion_df)
    merged = motion_df.merge(report_slim, on=join_keys, how="left")
    after = len(merged)

    new_cols = len(report_slim.columns) - len(join_keys)
    logger.info(
        "Merged %s report columns onto motion data (%s -> %s rows)",
        new_cols,
        f"{before:,}",
        f"{after:,}",
    )
    return merged


def _run_label(args: argparse.Namespace, raw_df) -> str:
    if args.input_csv is not None:
        return _safe_label(args.input_csv.stem)

    parts = [_game_label(args, raw_df)]
    if args.guid is not None:
        parts.append(_safe_label(args.guid))
    return "_".join(parts)


def _game_label(args: argparse.Namespace, raw_df) -> str:
    game_id = args.game_id or _single_value(raw_df, "MLBAM_GAME_ID") or "unknown_game"
    return _safe_label(game_id)


def _write_raw_snowflake_data(
    motion_df,
    hitting_report_df,
    *,
    data_dir: Path,
    game_label: str,
) -> dict[str, Path]:
    """Checkpoint raw Snowflake pulls to CSV for fast reload on next run."""
    output_dir = data_dir / game_label
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {
        "motion sequence": output_dir / "motion_sequence.csv",
    }
    motion_df.to_csv(paths["motion sequence"], index=False)

    if hitting_report_df is not None:
        paths["hitting report"] = output_dir / "hitting_report.csv"
        hitting_report_df.to_csv(paths["hitting report"], index=False)

    return paths


def _filter_guid_rows(df, guid: str):
    if "MLBAM_GUID" not in df.columns:
        return df.iloc[0:0].copy()
    return df[df["MLBAM_GUID"].astype(str) == str(guid)].copy()


def _single_value(df, column: str) -> object | None:
    if column not in df.columns:
        return None
    values = df[column].dropna().unique()
    if len(values) == 1:
        return values[0]
    return None


def _safe_label(value: object) -> str:
    text = str(value)
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    return clean or "unknown"


if __name__ == "__main__":
    main()
