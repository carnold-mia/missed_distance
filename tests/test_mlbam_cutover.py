from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from modules.data_service import (
    build_batting_hitting_report_query,
    build_batting_motion_query,
    diagnose_batting_motion_pull,
    format_empty_pull_diagnostics,
    get_batting_motion,
)
from modules.pipeline_normalization import normalize_mlbam_hitting_data


FIXTURES_DIR = Path(__file__).parent / "fixtures"
GENERATED_PREFIXES = (
    "MISSED_DISTANCE_",
    "MISS_VECTOR_",
    "MISS_VELOCITY_",
    "MISS_SPEED_",
    "MAX_MISS_",
    "T_MIN_",
    "BALL_AT_TMIN_",
    "BALL_IN_BAT_AT_TMIN_",
    "BAT_KNOB_AT_TMIN_",
    "BAT_TOP_AT_TMIN_",
    "BAT_TOP_IN_BAT_AT_TMIN_",
    "SWEET_SPOT_",
    "SS_",
)
UNIT_SUFFIXES = ("_MPH", "_KPH", "_M", "_HZ", "_KG", "_LB", "_MPS")


def _minimal_hitting_rows(n_frames: int = 4) -> pd.DataFrame:
    rows = []
    for frame in range(n_frames):
        rows.append(
            {
                "MLBAM_GUID": "guid-1",
                "MLBAM_GAME_ID": 746020,
                "MLBAM_PLAYER_ID": 686611,
                "SESSION_DATE": "2024-09-03",
                "SESSION_ID": "legacy-session",
                "PITCH_ID": "legacy-pitch",
                "TEAM_NAME": "Washington Nationals",
                "TIMESTAMP": frame / 300.0,
                "CENTER_TX": 0.10 + frame * 0.01,
                "CENTER_TY": 0.20,
                "CENTER_TZ": 0.30,
                "TOP_TX": 0.50,
                "TOP_TY": 0.10 + frame * 0.01,
                "TOP_TZ": 0.40,
                "KNOB_TX": 0.10,
                "KNOB_TY": 0.10,
                "KNOB_TZ": 0.10,
                "LEFTFOOT_TX": -0.2,
                "LEFTFOOT_TY": 0.01 + frame * 0.001,
                "LEFTFOOT_TZ": 0.02,
                "RIGHTFOOT_TX": -0.1,
                "RIGHTFOOT_TY": 0.03 + frame * 0.001,
                "RIGHTFOOT_TZ": 0.04,
                "LEFTANKLE_TX": -0.25,
                "LEFTANKLE_TY": 0.05 + frame * 0.001,
                "LEFTANKLE_TZ": 0.06,
                "RIGHTANKLE_TX": -0.15,
                "RIGHTANKLE_TY": 0.07 + frame * 0.001,
                "RIGHTANKLE_TZ": 0.08,
                "BAD": np.nan,
                "R": "R",
                "L": np.nan,
                "SWING": "SWING",
                "MISS": "MISS",
                "TAKE": np.nan,
                "BALL_START": 0,
                "DOWNSWING": 3,
                "BALL_MIN": 10,
                "END_DATA": n_frames - 1,
                "BAT_STOP": n_frames - 2,
                "START_DATA": 2,
                "MAX_BAT_SPEED_MPH": 73.5,
                "HANDEDNESS": 1,
                **{
                    f"SPINE_R{row}{col}": 1.0 if row == col else 0.0
                    for row in (1, 2, 3)
                    for col in (1, 2, 3)
                },
            }
        )
    return pd.DataFrame(rows)


def _compute_fixture_outputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    from biomech_functions import functions as funcs

    normalized = normalize_mlbam_hitting_data(_minimal_hitting_rows(n_frames=18))
    discrete_df, time_series, _ = funcs.compute_discrete_and_time_series(
        normalized,
        group_id_cols=("MLBAM_GAME_ID", "MLBAM_GUID"),
        output_id_cols=("MLBAM_GAME_ID", "MLBAM_GUID", "MLBAM_PLAYER_ID", "SESSION_DATE"),
        save_validation_plots=False,
    )
    return discrete_df, time_series


def _run_cli(monkeypatch: pytest.MonkeyPatch, args: list[object]) -> None:
    from missed_distance import main

    monkeypatch.setattr(sys, "argv", ["missed_distance.py", *[str(arg) for arg in args]])
    main()


def _assert_retired_columns_absent(df: pd.DataFrame) -> None:
    retired_tokens = (
        "CENTER",
        "K67",
        "K82",
        "CUT_",
        "RESIDUAL",
        "KT_MISS",
        "SS_LOCAL",
        "VELOCITY_FILTER",
        "PEAK_SPEED",
    )
    retired_exact = {
        "BAD",
        "R",
        "L",
        "HANDEDNESS",
        "TAKE",
        "SWING",
        "MISS",
        "BALL_CONTACT",
        "CHECK_SWING",
    }
    stale_columns = [
        column
        for column in df.columns
        if column in retired_exact or any(token in column.upper() for token in retired_tokens)
    ]
    assert stale_columns == []


def _assert_generated_columns_unitless(df: pd.DataFrame) -> None:
    offenders = [
        column
        for column in df.columns
        if column.startswith(GENERATED_PREFIXES) and column.endswith(UNIT_SUFFIXES)
    ]
    assert offenders == []


def _assert_position_columns_are_space_neutral(df: pd.DataFrame) -> None:
    retired_position_prefixes = (
        "BALL_GLOBAL",
        "BAT_KNOB_GLOBAL",
        "BAT_TOP_GLOBAL",
        "SS_GLOBAL",
        "BALL_IN_BAT_AT_TMIN_LOCAL",
        "BAT_TOP_IN_BAT_AT_TMIN_LOCAL",
    )
    offenders = [
        column
        for column in df.columns
        if any(column.startswith(prefix) for prefix in retired_position_prefixes)
    ]
    assert offenders == []


def test_batting_motion_query_uses_parameterized_guid_and_game_filter() -> None:
    query, params = build_batting_motion_query("guid-1", game_id=746020)

    assert "KINATRAX.BATTING_MOTION_SEQUENCE_BATTING AS pr" in query
    assert "KINATRAX.BATTING_PARAMETER_SET AS pps" in query
    assert "pr.SESSION_ID = pps.SESSION_ID" in query
    assert "pr.PITCH_ID   = pps.PITCH_ID" in query
    assert "pr.TEAM_NAME  = pps.TEAM_NAME" in query
    assert "pps.MLBAM_GUID = %(guid)s" in query
    assert "pps.MLBAM_GAME_ID = %(game_id)s" in query
    assert params == {"guid": "guid-1", "game_id": 746020}


def test_batting_motion_query_supports_game_only_pull() -> None:
    query, params = build_batting_motion_query(game_id=746020)

    assert "pps.MLBAM_GAME_ID = %(game_id)s" in query
    assert "pps.MLBAM_GUID = %(guid)s" not in query
    assert params == {"game_id": 746020}


def test_batting_hitting_report_query_supports_game_and_guid() -> None:
    query, params = build_batting_hitting_report_query("guid-1", game_id=746020)

    assert "KINATRAX.BATTING_PARAMETER_SET" in query
    assert "MLBAM_GUID = %(guid)s" in query
    assert "MLBAM_GAME_ID = %(game_id)s" in query
    assert params == {"guid": "guid-1", "game_id": 746020}


def test_get_batting_motion_uses_injected_connector() -> None:
    class FakeConnector:
        def __init__(self) -> None:
            self.query: str | None = None
            self.params: dict[str, object] | None = None

        def execute_query_cached(self, query: str, params: dict[str, object]):
            self.query = query
            self.params = params
            return [{"MLBAM_GUID": "guid-1", "MLBAM_GAME_ID": 746020}]

    connector = FakeConnector()

    df = get_batting_motion("guid-1", game_id=746020, connector=connector)

    assert connector.params == {"guid": "guid-1", "game_id": 746020}
    assert "BATTING_MOTION_SEQUENCE_BATTING" in connector.query
    assert df.to_dict("records") == [{"MLBAM_GUID": "guid-1", "MLBAM_GAME_ID": 746020}]


def test_diagnose_batting_motion_pull_reports_empty_join_causes() -> None:
    class FakeConnector:
        def execute_query_cached(self, query: str, params: dict[str, object]):
            assert params["guid"] == "guid-1"
            if "PARAMETER_SET_GAME_ROWS" in query:
                assert params["game_id"] == 746020
                return [{"PARAMETER_SET_GAME_ROWS": 0}]
            if "PARAMETER_SET_ROWS" in query:
                return [{"PARAMETER_SET_ROWS": 1}]
            if "JOINED_MOTION_GAME_ROWS_WITHOUT_TEAM" in query:
                return [{"JOINED_MOTION_GAME_ROWS_WITHOUT_TEAM": 0}]
            if "JOINED_MOTION_GAME_ROWS" in query:
                return [{"JOINED_MOTION_GAME_ROWS": 0}]
            if "JOINED_MOTION_ROWS" in query:
                return [{"JOINED_MOTION_ROWS": 12}]
            if "GROUP BY MLBAM_GAME_ID" in query:
                assert "COUNT(*) AS PARAMETER_ROWS" in query
                assert "ORDER BY PARAMETER_ROWS DESC" in query
                return [{"MLBAM_GAME_ID": 823873, "PARAMETER_ROWS": 1}]
            if "SESSION_ID" in query and "PITCH_ID" in query:
                return [
                    {
                        "SESSION_ID": "session-1",
                        "PITCH_ID": "pitch-1",
                        "TEAM_NAME": "Miami Marlins",
                        "MLBAM_GAME_ID": 823873,
                        "MLBAM_PLAYER_ID": 686611,
                        "SESSION_DATE": "2026-05-07",
                    }
                ]
            raise AssertionError(f"Unexpected query: {query}")

    diagnostics = diagnose_batting_motion_pull(
        "guid-1",
        game_id=746020,
        connector=FakeConnector(),
    )

    assert diagnostics["parameter_set_rows"] == 1
    assert diagnostics["parameter_set_game_rows"] == 0
    assert diagnostics["joined_motion_rows"] == 12
    assert diagnostics["joined_motion_game_rows"] == 0
    assert diagnostics["available_games"] == [
        {"MLBAM_GAME_ID": 823873, "PARAMETER_ROWS": 1}
    ]

    formatted = format_empty_pull_diagnostics(diagnostics)
    assert "No Snowflake batting motion rows were returned." in formatted
    assert "Likely issue: the game_id filter does not match this GUID." in formatted
    assert "823873" in formatted


def test_normalize_mlbam_hitting_data_requires_canonical_ids() -> None:
    df = _minimal_hitting_rows().drop(columns=["MLBAM_GUID"])

    with pytest.raises(ValueError, match="MLBAM_GUID"):
        normalize_mlbam_hitting_data(df)


def test_normalize_mlbam_hitting_data_creates_frames_and_preserves_internal_keys() -> None:
    df = _minimal_hitting_rows().drop(columns=["FRAME"], errors="ignore")

    normalized = normalize_mlbam_hitting_data(df)

    assert normalized["FRAME"].tolist() == [1, 2, 3, 4]
    assert {"MLBAM_GUID", "MLBAM_GAME_ID", "MLBAM_PLAYER_ID", "SESSION_DATE"}.issubset(
        normalized.columns
    )
    assert {"SESSION_ID", "PITCH_ID"}.issubset(normalized.columns)


def test_normalize_mlbam_hitting_data_no_longer_requires_spine_rotation() -> None:
    spine_cols = [f"SPINE_R{row}{col}" for row in (1, 2, 3) for col in (1, 2, 3)]
    df = _minimal_hitting_rows().drop(columns=spine_cols)

    normalized = normalize_mlbam_hitting_data(df)

    assert normalized["FRAME"].tolist() == [1, 2, 3, 4]


def test_fixed_butterworth_uses_explicit_30_hz_cutoff() -> None:
    from biomech_functions import functions as funcs
    from scipy.signal import butter, filtfilt

    t = np.arange(60) / 300.0
    signal = np.sin(2 * np.pi * 5 * t) + 0.2 * np.sin(2 * np.pi * 90 * t)

    expected_b, expected_a = butter(4, 30.0, btype="low", analog=False, fs=300.0)
    expected = filtfilt(expected_b, expected_a, signal, axis=0)

    actual = funcs._fixed_lowpass_filter(signal)

    np.testing.assert_allclose(actual, expected)


def test_fixed_butterworth_short_series_falls_back_deterministically() -> None:
    from biomech_functions import functions as funcs

    short = np.array([0.0, 1.0, 0.0])

    actual = funcs._fixed_lowpass_filter(short)

    np.testing.assert_array_equal(actual, short)


def test_fixed_lowpass_filter_long_series_preserves_shape_and_filters() -> None:
    from biomech_functions import functions as funcs

    signal = np.sin(np.linspace(0, 4 * np.pi, 600))

    actual = funcs._fixed_lowpass_filter(signal)

    assert actual.shape == signal.shape
    assert not np.array_equal(actual, signal)


def test_bat_80_lcs_axes_and_first_temp_backfill() -> None:
    from biomech_functions import functions as funcs

    knob = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    top = knob + np.array([[0.0], [1.0], [0.0]])

    bat_80, r_bat80 = funcs._bat_80_lcs(knob, top)

    np.testing.assert_allclose(bat_80[:, 0], [0.0, 0.8, 0.0])
    np.testing.assert_allclose(r_bat80[:, :, 1], np.tile([0.0, 1.0, 0.0], (4, 1)))
    np.testing.assert_allclose(r_bat80[0, :, 0], r_bat80[1, :, 0])
    np.testing.assert_allclose(np.linalg.norm(r_bat80, axis=1), 1.0)
    np.testing.assert_allclose(
        np.einsum("fij,fik->fjk", r_bat80, r_bat80),
        np.tile(np.eye(3), (4, 1, 1)),
        atol=1e-12,
    )


def test_bat_80_local_ball_origin_and_3d_distance() -> None:
    from biomech_functions import functions as funcs

    knob = np.array(
        [
            [0.0, 1.0, 2.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    top = knob + np.array([[0.0], [1.0], [0.0]])
    bat_80, r_bat80 = funcs._bat_80_lcs(knob, top)
    ball = bat_80 + np.array([[-0.5], [0.2], [0.3]])

    ball_in_bat = funcs._global_to_bat_local(ball, bat_80, r_bat80)
    distance = funcs._local_selection_distance(ball_in_bat)

    np.testing.assert_allclose(ball_in_bat[:, 0], [-0.5, 0.2, -0.3])
    np.testing.assert_allclose(distance, np.sqrt(0.5**2 + 0.2**2 + 0.3**2))


def test_local_tmin_selection_uses_3d_distance_not_depth_only() -> None:
    from biomech_functions import functions as funcs

    miss_vector = np.array(
        [
            [0.00, 0.20, 0.30],
            [1.00, 0.01, 0.20],
            [1.00, 0.01, 0.20],
        ]
    )

    selection_distance = funcs._local_selection_distance(miss_vector)
    idx = funcs._minimum_index_in_window(selection_distance, 0, 2)

    assert int(np.nanargmin(np.abs(miss_vector[0]))) == 0
    assert idx == 1


def test_local_tmin_selection_uses_3d_distance_not_planar_only() -> None:
    from biomech_functions import functions as funcs

    miss_vector = np.array(
        [
            [0.50, 0.01, 0.30],
            [0.00, 0.10, 0.20],
            [0.00, 0.00, 0.20],
        ]
    )

    planar_distance = np.linalg.norm(miss_vector[[1, 2], :], axis=0)
    selection_distance = funcs._local_selection_distance(miss_vector)
    idx = funcs._minimum_index_in_window(selection_distance, 0, 2)

    assert int(np.nanargmin(planar_distance)) == 0
    assert idx == 1


def test_sweet_spot_zone_uses_local_yz_ellipse_with_overflow() -> None:
    from biomech_functions import functions as funcs

    assert funcs._sweet_spot_zone_check(local_y=0.0, local_z=0.0)
    assert funcs._sweet_spot_zone_check(local_y=0.06, local_z=0.0)
    assert funcs._sweet_spot_zone_check(local_y=0.0, local_z=0.04)
    assert not funcs._sweet_spot_zone_check(local_y=0.0, local_z=0.0401)
    assert not funcs._sweet_spot_zone_check(local_y=0.061, local_z=0.0)


def test_sweet_spot_origin_copies_k80_orientation_with_local_z_offset() -> None:
    from biomech_functions import functions as funcs

    normalized = normalize_mlbam_hitting_data(_minimal_hitting_rows(n_frames=18))
    knob_global = funcs._to_vec3(normalized, "KNOB_TX", "KNOB_TY", "KNOB_TZ")
    top_global = funcs._to_vec3(normalized, "TOP_TX", "TOP_TY", "TOP_TZ")
    frame_vec = funcs._frame_vector(normalized, len(normalized))
    bat_80_global, r_bat80 = funcs._bat_80_lcs(
        knob_global,
        top_global,
        handedness=funcs._infer_handedness_from_report(normalized),
        axis_backfill_indices=funcs._axis_backfill_indices(normalized, frame_vec),
    )

    sweet_spot_origin = funcs._sweet_spot_origin_global(bat_80_global, r_bat80)
    offset_local = np.stack(
        [
            r_bat80[frame_idx].T
            @ (sweet_spot_origin[:, frame_idx] - bat_80_global[:, frame_idx])
            for frame_idx in range(r_bat80.shape[0])
        ]
    )

    np.testing.assert_allclose(
        offset_local,
        np.tile(funcs.SWEET_SPOT_ORIGIN_LOCAL_OFFSET, (r_bat80.shape[0], 1)),
        atol=1e-12,
    )


def test_vertical_direction_flags_are_swing_relative_to_ball() -> None:
    from biomech_functions import functions as funcs

    positive_z_row: dict[str, object] = {}
    positive_z_miss = np.array([[0.0], [0.0], [0.25]])
    funcs._add_local_miss_direction_flags(positive_z_row, positive_z_miss, 0)

    assert positive_z_row["SWUNG_OVER"] == 1
    assert positive_z_row["SWUNG_UNDER"] == 0
    assert "OVER" not in positive_z_row
    assert "UNDER" not in positive_z_row

    negative_z_row: dict[str, object] = {}
    negative_z_miss = np.array([[0.0], [0.0], [-0.25]])
    funcs._add_local_miss_direction_flags(negative_z_row, negative_z_miss, 0)

    assert negative_z_row["SWUNG_OVER"] == 0
    assert negative_z_row["SWUNG_UNDER"] == 1


def test_discrete_schema_replaces_in_band_with_sweet_spot_zone() -> None:
    discrete_df, _ = _compute_fixture_outputs()

    assert "IN_SWEET_SPOT_ZONE" in discrete_df.columns
    assert "IN_BAND" not in discrete_df.columns


def test_report_r_l_flags_collapse_to_hitter_handedness() -> None:
    from biomech_functions import functions as funcs

    assert funcs._infer_hitter_handedness_from_report_flags(
        pd.DataFrame({"R": ["R"], "L": [0]})
    ) == "R"
    assert funcs._infer_hitter_handedness_from_report_flags(
        pd.DataFrame({"R": [0], "L": ["L"]})
    ) == "L"
    assert funcs._infer_hitter_handedness_from_report_flags(
        pd.DataFrame({"R": ["R"], "L": ["L"]})
    ) == "AMBIGUOUS"
    assert pd.isna(
        funcs._infer_hitter_handedness_from_report_flags(
            pd.DataFrame({"R": [0], "L": [""]})
        )
    )

    discrete_df, time_series = _compute_fixture_outputs()

    assert discrete_df.loc[0, "HITTER_HANDEDNESS"] == "R"
    assert "HITTER_HANDEDNESS" not in time_series.columns
    for raw_col in ("R", "L", "BAD", "HANDEDNESS"):
        assert raw_col not in discrete_df.columns
        assert raw_col not in time_series.columns


def test_bad_report_flag_flows_to_outcome_without_raw_bad_column() -> None:
    from biomech_functions import functions as funcs

    rows = _minimal_hitting_rows(n_frames=18)
    rows["BAD"] = "BAD"
    rows["SWING"] = np.nan
    rows["MISS"] = np.nan
    rows["BALL_CONTACT"] = np.nan
    rows["CHECK_SWING"] = np.nan

    normalized = normalize_mlbam_hitting_data(rows)
    discrete_df, time_series, outcome_counts = funcs.compute_discrete_and_time_series(
        normalized,
        group_id_cols=("MLBAM_GAME_ID", "MLBAM_GUID"),
        output_id_cols=("MLBAM_GAME_ID", "MLBAM_GUID", "MLBAM_PLAYER_ID", "SESSION_DATE"),
        save_validation_plots=False,
    )

    assert discrete_df.loc[0, "OUTCOME"] == "BAD"
    assert outcome_counts["BAD"] == 1
    assert "BAD" not in discrete_df.columns
    assert "BAD" not in time_series.columns


def test_no_report_trials_are_skipped_without_blank_outcome_rows() -> None:
    from biomech_functions import functions as funcs

    rows = _minimal_hitting_rows(n_frames=18)
    for col in ("BAD", "SWING", "MISS", "BALL_CONTACT", "CHECK_SWING", "TAKE"):
        rows[col] = np.nan

    normalized = normalize_mlbam_hitting_data(rows)
    discrete_df, time_series, outcome_counts = funcs.compute_discrete_and_time_series(
        normalized,
        group_id_cols=("MLBAM_GAME_ID", "MLBAM_GUID"),
        output_id_cols=("MLBAM_GAME_ID", "MLBAM_GUID", "MLBAM_PLAYER_ID", "SESSION_DATE"),
        save_validation_plots=False,
    )

    assert discrete_df.empty
    assert time_series.empty
    assert outcome_counts["NO_REPORT_skipped"] == 1
    assert "no_report" not in outcome_counts


def test_sweet_spot_zone_uses_local_tmin_for_miss_outcomes() -> None:
    from biomech_functions import functions as funcs

    discrete_df, time_series = _compute_fixture_outputs()

    local_tmin_frame = discrete_df.loc[0, "T_MIN_LOCAL"]
    tmin_row = time_series.loc[time_series["FRAME"] == local_tmin_frame].iloc[0]
    expected = int(
        funcs._sweet_spot_zone_check(
            local_y=float(tmin_row["BALL_IN_BAT_Y"]),
            local_z=float(tmin_row["BALL_IN_BAT_Z"]),
        )
    )

    assert discrete_df.loc[0, "OUTCOME"] == "MISS"
    assert discrete_df.loc[0, "IN_SWEET_SPOT_ZONE"] == expected


def test_discrete_schema_matches_snapshot() -> None:
    discrete_df, _ = _compute_fixture_outputs()

    expected = json.loads((FIXTURES_DIR / "expected_discrete_schema.json").read_text())

    assert discrete_df.columns.tolist() == expected


def test_time_series_schema_matches_snapshot() -> None:
    _, time_series = _compute_fixture_outputs()

    expected = json.loads((FIXTURES_DIR / "expected_time_series_schema.json").read_text())

    assert time_series.columns.tolist() == expected


def test_generated_columns_have_no_unit_suffix() -> None:
    discrete_df, time_series = _compute_fixture_outputs()

    _assert_generated_columns_unitless(discrete_df)
    _assert_generated_columns_unitless(time_series)


def test_compute_mlbam_mode_outputs_k80_bat_80_local_global_schema() -> None:
    from biomech_functions import functions as funcs

    normalized = normalize_mlbam_hitting_data(_minimal_hitting_rows(n_frames=18))

    metrics_df, time_series, _ = funcs.compute_discrete_and_time_series(
        normalized,
        group_id_cols=("MLBAM_GAME_ID", "MLBAM_GUID"),
        output_id_cols=("MLBAM_GAME_ID", "MLBAM_GUID", "MLBAM_PLAYER_ID", "SESSION_DATE"),
        save_validation_plots=False,
    )

    for output in (metrics_df, time_series):
        assert "MLBAM_GUID" in output.columns
        assert "MLBAM_GAME_ID" in output.columns
        assert "SESSION_ID" not in output.columns
        assert "PITCH_ID" not in output.columns
        _assert_retired_columns_absent(output)

    expected_metric_columns = {
        "MLBAM_GAME_ID",
        "MLBAM_GUID",
        "MLBAM_PLAYER_ID",
        "SESSION_DATE",
        "HITTER_HANDEDNESS",
        "MAX_BAT_SPEED_MPH",
        "T_MIN_GLOBAL",
        "T_MIN_LOCAL",
        "MISSED_DISTANCE_GLOBAL",
        "MISSED_DISTANCE_LOCAL",
        "KT_BALL_MIN_FRAME",
        "L_FOOT_START_X",
        "R_ANKLE_START_Z",
        "BALL_AT_TMIN_X",
        "BAT_KNOB_AT_TMIN_X",
        "BAT_TOP_AT_TMIN_Z",
        "SWEET_SPOT_ORIGIN_AT_TMIN_X",
        "SWEET_SPOT_ORIGIN_AT_TMIN_Y",
        "SWEET_SPOT_ORIGIN_AT_TMIN_Z",
        "MISS_VECTOR_GLOBAL_X",
        "MISS_VECTOR_GLOBAL_Y",
        "MISS_VECTOR_GLOBAL_Z",
        "MISS_VECTOR_LOCAL_X",
        "MISS_VECTOR_LOCAL_Y",
        "MISS_VECTOR_LOCAL_Z",
        "MISS_VELOCITY_GLOBAL_X",
        "MISS_VELOCITY_LOCAL_Z",
        "MAX_MISS_VELOCITY_LOCAL_X",
        "MAX_MISS_VELOCITY_GLOBAL_Z",
        "MAX_MISS_SPEED_LOCAL",
        "MAX_MISS_SPEED_GLOBAL",
        "MISS_SPEED_GLOBAL_AT_TMIN",
        "MISS_SPEED_LOCAL_AT_TMIN",
    }
    expected_time_series_columns = {
        "MLBAM_GAME_ID",
        "MLBAM_GUID",
        "FRAME",
        "BALL_X",
        "BALL_Y",
        "BALL_Z",
        "BALL_IN_BAT_X",
        "BALL_IN_BAT_Y",
        "BALL_IN_BAT_Z",
        "BAT_KNOB_X",
        "BAT_KNOB_Y",
        "BAT_KNOB_Z",
        "BAT_TOP_X",
        "BAT_TOP_Y",
        "BAT_TOP_Z",
        "K80_X",
        "K80_Y",
        "K80_Z",
        "SWEET_SPOT_ORIGIN_X",
        "SWEET_SPOT_ORIGIN_Y",
        "SWEET_SPOT_ORIGIN_Z",
        "MISS_VECTOR_GLOBAL_X",
        "MISS_VECTOR_GLOBAL_Z",
        "MISSED_DISTANCE_GLOBAL",
        "MISSED_DISTANCE_LOCAL",
        "MISS_VELOCITY_GLOBAL_X",
        "MISS_VELOCITY_LOCAL_Z",
        "MISS_SPEED_GLOBAL",
        "MISS_SPEED_LOCAL",
    }

    assert len(metrics_df) == 1
    assert expected_metric_columns.issubset(metrics_df.columns)
    assert expected_time_series_columns.issubset(time_series.columns)
    assert metrics_df.loc[0, "HITTER_HANDEDNESS"] == "R"
    assert metrics_df.loc[0, "MAX_BAT_SPEED_MPH"] == pytest.approx(73.5)
    assert "MAX_BAT_SPEED" not in metrics_df.columns
    assert not any("K82" in column for column in metrics_df.columns)
    assert not any("K80" in column for column in metrics_df.columns)
    assert not any("K82" in column for column in time_series.columns)
    assert {"K80_X", "K80_Y", "K80_Z"}.issubset(time_series.columns)
    for axis in ("X", "Y", "Z"):
        assert f"SWEET_SPOT_{axis}" not in time_series.columns
        assert f"SWEET_SPOT_AT_TMIN_{axis}" not in metrics_df.columns
    assert time_series[["MLBAM_GAME_ID", "MLBAM_GUID", "FRAME"]].duplicated().sum() == 0
    assert "BACK_FOOT_CG_POS_X" not in metrics_df.columns
    assert "BALL_MIN" not in metrics_df.columns
    assert "KT_BALL_MIN_FRAME" in metrics_df.columns
    old_speed_velocity_names = {
        "MISS_VECTOR_VELOCITY_LOCAL_X",
        "MISS_VECTOR_VELOCITY_LOCAL_Y",
        "MISS_VECTOR_VELOCITY_LOCAL_Z",
        "MISS_VECTOR_VELOCITY_GLOBAL_X",
        "MISS_VECTOR_VELOCITY_GLOBAL_Y",
        "MISS_VECTOR_VELOCITY_GLOBAL_Z",
        "MISSED_DISTANCE_LOCAL_SPEED",
        "MISSED_DISTANCE_GLOBAL_SPEED",
        "MISSED_DISTANCE_LOCAL_SPEED_AT_TMIN",
        "MISSED_DISTANCE_GLOBAL_SPEED_AT_TMIN",
    }
    assert old_speed_velocity_names.isdisjoint(metrics_df.columns)
    assert old_speed_velocity_names.isdisjoint(time_series.columns)
    assert not any(column.startswith("BALL_IN_BAT_AT_TMIN_") for column in metrics_df.columns)
    assert not any(column.startswith("BAT_TOP_IN_BAT_AT_TMIN_") for column in metrics_df.columns)
    assert not any(column.startswith("BAT_KNOB_IN_BAT_AT_TMIN_") for column in metrics_df.columns)
    assert not any(column.startswith("MISS_VECTOR_LOCAL_") for column in time_series.columns)
    assert not any(column.startswith("BAT_TOP_IN_BAT_") for column in time_series.columns)
    assert not any(column.startswith("BAT_KNOB_IN_BAT_") for column in time_series.columns)
    assert metrics_df.loc[0, "L_FOOT_START_Y"] == pytest.approx(0.011)
    assert metrics_df.columns.get_loc("L_FOOT_START_X") < metrics_df.columns.get_loc(
        "T_MIN_LOCAL"
    )
    assert metrics_df.columns.get_loc("T_MIN_GLOBAL") < metrics_df.columns.get_loc(
        "BALL_AT_TMIN_X"
    )
    assert metrics_df.columns.get_loc("BAT_TOP_AT_TMIN_Z") < metrics_df.columns.get_loc(
        "MISS_VECTOR_LOCAL_X"
    )
    assert metrics_df.columns.get_loc("SWEET_SPOT_ORIGIN_AT_TMIN_Z") < metrics_df.columns.get_loc(
        "MISS_VECTOR_LOCAL_X"
    )
    assert metrics_df.columns.get_loc("MISS_VECTOR_LOCAL_X") < metrics_df.columns.get_loc(
        "MISS_VECTOR_GLOBAL_X"
    )
    assert time_series.columns.get_loc("BALL_Z") < time_series.columns.get_loc(
        "BALL_IN_BAT_X"
    )
    assert time_series.columns.get_loc("BALL_IN_BAT_Z") < time_series.columns.get_loc(
        "BAT_KNOB_X"
    )
    assert time_series.columns.get_loc("BAT_TOP_Z") < time_series.columns.get_loc(
        "K80_X"
    )
    assert time_series.columns.get_loc("K80_Z") < time_series.columns.get_loc(
        "SWEET_SPOT_ORIGIN_X"
    )
    assert time_series.columns.get_loc("SWEET_SPOT_ORIGIN_Z") < time_series.columns.get_loc(
        "MISS_VECTOR_GLOBAL_X"
    )
    pulled_unit_columns = {"MAX_BAT_SPEED_MPH"}
    for output in (metrics_df, time_series):
        _assert_position_columns_are_space_neutral(output)
        generated_unit_columns = [
            column for column in output.columns if column.endswith(("_M", "_MPS", "_MPH"))
            and column not in pulled_unit_columns
        ]
        assert generated_unit_columns == []

    assert time_series.columns.get_loc("MISS_VELOCITY_LOCAL_Z") < time_series.columns.get_loc(
        "MISS_VELOCITY_GLOBAL_X"
    )
    assert time_series.columns.get_loc("MISS_SPEED_LOCAL") < time_series.columns.get_loc(
        "LEFTFOOT_TX"
    )

    velocity_columns = [
        column for column in time_series.columns if "VELOCITY" in column or "SPEED" in column
    ]
    assert np.isfinite(time_series[velocity_columns].to_numpy(dtype=float)).all()
    for space in ("LOCAL", "GLOBAL"):
        np.testing.assert_allclose(
            time_series[f"MISS_SPEED_{space}"].to_numpy(float),
            np.linalg.norm(
                time_series[
                    [
                        f"MISS_VELOCITY_{space}_X",
                        f"MISS_VELOCITY_{space}_Y",
                        f"MISS_VELOCITY_{space}_Z",
                    ]
                ].to_numpy(float),
                axis=1,
            ),
        )

    ball_global = funcs._to_vec3(normalized, "CENTER_TX", "CENTER_TY", "CENTER_TZ")
    knob_global = funcs._to_vec3(normalized, "KNOB_TX", "KNOB_TY", "KNOB_TZ")
    top_global = funcs._to_vec3(normalized, "TOP_TX", "TOP_TY", "TOP_TZ")
    frame_vec = funcs._frame_vector(normalized, ball_global.shape[1])
    bat_80_global, r_bat80 = funcs._bat_80_lcs(
        knob_global,
        top_global,
        handedness=funcs._infer_handedness_from_report(normalized),
        axis_backfill_indices=funcs._axis_backfill_indices(normalized, frame_vec),
    )
    sweet_spot_origin = funcs._sweet_spot_origin_global(bat_80_global, r_bat80)
    expected_ball_in_bat = funcs._global_to_bat_local(
        ball_global,
        sweet_spot_origin,
        r_bat80,
    )

    np.testing.assert_allclose(
        time_series[["K80_X", "K80_Y", "K80_Z"]].to_numpy(float).T,
        bat_80_global,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        time_series[
            [
                "SWEET_SPOT_ORIGIN_X",
                "SWEET_SPOT_ORIGIN_Y",
                "SWEET_SPOT_ORIGIN_Z",
            ]
        ].to_numpy(float).T,
        sweet_spot_origin,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        time_series[["BALL_IN_BAT_X", "BALL_IN_BAT_Y", "BALL_IN_BAT_Z"]].to_numpy(float).T,
        expected_ball_in_bat,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        time_series[["MISS_VECTOR_GLOBAL_X", "MISS_VECTOR_GLOBAL_Y", "MISS_VECTOR_GLOBAL_Z"]].to_numpy(float),
        time_series[["BALL_X", "BALL_Y", "BALL_Z"]].to_numpy(float)
        - time_series[
            [
                "SWEET_SPOT_ORIGIN_X",
                "SWEET_SPOT_ORIGIN_Y",
                "SWEET_SPOT_ORIGIN_Z",
            ]
        ].to_numpy(float),
        atol=1e-12,
    )

    local_window = time_series[
        (time_series["FRAME"] >= 3) & (time_series["FRAME"] <= 16)
    ]
    local_selection_distance = np.linalg.norm(
        local_window[
            [
                "BALL_IN_BAT_X",
                "BALL_IN_BAT_Y",
                "BALL_IN_BAT_Z",
            ]
        ].to_numpy(float),
        axis=1,
    )
    local_tmin_idx = local_window.index[int(np.nanargmin(local_selection_distance))]
    local_tmin_frame = time_series.loc[local_tmin_idx, "FRAME"]
    assert metrics_df.loc[0, "T_MIN_LOCAL"] == local_tmin_frame
    global_tmin_frame = metrics_df.loc[0, "T_MIN_GLOBAL"]
    global_tmin_row = time_series.loc[time_series["FRAME"] == global_tmin_frame].iloc[0]
    assert metrics_df.loc[0, "MISS_SPEED_LOCAL_AT_TMIN"] == pytest.approx(
        time_series.loc[local_tmin_idx, "MISS_SPEED_LOCAL"]
    )
    assert metrics_df.loc[0, "MISS_SPEED_GLOBAL_AT_TMIN"] == pytest.approx(
        global_tmin_row["MISS_SPEED_GLOBAL"]
    )

    local_3d_distance = np.linalg.norm(
        time_series.loc[
            local_tmin_idx,
            [
                "BALL_IN_BAT_X",
                "BALL_IN_BAT_Y",
                "BALL_IN_BAT_Z",
            ],
        ].to_numpy(float)
    )
    assert metrics_df.loc[0, "MISSED_DISTANCE_LOCAL"] == pytest.approx(
        local_3d_distance
    )
    assert metrics_df.loc[0, "MISSED_DISTANCE_LOCAL"] == pytest.approx(
        metrics_df.loc[0, "MISSED_DISTANCE_GLOBAL"]
    )


def test_outcome_counts_written_as_root_csv(tmp_path: Path) -> None:
    from missed_distance import _write_outcome_counts_csv

    path = tmp_path / "831467_outcome_counts.csv"

    _write_outcome_counts_csv({"BALL_CONTACT": 2, "MISS": 1}, path)

    output = pd.read_csv(path)
    assert list(output.columns) == ["OUTCOME", "COUNT"]
    assert output.to_dict("records") == [
        {"OUTCOME": "BALL_CONTACT", "COUNT": 2},
        {"OUTCOME": "MISS", "COUNT": 1},
    ]


def test_cli_writes_canonical_discrete_time_series_and_counts_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_csv = FIXTURES_DIR / "minimal_motion.csv"

    _run_cli(
        monkeypatch,
        [
            "--input-csv",
            input_csv,
            "--results-dir",
            tmp_path,
            "--skip-validation-plots",
        ],
    )

    assert (tmp_path / "discrete" / "minimal_motion_discrete.csv").exists()
    assert (tmp_path / "time_series" / "minimal_motion_time_series.csv").exists()
    counts_path = tmp_path / "minimal_motion_outcome_counts.csv"
    assert counts_path.exists()
    assert list(pd.read_csv(counts_path).columns) == ["OUTCOME", "COUNT"]
    assert not (tmp_path / "minimal_motion_validation.json").exists()
    assert not (tmp_path / "minimal_motion_validation_summary.csv").exists()


def test_cli_loads_game_cache_before_snowflake(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import modules.data_service as data_service

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Snowflake should not be contacted when game cache exists.")

    data_dir = tmp_path / "data"
    cache_dir = data_dir / "746020"
    cache_dir.mkdir(parents=True)
    pd.read_csv(FIXTURES_DIR / "minimal_motion.csv").to_csv(
        cache_dir / "motion_sequence.csv",
        index=False,
    )

    monkeypatch.setattr(data_service, "get_batting_motion", fail_if_called)
    monkeypatch.setattr(data_service, "get_batting_hitting_report", fail_if_called)
    monkeypatch.setattr(data_service, "diagnose_batting_motion_pull", fail_if_called)

    _run_cli(
        monkeypatch,
        [
            "--game-id",
            "746020",
            "--data-dir",
            data_dir,
            "--results-dir",
            tmp_path / "results",
            "--skip-validation-plots",
        ],
    )

    assert (tmp_path / "results" / "discrete" / "746020_discrete.csv").exists()
    assert (tmp_path / "results" / "time_series" / "746020_time_series.csv").exists()
    assert (tmp_path / "results" / "746020_outcome_counts.csv").exists()
    assert not (tmp_path / "results" / "746020_validation.json").exists()
    assert not (tmp_path / "results" / "746020_validation_summary.csv").exists()


def test_validation_plots_use_game_guid_root(monkeypatch: pytest.MonkeyPatch) -> None:
    from biomech_functions import functions as funcs

    normalized = normalize_mlbam_hitting_data(_minimal_hitting_rows(n_frames=18))
    calls: list[tuple[Path, str]] = []

    def capture_plot(*args, **kwargs):
        calls.append((Path(kwargs["out_dir"]), kwargs["pitch_id"]))

    monkeypatch.setattr(funcs, "_plot_md_validation", capture_plot)
    monkeypatch.setattr(funcs, "_plot_md_validation_3d", capture_plot)

    funcs.compute_discrete_and_time_series(
        normalized,
        group_id_cols=("MLBAM_GAME_ID", "MLBAM_GUID"),
        output_id_cols=("MLBAM_GAME_ID", "MLBAM_GUID", "MLBAM_PLAYER_ID", "SESSION_DATE"),
        save_validation_plots=True,
    )

    assert calls
    assert {out_dir for out_dir, _ in calls} == {
        Path("fig_outputs/MLBAM_GAME_GUID_MD_VALIDATION/SWEET_SPOT_ORIGIN"),
    }
    assert {plot_id for _, plot_id in calls} == {"746020_guid-1"}


def test_validation_events_use_bat_stop_instead_of_end_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from biomech_functions import functions as funcs

    normalized = normalize_mlbam_hitting_data(_minimal_hitting_rows(n_frames=18))
    captured_events: list[dict[str, float]] = []

    def capture_plot(*args, **kwargs):
        captured_events.append(kwargs["events"])

    monkeypatch.setattr(funcs, "_plot_md_validation", capture_plot)
    monkeypatch.setattr(funcs, "_plot_md_validation_3d", capture_plot)

    funcs.compute_discrete_and_time_series(
        normalized,
        group_id_cols=("MLBAM_GAME_ID", "MLBAM_GUID"),
        output_id_cols=("MLBAM_GAME_ID", "MLBAM_GUID", "MLBAM_PLAYER_ID", "SESSION_DATE"),
        save_validation_plots=True,
    )

    assert captured_events
    for events in captured_events:
        assert events["BALL_START"] == 0
        assert events["BAT_STOP"] == 16
        assert "END_DATA" not in events


def test_raw_snowflake_data_writes_under_game_folder(tmp_path: Path) -> None:
    from missed_distance import _write_raw_snowflake_data

    motion_df = _minimal_hitting_rows(n_frames=2)
    motion_df.loc[1, "MLBAM_GUID"] = "guid-2"
    report_df = motion_df[["MLBAM_GAME_ID", "MLBAM_GUID", "SESSION_ID", "PITCH_ID"]].head(1)

    paths = _write_raw_snowflake_data(
        motion_df,
        report_df,
        data_dir=tmp_path,
        game_label="746020",
    )

    assert paths["motion sequence"] == tmp_path / "746020" / "motion_sequence.csv"
    assert paths["hitting report"] == tmp_path / "746020" / "hitting_report.csv"
    assert paths["motion sequence"].exists()
    assert paths["hitting report"].exists()
    assert pd.read_csv(paths["motion sequence"])["MLBAM_GUID"].tolist() == ["guid-1", "guid-2"]
