from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class OutputSchema:
    """Canonical output column ordering for discrete and time-series files."""

    axes: tuple[str, ...]
    sweet_spots: tuple[str, ...]
    spaces_local_global: tuple[str, ...]
    spaces_global_local: tuple[str, ...]
    metadata: tuple[str, ...]
    local_miss_direction_flags: tuple[str, ...]
    tags: tuple[str, ...]
    events: tuple[str, ...]

    def metric_keys(self, spot: str, space: str) -> list[str]:
        keys = [
            f"T_MIN_{space}",
            f"MISSED_DISTANCE_{space}",
        ]
        keys += [f"MISS_VECTOR_{space}_{axis}" for axis in self.axes]
        keys += [f"MISS_VELOCITY_{space}_{axis}" for axis in self.axes]
        keys += [
            f"MISS_SPEED_{space}_AT_TMIN",
        ]

        if space == "GLOBAL":
            keys += [f"BALL_AT_TMIN_{axis}" for axis in self.axes]
            keys += [f"BAT_KNOB_AT_TMIN_{axis}" for axis in self.axes]
            keys += [f"BAT_TOP_AT_TMIN_{axis}" for axis in self.axes]
            keys += [f"SWEET_SPOT_ORIGIN_AT_TMIN_{axis}" for axis in self.axes]
        return keys

    def time_series_columns(
        self,
        group_id_cols: Sequence[str],
        foot_ankle_cols: Iterable[str],
    ) -> list[str]:
        desired: list[str] = [*group_id_cols, "FRAME"]

        desired += [f"BALL_{axis}" for axis in self.axes]
        desired += [f"BALL_IN_BAT_{axis}" for axis in self.axes]
        desired += [f"BAT_KNOB_{axis}" for axis in self.axes]
        desired += [f"BAT_TOP_{axis}" for axis in self.axes]

        desired += [f"K80_{axis}" for axis in self.axes]
        desired += [f"SWEET_SPOT_ORIGIN_{axis}" for axis in self.axes]

        desired += [f"MISS_VECTOR_GLOBAL_{axis}" for axis in self.axes]

        for space in self.spaces_local_global:
            desired.append(f"MISSED_DISTANCE_{space}")

        for space in self.spaces_local_global:
            desired += [f"MISS_VELOCITY_{space}_{axis}" for axis in self.axes]

        for space in self.spaces_local_global:
            desired.append(f"MISS_SPEED_{space}")

        desired += list(foot_ankle_cols)
        return desired

    def discrete_front_columns(
        self,
        start_position_cols: Iterable[str],
    ) -> list[str]:
        event_cols = [col for col in self.events if not col.startswith("KT_BALL_MIN")]
        ball_min_cols = [col for col in self.events if col.startswith("KT_BALL_MIN")]

        tmin_cols: list[str] = []
        for space in self.spaces_local_global:
            tmin_cols.append(f"T_MIN_{space}")

        tmin_position_cols: list[str] = []
        tmin_position_cols += [
            f"BALL_AT_TMIN_{axis}" for axis in self.axes
        ]
        tmin_position_cols += [
            f"BAT_KNOB_AT_TMIN_{axis}" for axis in self.axes
        ]
        tmin_position_cols += [
            f"BAT_TOP_AT_TMIN_{axis}" for axis in self.axes
        ]
        tmin_position_cols += [
            f"SWEET_SPOT_ORIGIN_AT_TMIN_{axis}" for axis in self.axes
        ]

        miss_vector_cols: list[str] = []
        for space in self.spaces_local_global:
            miss_vector_cols += [f"MISS_VECTOR_{space}_{axis}" for axis in self.axes]

        distance_cols: list[str] = []
        for space in self.spaces_local_global:
            distance_cols.append(f"MISSED_DISTANCE_{space}")

        max_velocity_cols: list[str] = []
        for space in self.spaces_local_global:
            max_velocity_cols += [
                f"MAX_MISS_VELOCITY_{space}_{axis}"
                for axis in self.axes
            ]

        max_speed_cols: list[str] = []
        for space in self.spaces_local_global:
            max_speed_cols.append(f"MAX_MISS_SPEED_{space}")

        return [
            *self.metadata,
            *self.local_miss_direction_flags,
            *self.tags,
            *event_cols,
            *list(start_position_cols),
            *ball_min_cols,
            *tmin_cols,
            *tmin_position_cols,
            *miss_vector_cols,
            *distance_cols,
            *max_velocity_cols,
            *max_speed_cols,
        ]


OUTPUT_SCHEMA = OutputSchema(
    axes=("X", "Y", "Z"),
    sweet_spots=("K80",),
    spaces_local_global=("LOCAL", "GLOBAL"),
    spaces_global_local=("GLOBAL", "LOCAL"),
    metadata=(
        "MLBAM_GAME_ID",
        "MLBAM_GUID",
        "SESSION_DATE",
        "SESSION_ID",
        "SESSION_TIMESTAMP",
        "PITCH_ID",
        "PITCH_TIMESTAMP",
        "MLBAM_PLAYER_ID",
        "PLAYER_JERSEY_NUMBER",
        "TEAM_NAME",
        "BATTER_NAME",
        "HITTER_HANDEDNESS",
        "HEIGHT",
        "MASS",
        "MAX_BAT_SPEED_MPH",
        "BALL_PITCH_VELOCITY",
        "KT_DATA_TYPE",
        "GCS_PATH",
        "CREATED_AT",
        "ITEM",
    ),
    local_miss_direction_flags=(
        "CAPPED",
        "JAMMED",
        "SWUNG_OVER",
        "SWUNG_UNDER",
    ),
    tags=(
        "OUTCOME",
        "IN_SWEET_SPOT_ZONE",
    ),
    events=(
        "LOAD",
        "SETUP",
        "START_DATA",
        "DOWNSWING",
        "DS",
        "BALL_START",
        "BAT_START",
        "KT_BALL_MIN_FRAME",
        "KT_BALL_MIN_DIST_MISS",
        "BAT_STOP",
    ),
)
