from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from threading import Lock
from typing import Any, Protocol

import pandas as pd

logger = logging.getLogger(__name__)


class QueryExecutor(Protocol):
    """
    Minimal interface expected from any query backend.
    Both the built-in SnowflakeConnector adapter and any test mock must
    implement exactly this one method.
    """

    def execute_query_cached(self, query: str, params: dict[str, object]) -> Any:
        """Execute a parameterized query and return tabular rows."""


def build_batting_motion_query(
    guid: str | None = None,
    *,
    game_id: int | str | None = None,
) -> tuple[str, dict[str, object]]:
    """Build the enriched batting motion query and parameter dictionary."""
    params: dict[str, object] = {}
    filters: list[str] = []

    if guid is not None:
        filters.append("pps.MLBAM_GUID = %(guid)s")
        params["guid"] = guid
    if game_id is not None:
        filters.append("pps.MLBAM_GAME_ID = %(game_id)s")
        params["game_id"] = game_id
    if not filters:
        raise ValueError("At least one of guid or game_id is required.")

    where_clause = " AND ".join(filters)

    query = f"""
SELECT pps.*,
       pr.*
FROM   KINATRAX.BATTING_MOTION_SEQUENCE_BATTING AS pr
INNER JOIN KINATRAX.BATTING_PARAMETER_SET AS pps
  ON  pr.SESSION_ID = pps.SESSION_ID
 AND  pr.PITCH_ID   = pps.PITCH_ID
 AND  pr.TEAM_NAME  = pps.TEAM_NAME
WHERE {where_clause}
ORDER BY pps.MLBAM_GAME_ID ASC, pps.MLBAM_GUID ASC, pr.TIMESTAMP ASC
""".strip()

    return query, params


def _deduplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate column names produced by SELECT pps.*, pr.* joins.

    Keeps the first occurrence of each column (from the pps/parameter-set
    side) and drops later duplicates (from the pr/motion or report side).
    """
    seen: set[str] = set()
    keep: list[bool] = []
    for col in df.columns:
        if col in seen:
            keep.append(False)
        else:
            seen.add(col)
            keep.append(True)
    if not all(keep):
        dropped = [c for c, k in zip(df.columns, keep) if not k]
        logger.info("Dropped %d duplicate columns: %s", len(dropped), dropped)
        return df.loc[:, keep]
    return df


def get_batting_motion(
    guid: str | None = None,
    *,
    game_id: int | str | None = None,
    connector: QueryExecutor | None = None,
) -> pd.DataFrame:
    """Fetch enriched batting motion rows for a game, GUID, or game/GUID pair."""
    query, params = build_batting_motion_query(guid, game_id=game_id)
    executor = connector if connector is not None else _load_default_connector()
    rows = executor.execute_query_cached(query, params)

    df = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    return _deduplicate_columns(df)


def build_batting_hitting_report_query(
    guid: str | None = None,
    *,
    game_id: int | str | None = None,
) -> tuple[str, dict[str, object]]:
    """
    Build the per-pitch hitting-report query.

    Joins KINATRAX.BATTING_REPORTS (report-level metrics) to
    KINATRAX.BATTING_PARAMETER_SET (identity/metadata) so the output
    contains the full report row enriched with MLBAM IDs.
    """
    params: dict[str, object] = {}
    filters: list[str] = []

    if guid is not None:
        filters.append("pps.MLBAM_GUID = %(guid)s")
        params["guid"] = guid
    if game_id is not None:
        filters.append("pps.MLBAM_GAME_ID = %(game_id)s")
        params["game_id"] = game_id
    if not filters:
        raise ValueError("At least one of guid or game_id is required.")

    where_clause = " AND ".join(filters)
    query = f"""
SELECT pps.*,
       pr.*
FROM   KINATRAX.BATTING_REPORTS AS pr
INNER JOIN KINATRAX.BATTING_PARAMETER_SET AS pps
  ON  pr.SESSION_ID = pps.SESSION_ID
 AND  pr.PITCH_ID   = pps.PITCH_ID
 AND  pr.TEAM_NAME  = pps.TEAM_NAME
WHERE  {where_clause}
ORDER BY pps.MLBAM_GAME_ID ASC, pps.MLBAM_GUID ASC, pps.SESSION_DATE ASC, pps.SESSION_ID ASC, pps.PITCH_ID ASC
""".strip()
    return query, params


def get_batting_hitting_report(
    guid: str | None = None,
    *,
    game_id: int | str | None = None,
    connector: QueryExecutor | None = None,
) -> pd.DataFrame:
    """Fetch per-pitch metadata/report rows for a game, GUID, or game/GUID pair."""
    query, params = build_batting_hitting_report_query(guid, game_id=game_id)
    executor = connector if connector is not None else _load_default_connector()
    rows = executor.execute_query_cached(query, params)

    df = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    return _deduplicate_columns(df)


def diagnose_batting_motion_pull(
    guid: str,
    *,
    game_id: int | str | None = None,
    connector: QueryExecutor | None = None,
) -> dict[str, object]:
    """Return row-count diagnostics for an empty batting motion pull."""
    executor = connector if connector is not None else _load_default_connector()
    params: dict[str, object] = {"guid": guid}
    if game_id is not None:
        params["game_id"] = game_id

    diagnostics: dict[str, object] = {
        "guid": guid,
        "game_id": game_id,
    }

    diagnostics["parameter_set_rows"] = _scalar_query(
        executor,
        """
SELECT COUNT(*) AS PARAMETER_SET_ROWS
FROM   KINATRAX.BATTING_PARAMETER_SET
WHERE  MLBAM_GUID = %(guid)s
""".strip(),
        {"guid": guid},
        "PARAMETER_SET_ROWS",
    )

    if game_id is not None:
        diagnostics["parameter_set_game_rows"] = _scalar_query(
            executor,
            """
SELECT COUNT(*) AS PARAMETER_SET_GAME_ROWS
FROM   KINATRAX.BATTING_PARAMETER_SET
WHERE  MLBAM_GUID = %(guid)s
  AND  MLBAM_GAME_ID = %(game_id)s
""".strip(),
            params,
            "PARAMETER_SET_GAME_ROWS",
        )

    diagnostics["joined_motion_rows"] = _scalar_query(
        executor,
        """
SELECT COUNT(*) AS JOINED_MOTION_ROWS
FROM   KINATRAX.BATTING_MOTION_SEQUENCE_BATTING AS pr
INNER JOIN KINATRAX.BATTING_PARAMETER_SET AS pps
  ON  pr.SESSION_ID = pps.SESSION_ID
 AND  pr.PITCH_ID   = pps.PITCH_ID
 AND  pr.TEAM_NAME  = pps.TEAM_NAME
WHERE pps.MLBAM_GUID = %(guid)s
""".strip(),
        {"guid": guid},
        "JOINED_MOTION_ROWS",
    )

    if game_id is not None:
        diagnostics["joined_motion_game_rows"] = _scalar_query(
            executor,
            """
SELECT COUNT(*) AS JOINED_MOTION_GAME_ROWS
FROM   KINATRAX.BATTING_MOTION_SEQUENCE_BATTING AS pr
INNER JOIN KINATRAX.BATTING_PARAMETER_SET AS pps
  ON  pr.SESSION_ID = pps.SESSION_ID
 AND  pr.PITCH_ID   = pps.PITCH_ID
 AND  pr.TEAM_NAME  = pps.TEAM_NAME
WHERE pps.MLBAM_GUID = %(guid)s
  AND pps.MLBAM_GAME_ID = %(game_id)s
""".strip(),
            params,
            "JOINED_MOTION_GAME_ROWS",
        )
        diagnostics["joined_motion_game_rows_without_team_name"] = _scalar_query(
            executor,
            """
SELECT COUNT(*) AS JOINED_MOTION_GAME_ROWS_WITHOUT_TEAM
FROM   KINATRAX.BATTING_MOTION_SEQUENCE_BATTING AS pr
INNER JOIN KINATRAX.BATTING_PARAMETER_SET AS pps
  ON  pr.SESSION_ID = pps.SESSION_ID
 AND  pr.PITCH_ID   = pps.PITCH_ID
WHERE pps.MLBAM_GUID = %(guid)s
  AND pps.MLBAM_GAME_ID = %(game_id)s
""".strip(),
            params,
            "JOINED_MOTION_GAME_ROWS_WITHOUT_TEAM",
        )

    diagnostics["available_games"] = _query_records(
        executor,
        """
SELECT MLBAM_GAME_ID, COUNT(*) AS PARAMETER_ROWS
FROM   KINATRAX.BATTING_PARAMETER_SET
WHERE  MLBAM_GUID = %(guid)s
GROUP BY MLBAM_GAME_ID
ORDER BY PARAMETER_ROWS DESC
LIMIT 10
""".strip(),
        {"guid": guid},
    )
    diagnostics["parameter_set_samples"] = _query_records(
        executor,
        """
SELECT SESSION_ID,
       PITCH_ID,
       TEAM_NAME,
       MLBAM_GAME_ID,
       MLBAM_PLAYER_ID,
       SESSION_DATE
FROM   KINATRAX.BATTING_PARAMETER_SET
WHERE  MLBAM_GUID = %(guid)s
ORDER BY SESSION_DATE DESC
LIMIT 10
""".strip(),
        {"guid": guid},
    )

    return diagnostics


def format_empty_pull_diagnostics(diagnostics: dict[str, object]) -> str:
    """Format empty Snowflake batting-pull diagnostics for logs and CLI output."""
    guid = diagnostics.get("guid")
    game_id = diagnostics.get("game_id")
    lines = [
        "No Snowflake batting motion rows were returned.",
        f"GUID: {guid}",
        f"Game ID filter: {game_id if game_id is not None else '(none)'}",
        "",
        "Diagnostics:",
        f"  Parameter-set rows for GUID: {diagnostics.get('parameter_set_rows', 0)}",
    ]
    if game_id is not None:
        lines.append(
            "  Parameter-set rows for GUID + game_id: "
            f"{diagnostics.get('parameter_set_game_rows', 0)}"
        )
    lines.append(f"  Joined motion rows for GUID: {diagnostics.get('joined_motion_rows', 0)}")
    if game_id is not None:
        lines.extend(
            [
                "  Joined motion rows for GUID + game_id: "
                f"{diagnostics.get('joined_motion_game_rows', 0)}",
                "  Joined motion rows without TEAM_NAME join: "
                f"{diagnostics.get('joined_motion_game_rows_without_team_name', 0)}",
            ]
        )

    available_games = diagnostics.get("available_games") or []
    if available_games:
        lines.extend(["", "Available MLBAM_GAME_ID values for this GUID:"])
        for row in available_games:
            lines.append(
                f"  {row.get('MLBAM_GAME_ID')} "
                f"({row.get('PARAMETER_ROWS')} parameter rows)"
            )

    samples = diagnostics.get("parameter_set_samples") or []
    if samples:
        lines.extend(["", "Parameter-set samples for this GUID:"])
        for row in samples[:5]:
            lines.append(
                "  "
                f"GAME={row.get('MLBAM_GAME_ID')} "
                f"SESSION={row.get('SESSION_ID')} "
                f"PITCH={row.get('PITCH_ID')} "
                f"TEAM={row.get('TEAM_NAME')} "
                f"DATE={row.get('SESSION_DATE')}"
            )

    lines.append("")
    if diagnostics.get("parameter_set_rows", 0) == 0:
        lines.append("Likely issue: the GUID was not found in KINATRAX.BATTING_PARAMETER_SET.")
    elif game_id is not None and diagnostics.get("parameter_set_game_rows", 0) == 0:
        lines.extend(
            [
                "Likely issue: the game_id filter does not match this GUID.",
                "Try rerunning without --game-id, or use one of the available game IDs above.",
            ]
        )
    elif diagnostics.get("joined_motion_rows", 0) == 0:
        lines.append(
            "Likely issue: the GUID exists in BATTING_PARAMETER_SET, but no motion "
            "frames joined on SESSION_ID/PITCH_ID/TEAM_NAME."
        )
    elif (
        game_id is not None
        and diagnostics.get("joined_motion_game_rows", 0) == 0
        and diagnostics.get("joined_motion_game_rows_without_team_name", 0) > 0
    ):
        lines.append("Likely issue: TEAM_NAME differs between the motion and parameter tables.")
    else:
        lines.append("Likely issue: the GUID/game pair exists, but the final enriched motion query is empty.")

    return "\n".join(lines)


def _query_dataframe(
    executor: QueryExecutor,
    query: str,
    params: dict[str, object],
) -> pd.DataFrame:
    rows = executor.execute_query_cached(query, params)
    if isinstance(rows, pd.DataFrame):
        return rows.copy()
    return pd.DataFrame(rows)


def _scalar_query(
    executor: QueryExecutor,
    query: str,
    params: dict[str, object],
    column: str,
) -> int:
    df = _query_dataframe(executor, query, params)
    if df.empty or column not in df.columns:
        return 0
    value = df[column].iloc[0]
    if pd.isna(value):
        return 0
    return int(value)


def _query_records(
    executor: QueryExecutor,
    query: str,
    params: dict[str, object],
) -> list[dict[str, object]]:
    df = _query_dataframe(executor, query, params)
    return df.to_dict("records")


# ---------------------------------------------------------------------------
# In-memory query cache shared across all connector instances.
# Keyed by MD5(sql + params); evicts oldest entry when the cap is reached.
# ---------------------------------------------------------------------------
_QUERY_CACHE: dict[str, tuple[float, pd.DataFrame]] = {}
_CACHE_MAX_SIZE = 50
_CACHE_TTL_SECONDS = 300  # 5 minutes


def _build_cache_key(sql: str, params: dict[str, object]) -> str:
    """Deterministic MD5 key derived from the query text and its bind params."""
    payload = json.dumps({"sql": sql, "params": params}, sort_keys=True)
    return hashlib.md5(payload.encode()).hexdigest()


class _SnowflakeConnector:
    """
    Thread-safe Snowflake connector that satisfies the QueryExecutor protocol.

    Mirrors the production pattern used in the Biomechanics Viewer project:
      - Singleton connection pool (one live connection per process)
      - Stale-connection detection with automatic reconnection
      - SSO / externalbrowser auth by default (no password required)
      - Retry with exponential backoff for transient network errors
      - In-memory LRU query cache (5-minute TTL, 50-entry cap)

    Authentication modes (set SNOWFLAKE_AUTHENTICATOR in .env):
      'externalbrowser'  — opens browser for Okta/SSO login (default, no password)
      'snowflake'        — standard username + password
      '<okta-url>'       — native Okta endpoint (password required)

    Required env vars:
        SNOWFLAKE_ACCOUNT    — e.g. gs09656.us-east4.gcp
        SNOWFLAKE_USER       — login email / username
        SNOWFLAKE_WAREHOUSE  — virtual warehouse name
        SNOWFLAKE_DATABASE   — target database
        SNOWFLAKE_SCHEMA     — default schema

    Optional env vars:
        SNOWFLAKE_PASSWORD      — only needed for non-SSO auth modes
        SNOWFLAKE_ROLE          — session role (default: PUBLIC)
        SNOWFLAKE_AUTHENTICATOR — auth mode (default: externalbrowser)
    """

    # Singleton state — one connection shared across all instantiations.
    _instance: "_SnowflakeConnector | None" = None
    _class_lock: Lock = Lock()

    def __new__(cls) -> "_SnowflakeConnector":
        """Ensure only one connector instance exists per process."""
        with cls._class_lock:
            if cls._instance is None:
                obj = super().__new__(cls)
                obj._conn = None
                obj._conn_lock = Lock()
                obj._initialized = False
                cls._instance = obj
        return cls._instance

    def __init__(self) -> None:
        # __init__ is called every time even for a reused singleton; guard
        # against double-initialisation with the flag set in __new__.
        if self._initialized:
            return
        self._config = self._read_config()
        self._initialized = True
        logger.info(
            "SnowflakeConnector ready (account=%s, warehouse=%s, auth=%s)",
            self._config["account"],
            self._config["warehouse"],
            self._config["authenticator"],
        )

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    @staticmethod
    def _read_config() -> dict[str, str]:
        """
        Read connection settings from environment variables.

        Only validates the fields required by every auth mode; password is
        validated separately inside _open_connection() because it is only
        mandatory for non-SSO flows.
        """
        required_vars = (
            "SNOWFLAKE_ACCOUNT",
            "SNOWFLAKE_USER",
            "SNOWFLAKE_WAREHOUSE",
            "SNOWFLAKE_DATABASE",
            "SNOWFLAKE_SCHEMA",
        )
        missing = [v for v in required_vars if not os.environ.get(v)]
        if missing:
            raise EnvironmentError(
                f"Missing required Snowflake env vars: {', '.join(missing)}. "
                "Add them to your .env file."
            )

        return {
            "account":       os.environ["SNOWFLAKE_ACCOUNT"],
            "user":          os.environ["SNOWFLAKE_USER"],
            "password":      os.environ.get("SNOWFLAKE_PASSWORD", ""),
            "warehouse":     os.environ["SNOWFLAKE_WAREHOUSE"],
            "database":      os.environ["SNOWFLAKE_DATABASE"],
            "schema":        os.environ["SNOWFLAKE_SCHEMA"],
            "role":          os.environ.get("SNOWFLAKE_ROLE", "PUBLIC"),
            # Default to SSO; no password entry required in the terminal.
            "authenticator": os.environ.get("SNOWFLAKE_AUTHENTICATOR", "externalbrowser"),
        }

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def _open_connection(self):
        """Open a fresh Snowflake connection using the stored config."""
        try:
            import snowflake.connector  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "snowflake-connector-python is not installed. "
                "Run:  pip install snowflake-connector-python"
            ) from exc

        cfg = self._config
        auth = cfg["authenticator"]

        kwargs: dict[str, Any] = {
            "account":                  cfg["account"],
            "user":                     cfg["user"],
            "warehouse":                cfg["warehouse"],
            "database":                 cfg["database"],
            "schema":                   cfg["schema"],
            "role":                     cfg["role"],
            "authenticator":            auth,
            "client_session_keep_alive": True,
            "network_timeout":          60,
            "login_timeout":            120,  # SSO browser flow needs extra time
        }

        # Password is only supplied for non-browser auth modes.
        if auth != "externalbrowser" and cfg.get("password"):
            kwargs["password"] = cfg["password"]
        elif auth != "externalbrowser" and not cfg.get("password"):
            raise EnvironmentError(
                f"SNOWFLAKE_PASSWORD is required when SNOWFLAKE_AUTHENTICATOR={auth}. "
                "Add it to your .env file."
            )

        logger.info(
            "Connecting to Snowflake (account=%s, db=%s, schema=%s, auth=%s)",
            cfg["account"], cfg["database"], cfg["schema"], auth,
        )
        conn = snowflake.connector.connect(**kwargs)
        logger.info("Snowflake connection established")
        return conn

    def _get_connection(self):
        """
        Return a validated live connection, reconnecting automatically if the
        existing one has gone stale (e.g. after a long idle period).
        """
        with self._conn_lock:
            if self._conn is not None:
                # Ping the server to confirm the connection is still alive.
                try:
                    self._conn.cursor().execute("SELECT 1")
                    return self._conn
                except Exception:
                    logger.warning("Stale Snowflake connection detected — reconnecting.")
                    try:
                        self._conn.close()
                    except Exception:
                        pass
                    self._conn = None

            self._conn = self._open_connection()
            return self._conn

    def close(self) -> None:
        """Close the pooled connection and reset the singleton for clean teardown."""
        with self._conn_lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                    logger.info("Snowflake connection closed")
                except Exception as exc:
                    logger.warning("Error closing Snowflake connection: %s", exc)
                finally:
                    self._conn = None

    # ------------------------------------------------------------------
    # QueryExecutor interface
    # ------------------------------------------------------------------

    def execute_query_cached(
        self,
        query: str,
        params: dict[str, object],
        *,
        max_retries: int = 2,
        retry_delay: float = 1.0,
    ) -> pd.DataFrame:
        """
        Execute a parameterised SQL query and return a DataFrame.

        Results are cached in memory for _CACHE_TTL_SECONDS seconds using a
        content-hash key so identical queries skip the round-trip to Snowflake.
        Transient network errors are retried with exponential backoff.

        Parameters use Snowflake's %(name)s bind syntax for SQL-injection safety.
        """
        cache_key = _build_cache_key(query, params)
        now = time.time()

        # Return cached result if it is still fresh.
        if cache_key in _QUERY_CACHE:
            cached_at, cached_df = _QUERY_CACHE[cache_key]
            if now - cached_at < _CACHE_TTL_SECONDS:
                logger.debug("Cache hit (age=%.1fs)", now - cached_at)
                return cached_df.copy()

        last_exc: Exception | None = None
        delay = retry_delay

        for attempt in range(max_retries + 1):
            try:
                conn = self._get_connection()
                cur = conn.cursor()
                t0 = time.time()
                logger.debug("Executing query (attempt %d/%d): %.200s",
                             attempt + 1, max_retries + 1, query)

                cur.execute(query, params)

                # Build DataFrame from cursor results.
                columns = [desc[0] for desc in cur.description] if cur.description else []
                df = pd.DataFrame(cur.fetchall(), columns=columns)
                cur.close()

                logger.info("Query returned %d rows in %.2fs", len(df), time.time() - t0)

                # Evict oldest cache entry before inserting if at capacity.
                if len(_QUERY_CACHE) >= _CACHE_MAX_SIZE:
                    oldest = min(_QUERY_CACHE, key=lambda k: _QUERY_CACHE[k][0])
                    del _QUERY_CACHE[oldest]

                _QUERY_CACHE[cache_key] = (time.time(), df)
                return df

            except Exception as exc:
                last_exc = exc
                is_transient = any(
                    term in str(exc).lower()
                    for term in ("timeout", "connection", "network", "temporarily unavailable")
                )

                if is_transient and attempt < max_retries:
                    logger.warning(
                        "Transient error (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1, max_retries + 1, delay, exc,
                    )
                    time.sleep(delay)
                    delay *= 2  # exponential backoff
                    self.close()  # force fresh connection on next attempt
                else:
                    logger.error("Query failed after %d attempt(s): %s", attempt + 1, exc)
                    raise

        raise RuntimeError(
            f"Query failed after {max_retries + 1} attempts: {last_exc}"
        )


def _load_default_connector() -> QueryExecutor:
    """
    Return the process-wide _SnowflakeConnector singleton.

    Loads the nearest .env file first (if python-dotenv is available) so
    credentials are available without exporting them manually in the shell.
    """
    # Auto-load .env — walk up from this file's directory to the project root.
    try:
        from dotenv import load_dotenv  # type: ignore[import-not-found]
        search = Path(__file__).parent
        while search != search.parent:
            candidate = search / ".env"
            if candidate.exists():
                load_dotenv(candidate, override=False)
                logger.debug("Loaded .env from %s", candidate)
                break
            search = search.parent
    except ImportError:
        pass  # python-dotenv absent; rely on the shell environment

    return _SnowflakeConnector()
