"""KeySpark backend API. A thin FastAPI service that reads the parquet outputs
from the streaming and batch jobs and serves them as JSON for the React
dashboard (pandas reads parquet per request; no cache, no SQL layer).

This process also OWNS the batch job: on startup it builds the Spark session in a
background thread (so uvicorn accepts connections immediately - the data
endpoints only read parquet from disk), then an APScheduler fires compute_all
every REFRESH_INTERVAL_SEC. A batch failure does not take the API down: the
session is probed, and the process only hard-exits (for the bash restart loop to
respawn a fresh JVM) when the Spark RPC is actually dead.
"""

import json
import os
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from keyspark.batch_job import CELL_SIZE, build_session, compute_all

# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
METRICS_PATH = "output/metrics"
SESSIONS_PATH = "output/sessions"
BASELINE_PATH = "output/baseline"
HEATMAPS_PATH = "output/heatmaps"
DAY_MINUTE_METRICS_PATH = "output/day_minute_metrics"
HEATMAP_BY_DAY_PATH = "output/heatmap_by_day"
ML_METRICS_PATH = "output/models/metrics.json"
LIVENESS_PATH = "output/liveness.parquet"
DISPLAY_PATH = "output/display.json"

# Fallback when the agent has not written output/display.json yet: MacBook 16"
# built-in at default scaling. Keeps the heatmap frame sane on a fresh checkout.
DEFAULT_DISPLAY = {"width": 1728, "height": 1117}  # tune: fallback screen size

SESSIONS_LIMIT = 2000               # tune: max sessions returned (spans the calendar window)
HEATMAP_RANGES = {"1h", "6h", "1d", "3d", "1w"}
REFRESH_INTERVAL_SEC = 300          # tune: batch + liveness refresh cadence (5 min)


# --------------------------------------------------------------------------
# Batch run state (shared between the scheduler thread and request handlers)
# --------------------------------------------------------------------------
@dataclass
class BatchState:
    """In-memory snapshot of the most recent batch run, read by /api/batch_status."""

    last_run: Optional[datetime] = None
    last_status: str = "idle"  # "idle" | "running" | "ok" | "failed"
    last_error: Optional[str] = None


batch_state = BatchState()
# Guards every read/write of batch_state: the scheduler thread mutates it, request
# handlers read it from worker threads. Without the lock a request could observe
# torn state (e.g. status="running" with a stale last_run).
batch_state_lock = threading.Lock()


# --------------------------------------------------------------------------
# Spark liveness probe + batch runner
# --------------------------------------------------------------------------
def _spark_alive(spark, timeout_sec: float = 20.0) -> bool:
    """Probe whether the Spark RPC is still usable by running a trivial action in
    a worker thread (so a hung RPC - the macOS sleep/wake failure mode - is
    treated as dead rather than blocking forever). False on any error/timeout.
    Lets _run_batch tell a dead session (respawn via os._exit) from a transient
    batch error (keep serving the last-good parquet).
    """
    import concurrent.futures

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        executor.submit(lambda: spark.range(1).count()).result(timeout=timeout_sec)
        return True
    except Exception:
        return False
    finally:
        # Never block on a hung probe: if the session is dead the caller is about
        # to os._exit, which tears the thread down with the process.
        executor.shutdown(wait=False)


def _run_batch(spark) -> None:
    """Scheduler entry point: run compute_all then score liveness, recording
    success/failure in batch_state. On failure, probe the session:
      - dead (macOS sleep/wake permanently breaks the local-mode RPC): os._exit
        so the outer bash loop respawns the API with a fresh JVM.
      - alive (bad input file / transient error): log, mark failed, keep serving
        the last-good parquet, retry next tick.
    """
    with batch_state_lock:
        batch_state.last_status = "running"
    try:
        compute_all(spark)
        # Score liveness right after the batch refresh. Pure pandas/sklearn,
        # unrelated to the Spark RPC, so a failure here must not be treated as a
        # dead session - catch it and keep serving the last good flags. Imported
        # locally so a missing scikit-learn never breaks the batch.
        try:
            from keyspark.ml import write_liveness

            write_liveness()
        except Exception:
            pass
        with batch_state_lock:
            batch_state.last_status = "ok"
            batch_state.last_error = None
            batch_state.last_run = datetime.now(timezone.utc)
    except Exception as exc:  # noqa: BLE001 - see docstring
        # Record the failure first so a request landing now sees real state.
        with batch_state_lock:
            batch_state.last_status = "failed"
            batch_state.last_error = str(exc)
            batch_state.last_run = datetime.now(timezone.utc)
        if _spark_alive(spark):
            # Data/transient error: keep the session + last-good parquet, retry
            # next tick. Do NOT exit - the API stays up.
            return
        # Spark session is dead: hard-exit so the bash loop respawns a fresh JVM.
        # os._exit bypasses uvicorn/APScheduler/asyncio teardown intentionally -
        # anything touching the dead RPC would hang.
        os._exit(1)


# --------------------------------------------------------------------------
# App lifespan: bring Spark up off the startup path, then schedule the batch
# --------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Accept HTTP connections immediately and build Spark (JVM cold-start, tens
    of seconds) in a background thread, then schedule the batch once ready.
    Building it before yield kept uvicorn from accepting connections for that
    whole window, so after every respawn the dashboard was unreachable. The data
    endpoints only read parquet, so the API serves existing outputs instantly;
    fresh data lands a few seconds after Spark finishes building.
    """
    runtime = {"spark": None, "scheduler": None, "stopping": False}

    def _bring_up_spark() -> None:
        try:
            spark = build_session()
            spark.sparkContext.setLogLevel("WARN")
        except Exception:
            return
        if runtime["stopping"]:
            spark.stop()
            return
        runtime["spark"] = spark
        scheduler = BackgroundScheduler()
        scheduler.add_job(
            _run_batch,
            "interval",
            seconds=REFRESH_INTERVAL_SEC,
            args=[spark],
            max_instances=1,  # skip tick if previous run is still in progress
            coalesce=True,    # collapse multiple missed ticks into one run
            # First run ~2 s after Spark is ready, not one full interval later.
            next_run_time=datetime.now() + timedelta(seconds=2),
        )
        scheduler.start()
        runtime["scheduler"] = scheduler

    threading.Thread(target=_bring_up_spark, name="spark-init", daemon=True).start()
    try:
        yield
    finally:
        # May run before the background thread finished building Spark; the
        # stopping flag makes that thread skip starting the scheduler, and we tear
        # down whatever already exists.
        runtime["stopping"] = True
        if runtime["scheduler"] is not None:
            runtime["scheduler"].shutdown(wait=False)
        if runtime["spark"] is not None:
            runtime["spark"].stop()


# Data visualization backend. This FastAPI service reads the parquet outputs
# from the streaming and batch jobs and serves them as JSON to the React
# dashboard (frontend/src): live metrics chart, mouse heatmap, session table,
# month calendar with liveness flags, and the ML metrics panel.
app = FastAPI(title="KeySpark API", lifespan=lifespan)
# allow_origins=["*"] is intentionally permissive for the single-user dev setup
# (Vite on 5173 + this API on 8000). Tighten before exposing on a network.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# Parquet -> JSON helpers
# --------------------------------------------------------------------------
def _read_parquet(path: str) -> pd.DataFrame:
    # Empty frame when the dir does not exist yet (batch outputs are missing until
    # the job runs once). Spark uses atomic file-level commits, so a request lands
    # on a self-consistent snapshot; the rare race between _temporary rename and
    # _spark_metadata commit can yield an empty frame. Acceptable for one user.
    if not Path(path).exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _to_records(df: pd.DataFrame) -> list:
    # Round-trip through pandas's JSON serializer so timestamps become ISO strings
    # and NaN becomes JSON null in one step, then parse back to Python for FastAPI.
    if df.empty:
        return []
    return json.loads(df.to_json(orient="records", date_format="iso"))


# --------------------------------------------------------------------------
# Endpoints: live metrics (streaming)
# --------------------------------------------------------------------------
@app.get("/api/metrics")
def get_metrics(minutes: int = 60) -> list:
    df = _read_parquet(METRICS_PATH)
    if df.empty:
        return []
    # Spark writes UTC; pandas reads tz-naive datetime64 holding the UTC value.
    # Compute the threshold in UTC and strip tz so the comparison is naive-vs-naive.
    threshold = pd.Timestamp.now("UTC").tz_localize(None) - pd.Timedelta(minutes=minutes)
    df = df[df["window_start"] >= threshold].sort_values("window_start")
    return _to_records(df)


# --------------------------------------------------------------------------
# Endpoints: batch analytics (sessions, baseline, heatmaps, per-day)
# --------------------------------------------------------------------------
@app.get("/api/sessions")
def get_sessions() -> list:
    df = _read_parquet(SESSIONS_PATH)
    if df.empty:
        return []
    df = df.sort_values("session_start", ascending=False).head(SESSIONS_LIMIT)
    return _to_records(df)


@app.get("/api/baseline")
def get_baseline() -> list:
    return _to_records(_read_parquet(BASELINE_PATH))


@app.get("/api/heatmap")
def get_heatmap(range: str = "1h") -> list:
    # One parquet dir per preset range. An unknown range yields [] rather than an
    # error so the dashboard never breaks on a URL typo.
    if range not in HEATMAP_RANGES:
        return []
    return _to_records(_read_parquet(f"{HEATMAPS_PATH}/{range}"))


@app.get("/api/day_metrics")
def get_day_metrics(day: str, user: str) -> list:
    """Per-minute counts for one calendar day (History drill-down timeline). `day`
    is a local-day key "YYYY-MM-DD" matching the batch job's `day` column.
    """
    df = _read_parquet(DAY_MINUTE_METRICS_PATH)
    if df.empty:
        return []
    df = df[(df["day"] == day) & (df["user"] == user)].sort_values("window_start")
    return _to_records(df)


@app.get("/api/heatmap_day")
def get_heatmap_day(day: str, user: str) -> list:
    """Spatial heatmap cells (move + click) for one calendar day (History
    drill-down). Same day/user matching as /api/day_metrics.
    """
    df = _read_parquet(HEATMAP_BY_DAY_PATH)
    if df.empty:
        return []
    df = df[(df["day"] == day) & (df["user"] == user)]
    return _to_records(df)


@app.get("/api/display")
def get_display() -> dict:
    """Primary-screen grid bounds for framing the mouse heatmap. The agent writes
    output/display.json (screen size in points); we divide by the heatmap
    CELL_SIZE so the frontend gets bounds in the same cell units as the data.
    Falls back to the MacBook 16" default when the file is absent.
    """
    try:
        disp = json.loads(Path(DISPLAY_PATH).read_text())
        width = int(disp["width"])
        height = int(disp["height"])
    except (OSError, ValueError, KeyError, TypeError):
        width = DEFAULT_DISPLAY["width"]
        height = DEFAULT_DISPLAY["height"]
    # ceil so a partial trailing cell is included rather than clipped.
    grid_w = -(-width // CELL_SIZE)
    grid_h = -(-height // CELL_SIZE)
    return {"grid_w": grid_w, "grid_h": grid_h}


# --------------------------------------------------------------------------
# Endpoints: batch health / staleness
# --------------------------------------------------------------------------
@app.get("/api/batch_status")
def get_batch_status() -> dict:
    """Most recent batch-job run state; source of truth for the dashboard
    staleness chips and the watchdog's backend freshness check.
    """
    with batch_state_lock:
        last_run = batch_state.last_run
        status = batch_state.last_status
        error = batch_state.last_error
    return {
        "last_run": last_run.isoformat() if last_run else None,
        "status": status,
        "error": error,
    }


@app.get("/api/health")
def get_health() -> dict:
    """One-glance liveness/freshness snapshot for diagnosing flaky sleep/wake
    episodes. metrics_age_seconds is how long ago the freshest window ended
    (capture + streaming path); streaming_fresh mirrors the dashboard's 2-minute
    live threshold; batch_* reflect the in-process batch scheduler.
    """
    now = pd.Timestamp.now("UTC").tz_localize(None)
    metrics_age = None
    df = _read_parquet(METRICS_PATH)
    if not df.empty and "window_end" in df.columns:
        newest = pd.to_datetime(df["window_end"]).max()
        if pd.notna(newest):
            metrics_age = float((now - newest).total_seconds())

    with batch_state_lock:
        last_run = batch_state.last_run
        status = batch_state.last_status
        error = batch_state.last_error
    batch_age = (
        (datetime.now(timezone.utc) - last_run).total_seconds()
        if last_run
        else None
    )

    return {
        "now": now.isoformat(),
        "metrics_age_seconds": metrics_age,
        "streaming_fresh": metrics_age is not None and metrics_age < 120,
        "batch_last_run": last_run.isoformat() if last_run else None,
        "batch_age_seconds": batch_age,
        "batch_status": status,
        "batch_error": error,
    }


# --------------------------------------------------------------------------
# Endpoints: liveness model (human vs input automation)
# --------------------------------------------------------------------------
@app.get("/api/ml/metrics")
def get_ml_metrics() -> dict:
    """Held-out metrics for the human-vs-non-human liveness classifier (contents
    of output/models/metrics.json written by `keyspark.ml evaluate`). Returns
    {"available": false} when not trained yet so the dashboard renders cleanly.
    """
    p = Path(ML_METRICS_PATH)
    if not p.exists():
        return {"available": False}
    return {"available": True, **json.loads(p.read_text())}


@app.get("/api/liveness")
def get_liveness(user: Optional[str] = None) -> list:
    """Per-(user, day) human-vs-non-human flags from output/liveness.parquet
    (written by the batch scheduler's liveness scoring). The calendar colors a day
    red where `nonhuman` is true. Empty until the model has scored once. Optional
    `user` filter.
    """
    df = _read_parquet(LIVENESS_PATH)
    if df.empty:
        return []
    if user is not None:
        df = df[df["user"] == user]
    return _to_records(df)
