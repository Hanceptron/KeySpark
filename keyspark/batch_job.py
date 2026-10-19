"""Spark batch job (the Lambda batch layer). Reads the raw event archive at
output/events/, sessionizes activity, and writes: per-session summaries to
output/sessions/, a per-user baseline to output/baseline/, a flattened
per-window table to output/per_window/, per-day timelines + heatmaps for the
calendar drill-down, and one spatial heatmap per preset range under
output/heatmaps/{1h,6h,1d,3d,1w}/. Overwrites its output dirs each run.
"""

from datetime import timedelta
from pathlib import Path

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

from keyspark.aggregations import event_count_exprs

# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
EVENTS_PATH = "output/events"
SESSIONS_PATH = "output/sessions"
BASELINE_PATH = "output/baseline"
HEATMAPS_PATH = "output/heatmaps"
PER_WINDOW_PATH = "output/per_window"
DAY_MINUTE_METRICS_PATH = "output/day_minute_metrics"
HEATMAP_BY_DAY_PATH = "output/heatmap_by_day"

WINDOW_SIZE = "1 minute"      # tune: per-window aggregation size
SESSION_GAP_SECONDS = 5 * 60  # tune: idle gap (s) that starts a new session

# Heatmap grid resolution (screen px per cell). Smaller = finer/sharper but more
# cells. 16px gives a ~240-wide grid on a 4K display.
CELL_SIZE = 16                # tune: heatmap cell size in pixels
HEATMAP_PRESETS = [           # tune: heatmap preset ranges, (name, hours)
    ("1h", 1),
    ("6h", 6),
    ("1d", 24),
    ("3d", 72),
    ("1w", 168),
]


# --------------------------------------------------------------------------
# Spark session
# --------------------------------------------------------------------------
def build_session():
    return (
        SparkSession.builder
        .appName("keyspark-batch")
        .master("local[*]")                           # tune: Spark master
        .config("spark.sql.shuffle.partitions", "4")  # tune: shuffle parallelism
        .getOrCreate()
    )


# --------------------------------------------------------------------------
# Event archive reader (schema guard)
# --------------------------------------------------------------------------
def _conforming_event_files():
    """Event part files under output/events/ whose x/y/dx/dy are int64 (matching
    the streaming sink). A file with float/double coords (e.g. a pandas frame
    seeded with null coords via a plain to_parquet) makes Spark's vectorized
    reader raise PARQUET_COLUMN_DATA_TYPE_MISMATCH and crash-loop the batch, so
    it is skipped instead. Cheap: reads only parquet footers.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    good = []
    skipped = []
    for path in sorted(Path(EVENTS_PATH).glob("*.parquet")):
        try:
            schema = pq.read_schema(path)
            ok = all(
                name in schema.names
                and pa.types.is_integer(schema.field(name).type)
                for name in ("x", "y", "dx", "dy")
            )
        except Exception:
            ok = False
        if ok:
            good.append(str(path))
        else:
            skipped.append(path.name)
    return good


# --------------------------------------------------------------------------
# Sessionization + per-window metrics
# --------------------------------------------------------------------------
def per_window_metrics(events):
    """One row per (session_id, time_window, user) with the four count metrics."""
    # Preprocessing in the batch layer: sessionize raw events (split into
    # sessions on a >5 min idle gap), then aggregate into 1-minute windows.
    # Sessionize manually: walking each user's events in event-time order, a new
    # session starts when the gap exceeds SESSION_GAP_SECONDS; the cumulative sum
    # of those flags is a stable integer session_id. (session_window cannot share
    # a groupBy with the 1-minute window() used below.)
    user_order = Window.partitionBy("user").orderBy("event_time")
    cumulative = user_order.rowsBetween(Window.unboundedPreceding, Window.currentRow)
    events_s = (
        events
        .withColumn("_prev_ts", F.lag("event_time").over(user_order))
        .withColumn(
            "_new_session",
            F.when(
                F.col("_prev_ts").isNull()
                | (
                    F.col("event_time").cast("double")
                    - F.col("_prev_ts").cast("double")
                    > SESSION_GAP_SECONDS
                ),
                1,
            ).otherwise(0),
        )
        .withColumn("session_id", F.sum("_new_session").over(cumulative))
        .drop("_prev_ts", "_new_session")
    )

    events_w = events_s.withColumn(
        "time_window", F.window(F.col("event_time"), WINDOW_SIZE)
    )

    return (
        events_w.groupBy("session_id", "time_window", "user")
        .agg(*event_count_exprs())
    )


def session_summary(per_window):
    """One row per (session_id, user): session start/end, window count, and the
    four count totals. (The earlier fatigue trend index was removed.)
    """
    return (
        per_window.groupBy("session_id", "user")
        .agg(
            F.min(F.col("time_window.start")).alias("session_start"),
            F.max(F.col("time_window.end")).alias("session_end"),
            F.count("*").alias("window_count"),
            F.sum("keystrokes").alias("keystrokes_total"),
            F.sum("words").alias("words_total"),
            F.sum("corrections").alias("corrections_total"),
            F.sum("clicks").alias("clicks_total"),
        )
    )


def user_baseline(per_window):
    """One row per user with mean and stddev for each of the four metrics."""
    metrics = ["keystrokes", "words", "corrections", "clicks"]
    agg_exprs = []
    for m in metrics:
        agg_exprs.append(F.mean(m).alias(f"{m}_mean"))
        agg_exprs.append(F.stddev(m).alias(f"{m}_std"))
    return (
        per_window.groupBy("user")
        .agg(*agg_exprs)
        .withColumn("computed_at", F.current_timestamp())
    )


# --------------------------------------------------------------------------
# Spatial heatmaps + per-day rollups
# --------------------------------------------------------------------------
def heatmap_for_range(events, hours, max_event_time):
    """Spatial heatmap counts for the last `hours` hours of mouse events. The
    cutoff is measured back from max_event_time (the newest event), not wall-clock
    now, so an archive recorded days ago still produces a populated heatmap.
    """
    cutoff = max_event_time - timedelta(hours=hours)
    return (
        events
        .filter(F.col("event_time") >= F.lit(cutoff))
        .filter(F.col("type").isin("move", "click"))
        .withColumn("cell_x", (F.col("x") / CELL_SIZE).cast("int"))
        .withColumn("cell_y", (F.col("y") / CELL_SIZE).cast("int"))
        .groupBy("cell_x", "cell_y", "type", "user")
        .agg(F.count("*").alias("count"))
    )


def day_minute_metrics(events):
    """One row per (day, 1-minute window, user): the four counts plus a per-minute
    mouse-movement count (move + scroll) for the daily-graph timeline. `day` is
    the local calendar day via date_format, matching the browser's localDayKey on
    a single-user machine. mouse_moves is added here (not in event_count_exprs)
    so the streaming output schema and checkpoint stay untouched.
    """
    ev = (
        events
        .withColumn("time_window", F.window(F.col("event_time"), WINDOW_SIZE))
        .withColumn("day", F.date_format(F.col("event_time"), "yyyy-MM-dd"))
    )
    return (
        ev.groupBy("day", "time_window", "user")
        .agg(
            *event_count_exprs(),
            F.count(F.when(F.col("type").isin("move", "scroll"), 1)).alias("mouse_moves"),
        )
        .select(
            "day",
            F.col("time_window.start").alias("window_start"),
            F.col("time_window.end").alias("window_end"),
            "user",
            "keystrokes",
            "words",
            "corrections",
            "clicks",
            "mouse_moves",
        )
    )


def heatmap_by_day(events):
    """One row per (day, cell_x, cell_y, type, user) over the whole archive (no
    time cutoff), so the frontend can render any single day's move+click heatmap.
    """
    return (
        events
        .filter(F.col("type").isin("move", "click"))
        .withColumn("day", F.date_format(F.col("event_time"), "yyyy-MM-dd"))
        .withColumn("cell_x", (F.col("x") / CELL_SIZE).cast("int"))
        .withColumn("cell_y", (F.col("y") / CELL_SIZE).cast("int"))
        .groupBy("day", "cell_x", "cell_y", "type", "user")
        .agg(F.count("*").alias("count"))
    )


# --------------------------------------------------------------------------
# One batch pass
# --------------------------------------------------------------------------
def compute_all(spark):
    """Run one batch pass over the archive: read events, compute per-session
    summaries, the per-user baseline, the flattened per-window table, per-day
    rollups, and a heatmap per preset range, overwriting the output dirs. Safe to
    call repeatedly on a long-lived session (the unpersist() calls matter for the
    scheduled case). Returns early when output/events/ is missing or has no
    conforming files.
    """
    if not Path(EVENTS_PATH).exists():
        return

    # Read the part files directly (explicit list), not the directory root:
    # reading the root makes Spark honor output/events/_spark_metadata, which only
    # references files written since the last checkpoint reset - so a reset
    # silently orphans every earlier part file and loses history. Listing files
    # always reads the whole archive; _conforming_event_files() also drops any
    # file whose x/y/dx/dy are not int64 (would crash the vectorized reader).
    event_files = _conforming_event_files()
    if not event_files:
        return
    events = spark.read.parquet(*event_files).cache()
    try:
        per_window = per_window_metrics(events).cache()
        try:
            session_summary(per_window).write.mode("overwrite").parquet(SESSIONS_PATH)
            user_baseline(per_window).write.mode("overwrite").parquet(BASELINE_PATH)

            # Flatten the time_window struct so the ML pipeline can read this with
            # plain pandas. Rewritten every batch so training reflects the latest
            # sessionization.
            (
                per_window
                .select(
                    "session_id",
                    F.col("time_window.start").alias("window_start"),
                    F.col("time_window.end").alias("window_end"),
                    "user",
                    "keystrokes",
                    "words",
                    "corrections",
                    "clicks",
                )
                .write.mode("overwrite")
                .parquet(PER_WINDOW_PATH)
            )

            # Per-day keystroke timeline + per-day spatial heatmap (calendar drill-down).
            (
                day_minute_metrics(events)
                .write.mode("overwrite")
                .parquet(DAY_MINUTE_METRICS_PATH)
            )
            (
                heatmap_by_day(events)
                .write.mode("overwrite")
                .parquet(HEATMAP_BY_DAY_PATH)
            )

            # One spatial heatmap per preset time range.
            max_event_time = events.agg(F.max("event_time").alias("m")).first()["m"]
            if max_event_time is not None:
                for name, hours in HEATMAP_PRESETS:
                    (
                        heatmap_for_range(events, hours, max_event_time)
                        .write.mode("overwrite")
                        .parquet(f"{HEATMAPS_PATH}/{name}")
                    )
        finally:
            per_window.unpersist()
    finally:
        events.unpersist()


def main():
    spark = build_session()
    spark.sparkContext.setLogLevel("WARN")
    try:
        compute_all(spark)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
