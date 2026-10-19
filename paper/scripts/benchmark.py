"""KeySpark throughput / latency benchmark - the performance numbers reported in
Table IV of the paper. Two sub-commands:

  batch     - time the Spark batch analytical pipeline (sessionize + per-window
              counts + per-session summaries + per-user baseline) over the full
              archive; reports events/sec, excluding JVM/session startup.
  streaming - replay the archive through Structured Streaming (parquet source,
              availableNow trigger, the same 1-minute windowed aggregation) and
              read Spark's StreamingQueryProgress for rows/sec and per-micro-batch
              latency.

  .venv/bin/python paper/scripts/benchmark.py batch --repeats 3
  .venv/bin/python paper/scripts/benchmark.py streaming --repeats 3

Both read only output/events/ and write nothing back into the live output dirs
(throwaway checkpoint + noop sink), so a benchmark never disturbs the pipeline.

On the streaming throughput definition: an earlier version of this script divided
input rows by the summed triggerExecution time and the paper described the result
as an "input row rate". Those are different quantities. Under an availableNow
trigger the whole archive is already on disk, so Spark's inputRowsPerSecond -
which measures arrival rate between triggers - carries no meaning, while
processedRowsPerSecond = numInputRows / triggerExecution does. This script now
reports processed rows/s and labels it as such, and also records the wall-clock
rate so the two can never be confused again.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time

from pyspark.sql.functions import col, window
from pyspark.sql.streaming.listener import StreamingQueryListener
from pyspark.sql.types import TimestampType

# This script lives in paper/scripts/ but drives the pipeline modules at the
# repository root, and every path below is relative to that root.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

RESULT_PATH = "paper/tables/benchmark.json"

from keyspark.aggregations import event_count_exprs  # noqa: E402
from keyspark.batch_job import (  # noqa: E402
    EVENTS_PATH,
    build_session,
    per_window_metrics,
    session_summary,
    user_baseline,
)
from keyspark.streaming_job import (  # noqa: E402
    WATERMARK,
    WINDOW_DURATION,
    event_schema,
)

log = logging.getLogger("keyspark.benchmark")

# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
# Throwaway checkpoint so the streaming benchmark reprocesses the whole archive
# every run instead of resuming from a previous benchmark.
BENCH_CHECKPOINT = "output/checkpoint/_benchmark"

# All physical event part files, ignoring the streaming sink's _spark_metadata
# (a bare directory read honours that log, which after a reset only references
# recent events; the glob benchmarks the full archive).
EVENTS_GLOB = f"{EVENTS_PATH}/part-*.parquet"

# Throwaway dir the streaming benchmark stages the archive into (a clean dir with
# no _spark_metadata forces ingestion of every recorded event).
STAGE_DIR = "output/_benchmark_events"


# --------------------------------------------------------------------------
# Helper
# --------------------------------------------------------------------------
def _as_dict(progress) -> dict:
    # PySpark returns recentProgress entries as dicts on some versions and as
    # StreamingQueryProgress objects (with .json()) on others. Normalise to dict.
    if isinstance(progress, dict):
        return progress
    return json.loads(progress.json())


class _ProgressCollector(StreamingQueryListener):
    """Capture every micro-batch progress event.

    query.recentProgress only retains the last spark.sql.streaming
    .numRecentProgressUpdates entries (100 by default). Replaying the whole
    archive at maxFilesPerTrigger=4 produces several hundred micro-batches, so
    reading recentProgress silently undercounts both the row total and the
    number of batches. A listener sees all of them.
    """

    def __init__(self) -> None:
        self.progress: list[dict] = []

    def onQueryStarted(self, event) -> None:  # noqa: N802
        pass

    def onQueryProgress(self, event) -> None:  # noqa: N802
        self.progress.append(_as_dict(event.progress))

    def onQueryTerminated(self, event) -> None:  # noqa: N802
        pass

    def onQueryIdle(self, event) -> None:  # noqa: N802
        pass


# --------------------------------------------------------------------------
# Batch benchmark
# --------------------------------------------------------------------------
def benchmark_batch() -> dict:
    """Time the batch analytical core over the full event archive."""
    spark = build_session()
    spark.sparkContext.setLogLevel("WARN")
    try:
        events = spark.read.parquet(EVENTS_GLOB).cache()
        # Force the read+cache first so disk I/O isn't billed to compute.
        t0 = time.perf_counter()
        n_events = events.count()
        read_s = time.perf_counter() - t0

        # Time sessionization + per-window counts + per-session summaries + the
        # per-user baseline. count() forces execution; nothing is written to disk.
        t1 = time.perf_counter()
        per_window = per_window_metrics(events).cache()
        n_windows = per_window.count()
        n_sessions = session_summary(per_window).count()
        user_baseline(per_window).count()
        process_s = time.perf_counter() - t1

        return {
            "events": int(n_events),
            "windows": int(n_windows),
            "sessions": int(n_sessions),
            "read_seconds": round(read_s, 2),
            "process_seconds": round(process_s, 2),
            "throughput_events_per_sec": round(n_events / process_s)
            if process_s
            else None,
        }
    finally:
        spark.stop()


# --------------------------------------------------------------------------
# Streaming benchmark
# --------------------------------------------------------------------------
def benchmark_streaming(max_files_per_trigger: int = 4) -> dict:
    """Replay the archive through Structured Streaming and read Spark's own
    progress metrics for throughput and per-micro-batch latency.

    max_files_per_trigger sets how much work each micro-batch carries. Small
    values expose the fixed per-batch cost (planning, state-store commit, sink
    commit); large values amortise it. Sweeping it maps the throughput/latency
    trade-off that the trigger interval controls in production.
    """
    shutil.rmtree(BENCH_CHECKPOINT, ignore_errors=True)
    shutil.rmtree(STAGE_DIR, ignore_errors=True)
    os.makedirs(STAGE_DIR, exist_ok=True)
    for f in glob.glob(EVENTS_GLOB):
        shutil.copy2(f, STAGE_DIR)

    spark = build_session()
    spark.sparkContext.setLogLevel("WARN")
    collector = _ProgressCollector()
    spark.streams.addListener(collector)
    try:
        # The archive parquet carries the parsed columns plus event_time.
        schema = event_schema().add("event_time", TimestampType())
        stream = (
            spark.readStream.schema(schema)
            # maxFilesPerTrigger chunks the staged archive into several
            # micro-batches so the per-batch latency distribution is meaningful.
            .option("maxFilesPerTrigger", max_files_per_trigger)
            .parquet(STAGE_DIR)
            .withWatermark("event_time", WATERMARK)
        )
        agg = stream.groupBy(
            window(col("event_time"), WINDOW_DURATION), col("user")
        ).agg(*event_count_exprs())

        t0 = time.perf_counter()
        query = (
            agg.writeStream.format("noop")
            .option("checkpointLocation", BENCH_CHECKPOINT)
            .outputMode("append")
            .trigger(availableNow=True)
            .start()
        )
        query.awaitTermination()
        wall_s = time.perf_counter() - t0

        # Give the listener a moment to drain, then prefer its complete record
        # over the truncated recentProgress buffer.
        time.sleep(2)
        progress = collector.progress or [_as_dict(p) for p in query.recentProgress]
        total_rows = sum(p.get("numInputRows", 0) for p in progress)
        batch_ms = [
            p["durationMs"]["triggerExecution"]
            for p in progress
            if p.get("durationMs", {}).get("triggerExecution") is not None
        ]
        total_ms = sum(batch_ms)
        n_batches = len(batch_ms)

        # Two distinct rates, reported separately so they cannot be conflated.
        # "processed" bills only the time Spark spent inside triggerExecution;
        # "wall" bills the whole replay including planning and commit overhead.
        return {
            "max_files_per_trigger": max_files_per_trigger,
            "events": int(total_rows),
            "micro_batches": n_batches,
            "wall_seconds": round(wall_s, 2),
            "processing_seconds": round(total_ms / 1000, 2),
            "processed_rows_per_sec": round(total_rows / (total_ms / 1000))
            if total_ms
            else None,
            "wall_rows_per_sec": round(total_rows / wall_s) if wall_s else None,
            "mean_batch_latency_ms": round(total_ms / n_batches)
            if n_batches
            else None,
            "median_batch_latency_ms": round(statistics.median(batch_ms))
            if batch_ms
            else None,
            "max_batch_latency_ms": max(batch_ms) if batch_ms else None,
        }
    finally:
        query_stop = getattr(locals().get("query", None), "stop", None)
        if callable(query_stop):
            try:
                query_stop()
            except Exception:
                pass
        try:
            spark.streams.removeListener(collector)
        except Exception:
            pass
        spark.stop()
        shutil.rmtree(BENCH_CHECKPOINT, ignore_errors=True)
        shutil.rmtree(STAGE_DIR, ignore_errors=True)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def environment() -> dict:
    """Record the machine the numbers were measured on, so Table I and Table IV
    can never describe different hardware."""

    def sysctl(name: str) -> str:
        try:
            return subprocess.run(
                ["sysctl", "-n", name], capture_output=True, text=True, check=True
            ).stdout.strip()
        except Exception:
            return "unknown"

    import sklearn
    import pyspark

    return {
        "cpu": sysctl("machdep.cpu.brand_string"),
        "cores": sysctl("hw.ncpu"),
        "memory_gb": round(int(sysctl("hw.memsize")) / 1024**3)
        if sysctl("hw.memsize").isdigit()
        else "unknown",
        "os": f"{platform.system()} {platform.mac_ver()[0] or platform.release()}",
        "python": platform.python_version(),
        "pyspark": pyspark.__version__,
        "scikit_learn": sklearn.__version__,
    }


def summarize(runs: list[dict]) -> dict:
    """Mean and standard deviation of every numeric field across repeats."""
    if not runs:
        return {}
    out: dict = {}
    for key in runs[0]:
        values = [r[key] for r in runs if isinstance(r.get(key), (int, float))]
        if len(values) == len(runs) and values:
            out[key] = {
                "mean": round(statistics.mean(values), 2),
                "std": round(statistics.pstdev(values), 2) if len(values) > 1 else 0.0,
                "runs": values,
            }
    return out


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="KeySpark throughput/latency benchmark"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name, help_text in (
        ("batch", "batch analytical pipeline throughput"),
        ("streaming", "Structured Streaming throughput + latency"),
        ("all", "run both, with repeats, and save the result"),
        ("sweep", "streaming throughput/latency vs micro-batch size"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--repeats", type=int, default=1)

    args = parser.parse_args()
    fns = {"batch": benchmark_batch, "streaming": benchmark_streaming}
    result: dict = {"environment": environment(), "repeats": args.repeats}

    if args.cmd == "sweep":
        # One pass per micro-batch size, so the throughput/latency trade-off can
        # be tabulated instead of asserted.
        sweep = []
        for n_files in (1, 2, 4, 8, 16, 32, 64):
            runs = [
                benchmark_streaming(max_files_per_trigger=n_files)
                for _ in range(args.repeats)
            ]
            log.info(
                "maxFilesPerTrigger=%d -> %s rows/s",
                n_files,
                summarize(runs).get("processed_rows_per_sec", {}).get("mean"),
            )
            sweep.append({"summary": summarize(runs), "runs": runs})
        result["sweep"] = sweep
    else:
        todo = list(fns) if args.cmd == "all" else [args.cmd]
        for name in todo:
            runs = []
            for i in range(args.repeats):
                log.info("%s benchmark, run %d/%d", name, i + 1, args.repeats)
                runs.append(fns[name]())
            result[name] = {"summary": summarize(runs), "runs": runs}

    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
    # Merge into any existing result so running one sub-command does not discard
    # the other's numbers.
    if os.path.exists(RESULT_PATH):
        try:
            with open(RESULT_PATH) as fh:
                previous = json.load(fh)
            previous.update(result)
            result = previous
        except Exception:
            pass
    with open(RESULT_PATH, "w") as fh:
        json.dump(result, fh, indent=2)
    log.info("wrote %s", RESULT_PATH)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
