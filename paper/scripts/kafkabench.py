"""Deployed-path throughput, measured latency, and host overhead.

    .venv/bin/python paper/scripts/kafkabench.py sweep   --archive output/events --scratch DIR
    .venv/bin/python paper/scripts/kafkabench.py latency --archive output/events --scratch DIR
    .venv/bin/python paper/scripts/kafkabench.py tables

sweep    (a) Preload a private broker's topic with the archive of Table 4 and sweep
         maxOffsetsPerTrigger over the row counts of 1 to 64 files per batch:
         Kafka source, the deployed parse, watermark, and two queries with real
         Parquet sinks, availableNow trigger, three runs per point.
         (b) Repeat the Parquet-replay sweep of benchmark.py (its own
         benchmark_streaming function, noop sink) three times per point.
latency  The deployed job on a private broker with a real-time replay keeping the
         stream active; probe events (one user id each) spread over the seconds
         of the minute; latency = first appearance of the probe's window in the
         metrics Parquet output minus send time. Then one probe after which
         activity stops. CPU and memory of the Spark JVM and the Kafka container
         are sampled every second throughout (step 11; the capture agent's own
         overhead needs the author and is out of scope).
tables   kafkasweep.tex, figures/kafkasweep_plot.tex, latency.tex.

The archive is read only; every sink and checkpoint is under --scratch.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kafkalab as lab  # noqa: E402

REPO_ROOT = lab.REPO_ROOT
TABLES = os.path.join(REPO_ROOT, "paper", "tables")
FIGURES = os.path.join(REPO_ROOT, "paper", "figures")
FILES_PER_BATCH = [1, 2, 4, 8, 16, 32, 64]
RUNS = 3
BENCH_BROKER = ("keyspark-lab-bench", 9104)
LATENCY_BROKER = ("keyspark-lab-lat", 9102)
TOPIC = "bench.archive"
SPEC_BOUND_S = 65
N_PROBES, PROBES_PER_MINUTE = 120, 5
WARMUP_S, STOP_WATCH_S = 120, 300


def shared_benchmark_json(archive: str) -> str:
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(archive))),
                        "paper", "tables", "benchmark.json")


def table4_files(archive: str) -> list[str]:
    """The part files Table 4 replayed: those older than its benchmark.json.
    Their row count matches the 2,962,308 events stated in the table."""
    cutoff = os.path.getmtime(shared_benchmark_json(archive))
    files = glob.glob(os.path.join(archive, "part-*.parquet"))
    return sorted((f for f in files if os.path.getmtime(f) < cutoff), key=os.path.getmtime)


def snapshot(archive: str, scratch: str) -> tuple[str, list[str]]:
    """Copy Table 4's files (mtimes kept) so the archive is never read twice."""
    snap = os.path.join(scratch, "archive_snapshot")
    files = table4_files(archive)
    if not os.path.isdir(snap) or len(os.listdir(snap)) != len(files):
        shutil.rmtree(snap, ignore_errors=True)
        os.makedirs(snap)
        for f in files:
            shutil.copy2(f, snap)
    return snap, sorted(glob.glob(os.path.join(snap, "part-*.parquet")), key=os.path.getmtime)


# --------------------------------------------------------------------------
# (a) Kafka-source sweep
# --------------------------------------------------------------------------
def preload(broker: lab.Broker, files: list[str]) -> dict:
    """Produce every archive row, in the file order the Parquet replay uses."""
    import pyarrow.parquet as pq
    broker.create_topic(TOPIC)
    producer = lab.make_producer(broker.bootstrap)
    t0, n = time.time(), 0
    for f in files:
        try:
            df = pq.read_table(f).to_pandas()
        except Exception:
            continue
        users = df["user"].tolist()
        tss = df["ts"].tolist()
        for ev, user, ts in zip(lab.event_dicts(df), users, tss):
            ev["user"], ev["ts"] = user, ts
            while True:
                try:
                    producer.produce(TOPIC, key=user.encode(), value=json.dumps(ev).encode())
                    break
                except BufferError:
                    producer.poll(0.1)
            n += 1
        producer.poll(0)
    left = producer.flush(120)
    return {"rows": n, "seconds": time.time() - t0, "unflushed": left}


def kafka_run(bootstrap: str, max_offsets: int, scratch: str) -> dict:
    """The deployed parse and both queries, Kafka source, Parquet sinks, run to
    the end of the preloaded topic with availableNow."""
    from pyspark.sql.functions import col, from_json, window
    from keyspark import streaming_job as sj
    from keyspark.aggregations import event_count_exprs

    out = os.path.join(scratch, "kafka_run")
    shutil.rmtree(out, ignore_errors=True)
    spark = sj.build_session()
    spark.sparkContext.setLogLevel("WARN")
    bench = load_benchmark_module()
    collector = bench._ProgressCollector()
    spark.streams.addListener(collector)
    try:
        raw = (spark.readStream.format("kafka")
               .option("kafka.bootstrap.servers", bootstrap)
               .option("subscribe", TOPIC)
               .option("startingOffsets", "earliest")
               .option("maxOffsetsPerTrigger", max_offsets)
               .option("failOnDataLoss", "false")
               .load())
        parsed = (raw.selectExpr("CAST(value AS STRING) AS json")
                  .select(from_json(col("json"), sj.event_schema()).alias("e"))
                  .select("e.*")
                  .withColumn("event_time", col("ts").cast("timestamp"))
                  .withWatermark("event_time", sj.WATERMARK))
        metrics = (parsed.groupBy(window(col("event_time"), sj.WINDOW_DURATION), col("user"))
                   .agg(*event_count_exprs())
                   .select(col("window.start").alias("window_start"),
                           col("window.end").alias("window_end"), col("user"),
                           col("keystrokes"), col("words"), col("corrections"), col("clicks")))
        t0 = time.perf_counter()
        qm = (metrics.writeStream.format("parquet").option("path", f"{out}/metrics")
              .option("checkpointLocation", f"{out}/ckpt_metrics").outputMode("append")
              .trigger(availableNow=True).start())
        qe = (parsed.writeStream.format("parquet").option("path", f"{out}/events")
              .option("checkpointLocation", f"{out}/ckpt_events").outputMode("append")
              .trigger(availableNow=True).start())
        done = {}
        while len(done) < 2:
            for name, q in (("metrics", qm), ("events", qe)):
                if name not in done and not q.isActive:
                    done[name] = time.perf_counter() - t0
            time.sleep(0.2)
        wall = time.perf_counter() - t0
        time.sleep(2)
        result = {"max_offsets_per_trigger": max_offsets, "wall_seconds": wall}
        for name, q in (("metrics", qm), ("events", qe)):
            prog = [p for p in collector.progress if str(p.get("id")) == str(q.id)]
            rows = sum(p.get("numInputRows", 0) for p in prog)
            ms = [p["durationMs"]["triggerExecution"] for p in prog
                  if p.get("durationMs", {}).get("triggerExecution") is not None]
            result[name] = {
                "events": int(rows), "micro_batches": len(ms),
                "processing_seconds": sum(ms) / 1000,
                "processed_rows_per_sec": rows / (sum(ms) / 1000) if ms else None,
                "wall_seconds": done[name], "wall_rows_per_sec": rows / done[name],
                "mean_batch_latency_ms": statistics.mean(ms) if ms else None,
                "median_batch_latency_ms": statistics.median(ms) if ms else None,
                "max_batch_latency_ms": max(ms) if ms else None,
            }
        return result
    finally:
        try:
            spark.streams.removeListener(collector)
        except Exception:
            pass
        spark.stop()
        shutil.rmtree(out, ignore_errors=True)


_BENCH = None


def load_benchmark_module():
    """paper/scripts/benchmark.py, loaded from the checkout that holds the
    archive, read only. It chdirs to its own root on import, so restore cwd and
    point its staging, checkpoint, and input at scratch."""
    global _BENCH
    if _BENCH is None:
        here = os.getcwd()
        path = os.path.join(ARGS.repo, "paper", "scripts", "benchmark.py")
        spec = importlib.util.spec_from_file_location("benchmark_shared", path)
        _BENCH = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_BENCH)
        os.chdir(here)
        _BENCH.EVENTS_GLOB = os.path.join(ARGS.scratch, "archive_snapshot", "part-*.parquet")
        _BENCH.STAGE_DIR = os.path.join(ARGS.scratch, "bench_stage")
        _BENCH.BENCH_CHECKPOINT = os.path.join(ARGS.scratch, "bench_checkpoint")
    return _BENCH


def summary(runs: list[dict], get) -> dict:
    vals = [get(r) for r in runs]
    return {"mean": statistics.mean(vals), "std": statistics.pstdev(vals),
            "min": min(vals), "max": max(vals), "runs": vals}


def sweep(args) -> None:
    snap, files = snapshot(args.archive, args.scratch)
    rows_per_file = None
    import pyarrow.parquet as pq
    total = 0
    for f in files:
        try:
            total += pq.ParquetFile(f).metadata.num_rows
        except Exception:
            pass
    rows_per_file = total / len(files)
    out_path = os.path.join(TABLES, "kafkasweep.json")
    # Each series runs in its own process (the Parquet replay's session has no Kafka
    # package), so a run updates only its own series in the shared JSON.
    result = {"kafka": [], "parquet": []}
    if os.path.exists(out_path):
        with open(out_path) as fh:
            result.update(json.load(fh))
    result.update({"archive": {"files": len(files), "rows": total, "rows_per_file": rows_per_file,
                               "selection": "part files older than paper/tables/benchmark.json "
                                            "(the input of Table 4)"},
                   "runs_per_point": args.runs})
    for name in args.series:
        result[name] = []

    def save():
        os.makedirs(TABLES, exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump(result, fh, indent=2)

    bench = load_benchmark_module()
    result["environment"] = bench.environment()
    if "parquet" in args.series:
        for n in args.points:
            runs = [bench.benchmark_streaming(max_files_per_trigger=n) for _ in range(args.runs)]
            result["parquet"].append({
                "files_per_batch": n, "runs": runs,
                "processed_rows_per_sec": summary(runs, lambda r: r["processed_rows_per_sec"]),
                "mean_batch_latency_ms": summary(runs, lambda r: r["mean_batch_latency_ms"]),
                "max_batch_latency_ms": summary(runs, lambda r: r["max_batch_latency_ms"]),
            })
            print("parquet", n, result["parquet"][-1]["processed_rows_per_sec"]["mean"], flush=True)
            save()
    if "kafka" in args.series:
        broker = lab.Broker(*BENCH_BROKER)
        broker.start()
        try:
            result["preload"] = preload(broker, files)
            save()
            for n in args.points:
                max_offsets = round(n * rows_per_file)
                runs = [kafka_run(broker.bootstrap, max_offsets, args.scratch) for _ in range(args.runs)]
                result["kafka"].append({
                    "files_per_batch": n, "max_offsets_per_trigger": max_offsets, "runs": runs,
                    "processed_rows_per_sec": summary(runs, lambda r: r["metrics"]["processed_rows_per_sec"]),
                    "mean_batch_latency_ms": summary(runs, lambda r: r["metrics"]["mean_batch_latency_ms"]),
                    "max_batch_latency_ms": summary(runs, lambda r: r["metrics"]["max_batch_latency_ms"]),
                    "events_query_rows_per_sec": summary(runs, lambda r: r["events"]["processed_rows_per_sec"]),
                    "both_queries_wall_rows_per_sec": summary(
                        runs, lambda r: r["metrics"]["events"] / r["wall_seconds"]),
                })
                print("kafka", n, result["kafka"][-1]["processed_rows_per_sec"]["mean"], flush=True)
                save()
        finally:
            broker.remove()
    save()


# --------------------------------------------------------------------------
# Overhead sampler (step 11)
# --------------------------------------------------------------------------
def cpu_seconds(text: str) -> float:
    """ps cputime ([[dd-]hh:]mm:ss.ss) in seconds."""
    days = 0
    if "-" in text:
        d, text = text.split("-", 1)
        days = int(d)
    parts = [float(p) for p in text.split(":")]
    secs = 0.0
    for p in parts:
        secs = secs * 60 + p
    return days * 86400 + secs


class Sampler:
    """Every second: the Spark JVM's CPU (from cputime deltas) and RSS via ps, and
    the Kafka container's CPU and memory via a streaming docker stats."""

    def __init__(self, job: lab.Job, container: str):
        self.job, self.container = job, container
        self.jvm: list[tuple] = []
        self.kafka: list[tuple] = []
        self.stop_flag = threading.Event()

    def _jvm_loop(self) -> None:
        prev = None
        while not self.stop_flag.is_set():
            pid = self.job.jvm_pid()
            if pid:
                out = subprocess.run(["ps", "-o", "cputime=,rss=", "-p", str(pid)],
                                     capture_output=True, text=True).stdout.split()
                if len(out) == 2:
                    t, cpu, rss = time.time(), cpu_seconds(out[0]), int(out[1]) * 1024
                    if prev and prev[0] == pid:
                        self.jvm.append((t, 100 * (cpu - prev[2]) / (t - prev[1]), rss))
                    prev = (pid, t, cpu)
            self.stop_flag.wait(1.0)

    def _kafka_loop(self) -> None:
        p = subprocess.Popen(["docker", "stats", "--format", "{{json .}}", self.container],
                             stdout=subprocess.PIPE, text=True)
        try:
            for line in p.stdout:
                if self.stop_flag.is_set():
                    break
                m = re.search(r"\{.*\}", line)
                if not m:
                    continue
                d = json.loads(m.group(0))
                mem = d.get("MemUsage", "").split("/")[0].strip()
                self.kafka.append((time.time(), float(d.get("CPUPerc", "0%").rstrip("%")),
                                   parse_bytes(mem)))
        finally:
            p.kill()

    def start(self) -> None:
        for fn in (self._jvm_loop, self._kafka_loop):
            threading.Thread(target=fn, daemon=True).start()

    def stop(self) -> None:
        self.stop_flag.set()


def parse_bytes(text: str) -> float:
    m = re.match(r"([\d.]+)\s*([KMGT]?i?B)", text)
    if not m:
        return float("nan")
    scale = {"B": 1, "KiB": 1024, "MiB": 1024**2, "GiB": 1024**3, "TiB": 1024**4,
             "KB": 1e3, "MB": 1e6, "GB": 1e9, "TB": 1e12}[m.group(2)]
    return float(m.group(1)) * scale


def overhead_summary(samples: list[tuple], start: float, end: float) -> dict | None:
    s = [x for x in samples if start <= x[0] < end]
    if not s:
        return None
    cpu = np.array([x[1] for x in s])
    mem = np.array([x[2] for x in s]) / 1024**2
    return {"samples": len(s), "cpu_percent_mean": float(cpu.mean()), "cpu_percent_peak": float(cpu.max()),
            "memory_mib_mean": float(mem.mean()), "memory_mib_peak": float(mem.max())}


# --------------------------------------------------------------------------
# Latency (step 10c)
# --------------------------------------------------------------------------
class Watcher:
    """Polls the metrics sink for new part files and records when each probe
    user's window first appears (the time the dashboard could first read it)."""

    def __init__(self, sink: str):
        self.sink = sink
        self.seen_files: set = set()
        self.first: dict = {}
        self.stop_flag = threading.Event()

    def _loop(self) -> None:
        import pyarrow.parquet as pq
        while not self.stop_flag.is_set():
            for f in glob.glob(os.path.join(self.sink, "part-*.parquet")):
                if f in self.seen_files:
                    continue
                try:
                    users = pq.read_table(f, columns=["user", "window_start"]).to_pandas()
                except Exception:
                    continue   # still being written; retry next poll
                self.seen_files.add(f)
                now = time.time()
                for u in users["user"]:
                    if u.startswith("probe-") and u not in self.first:
                        self.first[u] = {"seen": now, "file_mtime": os.path.getmtime(f)}
            self.stop_flag.wait(0.25)

    def start(self) -> None:
        threading.Thread(target=self._loop, daemon=True).start()


class Background:
    """Real-time replay of the active archive hour, looped, as background activity."""

    def __init__(self, sender: lab.Sender, events: list[dict], offsets: np.ndarray):
        self.sender, self.events, self.offsets = sender, events, offsets
        self.stop_flag = threading.Event()

    def _loop(self) -> None:
        seq = 0
        while not self.stop_flag.is_set():
            t0 = time.time() + 0.5
            for ev, off in zip(self.events, self.offsets):
                if self.stop_flag.is_set():
                    return
                d = t0 + off - time.time()
                if d > 0:
                    time.sleep(d)
                self.sender.send(dict(ev, user=lab.REPLAY_USER, ts=float(t0 + off), seq=seq))
                seq += 1

    def start(self) -> None:
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self) -> float:
        self.stop_flag.set()
        self.thread.join(10)
        return time.time()


def host_context() -> dict:
    """Other load on the host: Spark JVMs and lab brokers running at the time."""
    jvms = subprocess.run(["pgrep", "-f", "SparkSubmit"], capture_output=True, text=True).stdout.split()
    brokers = subprocess.run(["docker", "ps", "--filter", f"label={lab.LABEL}", "--format", "{{.Names}}"],
                             capture_output=True, text=True).stdout.split()
    return {"spark_jvms": len(jvms), "lab_brokers": brokers}


def latency(args) -> None:
    run_dir = os.path.join(args.scratch, "latency")
    shutil.rmtree(run_dir, ignore_errors=True)
    os.makedirs(run_dir)
    sl = lab.load_slice(args.archive, "2026-06-01 20:00", os.path.join(args.scratch, "slice.parquet"), 60)
    events, offsets = lab.event_dicts(sl), (sl["ts"] - sl["ts"].iloc[0]).to_numpy()
    broker = lab.Broker(*LATENCY_BROKER)
    broker.start()
    topic = "latency"
    broker.create_topic(topic)
    job = lab.Job(broker.bootstrap, topic, run_dir, with_seq=False)
    sampler = Sampler(job, broker.name)
    result: dict = {"spec_bound_s": SPEC_BOUND_S}
    try:
        job.start()
        job.wait_ready()
        result["host_context_start"] = host_context()
        sampler.start()
        sender = lab.Sender(broker.bootstrap, topic)
        watcher = Watcher(job.paths["metrics"])
        watcher.start()
        bg = Background(sender, events, offsets)
        t_active = time.time()
        bg.start()
        time.sleep(args.warmup)

        # Probes: PROBES_PER_MINUTE per minute, one in each equal band of the minute,
        # at a random second inside the band (seed 42).
        rng = np.random.default_rng(42)
        band = 60 / PROBES_PER_MINUTE
        base = (time.time() // 60 + 1) * 60
        schedule = []
        for k in range(args.probes):
            minute, slot = divmod(k, PROBES_PER_MINUTE)
            schedule.append(base + minute * 60 + slot * band + rng.uniform(0, band))
        probes = []
        for k, t in enumerate(schedule):
            lab.sleep_until(t)
            ts = time.time()
            sender.send({"type": "key_down", "key": "a", "user": f"probe-{k:03d}", "ts": ts}, kind="probe")
            probes.append({"user": f"probe-{k:03d}", "sent": ts, "second_of_minute": ts % 60})
        # Let the last probes' windows close while activity continues.
        deadline = time.time() + 300
        while time.time() < deadline and not all(p["user"] in watcher.first for p in probes):
            time.sleep(1)
        t_active_end = time.time()

        # Activity stops after one more probe.
        lab.sleep_until((time.time() // 60 + 1) * 60 + 20)
        stop_probe = {"user": "probe-stop", "sent": time.time()}
        sender.send({"type": "key_down", "key": "a", "user": "probe-stop", "ts": stop_probe["sent"]},
                    kind="probe")
        time.sleep(1)
        t_stopped = bg.stop()
        sender.flush(30)
        watcher_stop_key = "probe-stop"
        time.sleep(args.stop_watch)
        stop_probe["seen_during_idle"] = watcher_stop_key in watcher.first
        t_idle_end = time.time()
        # Resume activity: one ordinary event, as the next keystroke would be.
        t_resume = time.time()
        sender.send({"type": "key_down", "key": "a", "user": lab.REPLAY_USER, "ts": t_resume}, kind="resume")
        sender.flush(30)
        deadline = time.time() + 180
        while time.time() < deadline and watcher_stop_key not in watcher.first:
            time.sleep(0.5)
        if watcher_stop_key in watcher.first:
            seen = watcher.first[watcher_stop_key]["seen"]
            stop_probe.update(latency_s=seen - stop_probe["sent"], after_resume_s=seen - t_resume)
        stop_probe.update(idle_s=t_resume - t_stopped)
        # Keep sampling a quiet job (no events at all) for the idle overhead.
        lab.sleep_until(t_resume + 60)
        t_quiet = time.time()
        time.sleep(args.idle_minutes * 60)
        t_quiet_end = time.time()
        watcher.stop_flag.set()
        sampler.stop()
        result["host_context_end"] = host_context()

        lat = []
        for p in probes:
            f = watcher.first.get(p["user"])
            p["latency_s"] = f["seen"] - p["sent"] if f else None
            p["latency_file_mtime_s"] = f["file_mtime"] - p["sent"] if f else None
            lat.append(p["latency_s"])
        vals = np.array([v for v in lat if v is not None])
        result.update({
            "probes": probes,
            "n_probes": len(probes), "n_seen": int(len(vals)),
            "distribution_s": {q: float(np.percentile(vals, p)) for q, p in
                               (("min", 0), ("p05", 5), ("p25", 25), ("median", 50),
                                ("p75", 75), ("p95", 95), ("max", 100))},
            "mean_s": float(vals.mean()),
            "share_within_bound": float((vals <= SPEC_BOUND_S).mean()),
            "stop_probe": stop_probe, "stop_watch_s": args.stop_watch,
            "background": "real-time replay of the 2026-06-01 20:00 archive hour (looped)",
            "detection": "metrics sink polled every 0.25 s for new part files",
        })
        result["overhead"] = {
            "active_replay": {"spark_jvm": overhead_summary(sampler.jvm, t_active, t_active_end),
                              "kafka_container": overhead_summary(sampler.kafka, t_active, t_active_end),
                              "seconds": t_active_end - t_active},
            "idle": {"spark_jvm": overhead_summary(sampler.jvm, t_quiet, t_quiet_end),
                     "kafka_container": overhead_summary(sampler.kafka, t_quiet, t_quiet_end),
                     "seconds": t_quiet_end - t_quiet},
        }
    finally:
        sampler.stop()
        job.stop()
        broker.remove()
    os.makedirs(TABLES, exist_ok=True)
    overhead = result.pop("overhead", None)
    with open(os.path.join(TABLES, "latency.json"), "w") as fh:
        json.dump(result, fh, indent=2)
    if overhead is not None:
        overhead["host_context"] = {k: result[k] for k in ("host_context_start", "host_context_end")
                                    if k in result}
        overhead["notes"] = {
            "method": "ps cputime deltas and RSS of the Spark JVM; docker stats CPUPerc and "
                      "MemUsage of the private Kafka container; one sample per second",
            "out_of_scope": "The capture agent's own CPU and memory need the author at the "
                            "machine (no-author-time variant) and were not measured.",
            "docker_vm": "docker stats reports the container inside Docker Desktop's VM; the "
                         "VM's own overhead is not attributed.",
        }
        with open(os.path.join(TABLES, "overhead.json"), "w") as fh:
            json.dump(overhead, fh, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "probes"}, indent=1))


# --------------------------------------------------------------------------
# Tables and figure
# --------------------------------------------------------------------------
def tables(args) -> None:
    with open(os.path.join(TABLES, "kafkasweep.json")) as fh:
        sw = json.load(fh)
    par = {p["files_per_batch"]: p for p in sw["parquet"]}
    kaf = {p["files_per_batch"]: p for p in sw["kafka"]}
    rows = []
    for n in FILES_PER_BATCH:
        p, k = par.get(n), kaf.get(n)

        def cell(s, places=0):
            return (f"{s['mean']:,.{places}f} ({s['min']:,.{places}f}--{s['max']:,.{places}f})"
                    if s else "--")
        rows.append([str(n), f"{k['max_offsets_per_trigger']:,}" if k else "--",
                     cell(p and p["processed_rows_per_sec"]), cell(k and k["processed_rows_per_sec"]),
                     f"{p['mean_batch_latency_ms']['mean']:,.0f}" if p else "--",
                     f"{k['mean_batch_latency_ms']['mean']:,.0f}" if k else "--"])
    notes = (
        f"Mean (min--max) over {sw['runs_per_point']} runs per point. Both series replay the "
        f"{sw['archive']['rows']:,} events of the complete archive. Parquet: the streaming "
        "replay (files from disk, the windowed aggregation, a \\texttt{noop} sink). Kafka: the "
        "same events preloaded into a Kafka topic and read by the deployed parse and both queries "
        "(windowed counts and archive), each with a Parquet sink and checkpoint, "
        "\\texttt{maxOffsetsPerTrigger} set to the rows of the given number of files; rows/s is the "
        "windowed-count query's processed rate while the archive query runs alongside. Batch "
        "latency is the mean \\texttt{triggerExecution} time of that query."
    )
    lines = [
        "% generated by paper/scripts/kafkabench.py -- do not edit by hand",
        r"\begin{table}[H]",
        r"\caption{Streaming throughput against micro-batch size, from Parquet files and from Kafka.}",
        r"\label{tab:kafkasweep}",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{rrrrrr}",
        r"\toprule",
        r" & & \multicolumn{2}{c}{Processed rows/s} & \multicolumn{2}{c}{Mean batch ms} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
        r"Files & Rows/batch & Parquet & Kafka & Parquet & Kafka \\",
        r"\midrule",
        *[" & ".join(r) + r" \\" for r in rows],
        r"\bottomrule",
        r"\end{tabular}",
        rf"\\[2pt] \footnotesize {notes}",
        r"\end{table}",
    ]
    with open(os.path.join(TABLES, "kafkasweep.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    def series(points, colour, mark, name):
        coords = " ".join(
            f"({p['files_per_batch']},{p['processed_rows_per_sec']['mean']:.0f}) "
            f"+= (0,{p['processed_rows_per_sec']['max'] - p['processed_rows_per_sec']['mean']:.0f}) "
            f"-= (0,{p['processed_rows_per_sec']['mean'] - p['processed_rows_per_sec']['min']:.0f})"
            for p in points)
        return (f"\\addplot[mark={mark}, thick, color={colour}, error bars/.cd, y dir=both, y explicit] "
                f"coordinates {{{coords}}};\n\\addlegendentry{{{name}}}")

    fig = "\n".join([
        "% generated by paper/scripts/kafkabench.py -- do not edit by hand",
        r"\begin{figure}[H]",
        r"\centering",
        r"\begin{tikzpicture}",
        r"\begin{axis}[",
        r"  width=0.70\columnwidth, height=5cm, scale only axis,",
        r"  xmode=log, ymode=log, log basis x=2,",
        r"  xtick={1,2,4,8,16,32,64}, xticklabels={1,2,4,8,16,32,64},",
        r"  xlabel={Files per micro-batch (Kafka: the same number of rows)}, ylabel={Processed rows/s},",
        r"  grid=major, legend pos=north west, legend cell align=left,",
        r"]",
        series(sw["parquet"], "blue", "*", "Parquet replay, \\texttt{noop} sink") if sw["parquet"] else "",
        series(sw["kafka"], "red", "square*", "Kafka source, Parquet sinks") if sw["kafka"] else "",
        r"\end{axis}",
        r"\end{tikzpicture}",
        r"\caption{Streaming throughput against micro-batch size from Parquet files and from Kafka "
        r"(Table~\ref{tab:kafkasweep}); markers give the mean and bars the range of three runs.}",
        r"\label{fig:kafkasweep}",
        r"\end{figure}",
    ])
    os.makedirs(FIGURES, exist_ok=True)
    with open(os.path.join(FIGURES, "kafkasweep_plot.tex"), "w") as fh:
        fh.write(fig + "\n")

    lat_path = os.path.join(TABLES, "latency.json")
    if os.path.exists(lat_path):
        with open(lat_path) as fh:
            lat = json.load(fh)
        d = lat["distribution_s"]
        bands = {}
        for p in lat["probes"]:
            if p["latency_s"] is not None:
                bands.setdefault(int(p["second_of_minute"] // 15), []).append(p["latency_s"])
        rows = [[f"All probes ($n={lat['n_seen']}$)", f"{d['min']:.0f}", f"{d['median']:.0f}",
                 f"{d['p95']:.0f}", f"{d['max']:.0f}"]]
        for b in sorted(bands):
            v = np.array(bands[b])
            rows.append([f"Sent at second {15 * b}--{15 * b + 15} ($n={len(v)}$)", f"{v.min():.0f}",
                         f"{np.median(v):.0f}", f"{np.percentile(v, 95):.0f}", f"{v.max():.0f}"])
        sp = lat["stop_probe"]
        close = np.array([p["latency_s"] + p["second_of_minute"] - 60 for p in lat["probes"]
                          if p["latency_s"] is not None])
        stop_text = (f"it appeared {sp['after_resume_s']:.0f}~s after the next event, "
                     f"{sp['latency_s']:.0f}~s after it was sent" if sp.get("latency_s") is not None
                     else "it did not appear within the observation period")
        notes = (
            f"Seconds from sending a probe event to its window's first appearance in the metrics "
            f"Parquet output, with a real-time archive replay keeping the stream active. Every window "
            f"appeared {close.min():.0f} to {close.max():.0f}~s after it closed (median "
            f"{np.median(close):.0f}~s), so latency falls with the probe's second in the minute; "
            f"{lat['share_within_bound'] * 100:.0f}\\% of probes appeared within "
            f"{lat['spec_bound_s']}~s, the window plus the watermark. When activity paused for {sp['idle_s'] / 60:.0f}~min after a "
            f"probe, its window was {'not ' if not sp['seen_during_idle'] else ''}emitted during the "
            f"pause; {stop_text}."
        )
        lines = [
            "% generated by paper/scripts/kafkabench.py -- do not edit by hand",
            r"\begin{table}[H]",
            r"\caption{Measured latency of the speed layer from the Kafka topic to the per-minute output.}",
            r"\label{tab:latency}",
            r"\centering",
            r"\small",
            r"\begin{tabular}{lrrrr}",
            r"\toprule",
            r"Probes & Min & Median & 95th pct. & Max \\",
            r"\midrule",
            *[" & ".join(r) + r" \\" for r in rows],
            r"\bottomrule",
            r"\end{tabular}",
            rf"\\[2pt] \footnotesize {notes}",
            r"\end{table}",
        ]
        with open(os.path.join(TABLES, "latency.tex"), "w") as fh:
            fh.write("\n".join(lines) + "\n")


ARGS = None


def main() -> None:
    global ARGS
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("cmd", choices=["sweep", "latency", "tables"])
    parser.add_argument("--archive", default=os.path.join(REPO_ROOT, "output", "events"))
    parser.add_argument("--repo", default=None,
                        help="checkout holding paper/scripts/benchmark.py (default: the archive's)")
    parser.add_argument("--scratch", default=os.path.join("/tmp", "keyspark-lab"))
    parser.add_argument("--runs", type=int, default=RUNS)
    parser.add_argument("--series", choices=["parquet", "kafka"], nargs=1, default=["parquet"])
    parser.add_argument("--idle-minutes", type=float, default=10.0)
    parser.add_argument("--points", type=int, nargs="+", default=FILES_PER_BATCH)
    parser.add_argument("--probes", type=int, default=N_PROBES)
    parser.add_argument("--warmup", type=float, default=WARMUP_S)
    parser.add_argument("--stop-watch", type=float, default=STOP_WATCH_S)
    ARGS = parser.parse_args()
    ARGS.repo = ARGS.repo or os.path.dirname(os.path.dirname(os.path.abspath(ARGS.archive)))
    if os.path.abspath(ARGS.scratch).startswith(os.path.dirname(os.path.abspath(ARGS.archive))):
        sys.exit("scratch must not be under the archive's output/ directory")
    os.makedirs(ARGS.scratch, exist_ok=True)
    {"sweep": sweep, "latency": latency, "tables": tables}[ARGS.cmd](ARGS)


if __name__ == "__main__":
    main()
