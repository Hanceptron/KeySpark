"""Shared helpers for the deployed-path experiments (faults.py, kafkabench.py).

A private single-node Kafka broker in Docker, a replay producer configured like
keyspark/agent.py, the deployed streaming job (keyspark.streaming_job.main)
under the restart loop of run-with-backoff.sh, and readers for the job's
Parquet sinks and checkpoints.

Nothing here touches the live broker (container keyspark-kafka) or output/: every
container this module starts carries the label keyspark-lab=1 and its own named
volume, and every sink and checkpoint lives under the scratch directory passed in.
"""
from __future__ import annotations

import glob
import json
import os
import signal
import subprocess
import sys
import threading
import time

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from keyspark.aggregations import CORRECTION_KEYS, WORD_KEYS  # noqa: E402

IMAGE = "apache/kafka:3.8.1"
LABEL = "keyspark-lab=1"
REPLAY_USER = "replay"
FIELDS = ["type", "key", "x", "y", "button", "pressed", "dx", "dy"]

# The Spark UI is disabled for test jobs only: many jobs run at once and the UI
# port range (4040 + 16 retries) would run out. It does not change processing.
JOB_ENV = {**os.environ, "PYSPARK_SUBMIT_ARGS": "--conf spark.ui.enabled=false pyspark-shell"}


# --------------------------------------------------------------------------
# Broker
# --------------------------------------------------------------------------
class Broker:
    """A private broker with the settings of docker-compose.yml on another port."""

    def __init__(self, name: str, port: int, extra_env: dict | None = None):
        self.name, self.port = name, port
        self.volume = f"{name}-data"
        self.bootstrap = f"localhost:{port}"
        self.extra_env = extra_env or {}

    def env(self) -> dict:
        controller = self.port + 1
        return {
            "KAFKA_NODE_ID": "1",
            "KAFKA_PROCESS_ROLES": "broker,controller",
            "KAFKA_LISTENERS": f"PLAINTEXT://:{self.port},CONTROLLER://:{controller}",
            "KAFKA_ADVERTISED_LISTENERS": f"PLAINTEXT://localhost:{self.port}",
            "KAFKA_CONTROLLER_LISTENER_NAMES": "CONTROLLER",
            "KAFKA_CONTROLLER_QUORUM_VOTERS": f"1@localhost:{controller}",
            "KAFKA_LISTENER_SECURITY_PROTOCOL_MAP": "CONTROLLER:PLAINTEXT,PLAINTEXT:PLAINTEXT",
            "KAFKA_INTER_BROKER_LISTENER_NAME": "PLAINTEXT",
            "KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR": "1",
            "KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR": "1",
            "KAFKA_TRANSACTION_STATE_LOG_MIN_ISR": "1",
            "KAFKA_AUTO_CREATE_TOPICS_ENABLE": "true",
            "KAFKA_LOG_DIRS": "/var/lib/kafka/data",
            **self.extra_env,
        }

    def start(self) -> None:
        if self.name == "keyspark-kafka":
            raise ValueError("refusing to manage the live broker")
        self.remove()   # a leftover of an interrupted run with this name
        cmd = ["docker", "run", "-d", "--name", self.name, "--label", LABEL,
               "-p", f"{self.port}:{self.port}", "-v", f"{self.volume}:/var/lib/kafka/data"]
        for k, v in self.env().items():
            cmd += ["-e", f"{k}={v}"]
        subprocess.run(cmd + [IMAGE], check=True, capture_output=True)
        subprocess.run(["docker", "volume", "inspect", self.volume], check=True, capture_output=True)
        self.wait_ready()

    def ready(self, timeout: float = 2.0) -> bool:
        from confluent_kafka.admin import AdminClient
        try:
            AdminClient({"bootstrap.servers": self.bootstrap}).list_topics(timeout=timeout)
            return True
        except Exception:
            return False

    def wait_ready(self, limit: float = 120.0) -> float:
        t0 = time.time()
        while time.time() - t0 < limit:
            if self.ready():
                return time.time()
            time.sleep(0.5)
        raise TimeoutError(f"broker {self.name} not ready after {limit}s")

    def restart(self) -> dict:
        """docker restart (SIGTERM, then start); returns the timings."""
        t0 = time.time()
        subprocess.run(["docker", "restart", self.name], check=True, capture_output=True)
        t_started = time.time()
        t_ready = self.wait_ready()
        return {"issued": t0, "container_started": t_started, "broker_ready": t_ready}

    def create_topic(self, topic: str, config: dict | None = None) -> None:
        from confluent_kafka.admin import AdminClient, NewTopic
        admin = AdminClient({"bootstrap.servers": self.bootstrap})
        fut = admin.create_topics([NewTopic(topic, num_partitions=1, replication_factor=1,
                                            config=config or {})])
        fut[topic].result(30)

    def remove(self) -> None:
        if self.name == "keyspark-kafka":
            raise ValueError("refusing to manage the live broker")
        subprocess.run(["docker", "rm", "-f", self.name], capture_output=True)
        subprocess.run(["docker", "volume", "rm", self.volume], capture_output=True)


def remove_all_lab_containers() -> list:
    """Remove every container and volume this module created (label keyspark-lab=1)."""
    out = subprocess.run(["docker", "ps", "-a", "--filter", f"label={LABEL}", "--format", "{{.Names}}"],
                         capture_output=True, text=True).stdout.split()
    for name in out:
        if name == "keyspark-kafka":
            continue
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        subprocess.run(["docker", "volume", "rm", f"{name}-data"], capture_output=True)
    return out


# --------------------------------------------------------------------------
# Archive slice and replay producer
# --------------------------------------------------------------------------
def read_parts(files: list[str]) -> pd.DataFrame:
    """Rows of the given part files, skipping unreadable ones (zero-byte or
    truncated files occur in the archive, as keyspark.events_io notes)."""
    import pyarrow.parquet as pq
    frames = []
    for f in files:
        try:
            frames.append(pq.read_table(f).to_pandas())
        except Exception:
            pass
    return pd.concat(frames, ignore_index=True)


def load_slice(archive: str, start: str, cache: str, minutes: float = 60.0) -> pd.DataFrame:
    """The archive events of user-001 in [start, start + minutes), cached."""
    if os.path.exists(cache):
        df = pd.read_parquet(cache)
    else:
        df = read_parts(sorted(glob.glob(os.path.join(archive, "part-*.parquet"))))
        t0 = pd.Timestamp(start, tz="UTC").timestamp()
        df = df[(df["user"] == "user-001") & (df["ts"] >= t0) & (df["ts"] < t0 + 3600)]
        df = df.sort_values("ts", kind="stable").reset_index(drop=True)[FIELDS + ["ts"]]
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        df.to_parquet(cache)
    return df[df["ts"] < df["ts"].iloc[0] + minutes * 60].reset_index(drop=True)


def event_dicts(df: pd.DataFrame) -> list[dict]:
    """Archive rows as the agent's JSON events: only the fields a type carries."""
    out = []
    for row in df[FIELDS].itertuples(index=False):
        ev = {}
        for name, value in zip(FIELDS, row):
            if value is None or (isinstance(value, float) and np.isnan(value)) or value is pd.NA:
                continue
            ev[name] = value.item() if hasattr(value, "item") else value
        out.append(ev)
    return out


def make_producer(bootstrap: str):
    """The producer configuration of keyspark/agent.py _kafka_sink, on another broker."""
    from confluent_kafka import Producer
    return Producer({
        "bootstrap.servers": bootstrap,
        "client.id": "keyspark-agent",
        "reconnect.backoff.ms": 500,
        "reconnect.backoff.max.ms": 10_000,
        "socket.keepalive.enable": True,
        "message.send.max.retries": 5,
        "retry.backoff.ms": 200,
    })


class Sender:
    """Produce JSON events keyed by user, as the agent does, and keep a send log
    plus delivery reports (the agent ignores them; here they separate producer
    loss from pipeline loss)."""

    def __init__(self, bootstrap: str, topic: str):
        self.producer = make_producer(bootstrap)
        self.topic = topic
        self.log: list[tuple] = []
        self.failed: list[int] = []
        self.lock = threading.Lock()

    def send(self, ev: dict, kind: str = "replay") -> None:
        seq = ev.get("seq", -1)

        def report(err, _msg, seq=seq):
            if err is not None:
                with self.lock:
                    self.failed.append(seq)

        self.producer.poll(0)
        while True:
            try:
                self.producer.produce(self.topic, key=ev["user"].encode(),
                                      value=json.dumps(ev).encode(), on_delivery=report)
                break
            except BufferError:
                self.producer.poll(0.1)
        with self.lock:
            self.log.append((seq, ev["ts"], ev["user"], ev["type"], ev.get("key"), time.time(), kind))

    def flush(self, timeout: float = 60.0) -> int:
        return self.producer.flush(timeout)

    def frame(self) -> pd.DataFrame:
        with self.lock:
            return pd.DataFrame(self.log, columns=["seq", "ts", "user", "type", "key", "sent", "kind"])


def sleep_until(t: float) -> None:
    while True:
        d = t - time.time()
        if d <= 0:
            return
        time.sleep(min(d, 0.5))


# --------------------------------------------------------------------------
# The deployed streaming job under the restart loop
# --------------------------------------------------------------------------
class Job:
    """keyspark.streaming_job.main on a private broker, topic, and scratch paths,
    restarted on exit with the backoff of run-with-backoff.sh (5 s, doubling to
    60 s, reset after a run longer than 60 s)."""

    def __init__(self, bootstrap: str, topic: str, run_dir: str, with_seq: bool = True):
        self.run_dir = run_dir
        self.paths = {k: os.path.join(run_dir, k) for k in
                      ("metrics", "events", "checkpoint/metrics", "checkpoint/events")}
        args = dict(bootstrap=bootstrap, topic=topic, metrics_path=self.paths["metrics"],
                    events_path=self.paths["events"],
                    metrics_checkpoint=self.paths["checkpoint/metrics"],
                    events_checkpoint=self.paths["checkpoint/events"], with_seq=with_seq)
        self.code = (f"import sys; sys.path.insert(0, {REPO_ROOT!r}); "
                     f"from keyspark import streaming_job; streaming_job.main(**{args!r})")
        self.proc = None
        self.thread = None
        self.stopping = threading.Event()
        self.history: list[dict] = []

    def _loop(self) -> None:
        delay = 5
        log = open(os.path.join(self.run_dir, "job.log"), "a")
        while not self.stopping.is_set():
            start = time.time()
            self.proc = subprocess.Popen([sys.executable, "-c", self.code], cwd=REPO_ROOT,
                                         stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True, env=JOB_ENV)
            code = self.proc.wait()
            elapsed = time.time() - start
            self.history.append({"start": start, "exit": time.time(), "code": code})
            if self.stopping.is_set():
                break
            if elapsed > 60:
                delay = 5
            self.stopping.wait(delay)
            if elapsed <= 60:
                delay = min(delay * 2, 60)
        log.close()

    def start(self) -> None:
        self.stopping.clear()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def wait_ready(self, limit: float = 300.0) -> float:
        """Both queries have fixed their starting offsets (startingOffsets=latest)."""
        t0 = time.time()
        files = [os.path.join(self.paths[c], "sources", "0", "0")
                 for c in ("checkpoint/metrics", "checkpoint/events")]
        while time.time() - t0 < limit:
            if all(os.path.exists(f) for f in files):
                return time.time()
            time.sleep(0.5)
        raise TimeoutError("streaming job did not start")

    def jvm_pid(self) -> int | None:
        if self.proc is None or self.proc.poll() is not None:
            return None
        out = subprocess.run(["pgrep", "-P", str(self.proc.pid)], capture_output=True, text=True).stdout
        for pid in out.split():
            cmd = subprocess.run(["ps", "-o", "command=", "-p", pid], capture_output=True, text=True).stdout
            if "java" in cmd:
                return int(pid)
        return None

    def kill_jvm(self) -> float:
        pid = self.jvm_pid()
        if pid is None:
            raise RuntimeError("no JVM to kill")
        t = time.time()
        os.kill(pid, signal.SIGKILL)
        return t

    def stop(self, limit: float = 90.0) -> None:
        """Graceful stop (SIGINT runs the job's finally block), then SIGKILL."""
        self.stopping.set()
        p = self.proc
        if p is not None and p.poll() is None:
            p.send_signal(signal.SIGINT)
            try:
                p.wait(limit)
            except subprocess.TimeoutExpired:
                pass
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
        if self.thread is not None:
            self.thread.join(30)


# --------------------------------------------------------------------------
# Sink and checkpoint readers
# --------------------------------------------------------------------------
def committed_files(sink: str) -> set:
    """Part files recorded in the sink's _spark_metadata log (Spark's committed view)."""
    names = set()
    for f in glob.glob(os.path.join(sink, "_spark_metadata", "*")):
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line.startswith("{"):
                    entry = json.loads(line)
                    if entry.get("action", "add") == "add":
                        names.add(os.path.basename(entry["path"]))
    return names


def read_sink(sink: str, committed_only: bool) -> tuple[pd.DataFrame, dict]:
    """Rows of the sink's part files. committed_only=False reads every part file,
    which is what the batch layer, keyspark.events_io, and the API do."""
    import pyarrow.parquet as pq
    files = sorted(glob.glob(os.path.join(sink, "part-*.parquet")))
    keep = committed_files(sink) if committed_only else None
    frames, unreadable, orphans = [], 0, 0
    for f in files:
        if keep is not None and os.path.basename(f) not in keep:
            orphans += 1
            continue
        try:
            frames.append(pq.read_table(f).to_pandas())
        except Exception:
            unreadable += 1
    info = {"files": len(files), "unreadable": unreadable}
    if keep is not None:
        info["not_committed"] = orphans
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), info


def batch_log(checkpoint: str) -> pd.DataFrame:
    """Per committed batch: planning time (offset log mtime), commit time (commit
    log mtime), and end offset of partition 0."""
    rows = []
    for f in glob.glob(os.path.join(checkpoint, "commits", "[0-9]*")):
        n = int(os.path.basename(f))
        off_file = os.path.join(checkpoint, "offsets", str(n))
        end, planned = None, None
        if os.path.exists(off_file):
            planned = os.path.getmtime(off_file)
            with open(off_file) as fh:
                lines = fh.read().splitlines()
            src = json.loads(lines[-1])
            end = sum(int(v) for part in src.values() for v in part.values())
        rows.append((n, planned, os.path.getmtime(f), end))
    return pd.DataFrame(rows, columns=["batch", "planned", "committed", "end_offset"]).sort_values("batch")


# --------------------------------------------------------------------------
# Ground truth
# --------------------------------------------------------------------------
def window_counts(events: pd.DataFrame) -> pd.DataFrame:
    """Per (user, 1-minute window) counts with the definitions of
    keyspark.aggregations.event_count_exprs, on epoch-microsecond window starts
    (Spark casts the double ts to a timestamp by truncating to microseconds)."""
    micros = (events["ts"].astype("float64") * 1e6).astype("int64")
    kd = events["type"] == "key_down"
    df = pd.DataFrame({
        "user": events["user"].values,
        "window": (micros - micros % 60_000_000).values,
        "keystrokes": kd.values,
        "words": (kd & events["key"].isin(WORD_KEYS)).values,
        "corrections": (kd & events["key"].isin(CORRECTION_KEYS)).values,
        "clicks": (events["type"] == "click").values,
    })
    return df.groupby(["user", "window"]).sum().astype("int64").reset_index()


def speed_counts(metrics: pd.DataFrame) -> pd.DataFrame:
    """The metrics sink in the same shape as window_counts."""
    if metrics.empty:
        return pd.DataFrame(columns=["user", "window", "keystrokes", "words", "corrections", "clicks"])
    start = pd.to_datetime(metrics["window_start"])
    if start.dt.tz is None:
        start = start.dt.tz_localize("UTC")
    micros = (start - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(microseconds=1)
    return pd.DataFrame({
        "user": metrics["user"].values, "window": micros.values.astype("int64"),
        **{c: metrics[c].values.astype("int64")
           for c in ("keystrokes", "words", "corrections", "clicks")},
    })


def window_agreement(truth: pd.DataFrame, speed: pd.DataFrame) -> dict:
    """Precision and recall over windows: a speed-layer row is correct when its
    (user, window) is in the ground truth with all four counts equal."""
    keys = ["user", "window"]
    counts = ["keystrokes", "words", "corrections", "clicks"]
    merged = speed.merge(truth, on=keys, how="left", suffixes=("", "_true"), indicator=True)
    correct = (merged["_merge"] == "both")
    for c in counts:
        correct &= merged[c] == merged[f"{c}_true"]
    n_correct = int(correct.sum())
    emitted = speed.drop_duplicates(keys)
    missing = truth.merge(emitted[keys], on=keys, how="left", indicator=True)
    return {
        "truth_windows": len(truth),
        "speed_rows": len(speed),
        "correct": n_correct,
        "precision": n_correct / len(speed) if len(speed) else None,
        "recall": n_correct / len(truth) if len(truth) else None,
        "missing_windows": int((missing["_merge"] == "left_only").sum()),
        "duplicate_rows": int(len(speed) - len(emitted)),
        "wrong_count_rows": int(((merged["_merge"] == "both") & ~correct).sum()),
        "spurious_rows": int((merged["_merge"] == "left_only").sum()),
    }
