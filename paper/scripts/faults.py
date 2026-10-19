"""Fault-injection test of the deployed speed layer.

    .venv/bin/python paper/scripts/faults.py all --archive output/events --scratch DIR
    .venv/bin/python paper/scripts/faults.py analyze --scratch DIR
    .venv/bin/python paper/scripts/faults.py cleanup

A replay producer (the producer settings of keyspark/agent.py) replays one active
archive hour into a private Kafka broker, with timestamps moved to the present
(spacing kept) and a sequence number on every event. The deployed streaming job
(keyspark.streaming_job.main: the same two queries, 5 s watermark, 1-minute
windows, append mode, Parquet sinks, checkpoints, 30 s and 60 s triggers,
startingOffsets latest, failOnDataLoss false) consumes it under the restart loop
of run-with-backoff.sh. Ground truth is the per-minute count of the replayed
events, computed in batch with the definitions of keyspark.aggregations.

Scenarios, each run --runs times (all runs in parallel, each on its own topic):
  none           no fault
  kill_jvm       kill -9 of the streaming JVM mid-window; the loop restarts it
  kafka_restart  docker restart of the run's own broker mid-replay
  offsets_lost   topic retention 2 min; consumer stopped 6 min; restart
  duplicates     the 200 most recent events re-sent (same sequence numbers)
  late           200 events with timestamps 10 min old (behind the watermark)

Host sleep is out of scope: it needs the author at the machine.
Writes paper/tables/faults.json and faults.tex.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
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
SLICE_START = "2026-06-01 20:00"   # the only archive hour with events in all 60 minutes
SCENARIOS = ["none", "kill_jvm", "kafka_restart", "offsets_lost", "duplicates", "late"]
LABELS = {
    "none": "No fault",
    "kill_jvm": r"\texttt{kill -9} of the Spark JVM",
    "kafka_restart": "Kafka broker restart",
    "offsets_lost": "Offsets lost (retention)",
    "duplicates": "Duplicate block re-sent",
    "late": "Block behind the watermark",
}
FAULT_AT = 0.4                 # fraction of the replay at which the fault starts
KILL_PHASE = [15.0, 0.2, 30.2]  # wall-clock second of the minute for kill -9, per run
RETENTION_MS, SEGMENT_MS, STOP_S = 120_000, 20_000, 360
BLOCK = 200                     # events re-sent (duplicates) or sent late
LATE_S = 600                    # age of the late block
SENTINEL_AHEAD_S = 66           # closes the last window: 60 s window + 5 s watermark + 1 s
BROKER_A = ("keyspark-lab-a", 9094)
RESTART_BROKERS = [("keyspark-lab-r0", 9096), ("keyspark-lab-r1", 9098), ("keyspark-lab-r2", 9100)]


# --------------------------------------------------------------------------
# One run
# --------------------------------------------------------------------------
def high_watermark(bootstrap: str, topic: str) -> tuple[int, int]:
    from confluent_kafka import Consumer, TopicPartition
    c = Consumer({"bootstrap.servers": bootstrap, "group.id": "keyspark-lab-probe"})
    try:
        return c.get_watermark_offsets(TopicPartition(topic, 0), timeout=10)
    finally:
        c.close()


def wait_drain(job: lab.Job, bootstrap: str, topic: str, limit: float = 300.0) -> dict:
    """Wait until both queries have committed the last offset and the metrics
    query has run one more batch (the one that emits the final window)."""
    high = high_watermark(bootstrap, topic)[1]
    t0 = time.time()
    while time.time() - t0 < limit:
        m = lab.batch_log(job.paths["checkpoint/metrics"])
        e = lab.batch_log(job.paths["checkpoint/events"])
        m_done = m[m["end_offset"] >= high]
        if len(m_done) and (e["end_offset"] >= high).any() and m["batch"].max() > m_done["batch"].min():
            return {"drained": True, "seconds": time.time() - t0, "high": int(high)}
        time.sleep(2)
    return {"drained": False, "seconds": limit, "high": int(high)}


def run_one(scenario: str, run: int, container: str, port: int, scratch: str,
            archive: str, minutes: float) -> None:
    run_dir = os.path.join(scratch, "faults", f"{scenario}-{run}")
    shutil.rmtree(run_dir, ignore_errors=True)
    os.makedirs(run_dir)
    broker = lab.Broker(container, port)
    topic = f"faults.{scenario}.{run}"
    config = ({"retention.ms": str(RETENTION_MS), "segment.ms": str(SEGMENT_MS)}
              if scenario == "offsets_lost" else None)
    broker.create_topic(topic, config)

    sl = lab.load_slice(archive, SLICE_START, os.path.join(scratch, "slice.parquet"), minutes)
    events = lab.event_dicts(sl)
    offsets = (sl["ts"] - sl["ts"].iloc[0]).to_numpy()
    meta = {"scenario": scenario, "run": run, "topic": topic, "broker": container,
            "slice_start": SLICE_START, "minutes": minutes, "events": len(events)}

    job = lab.Job(broker.bootstrap, topic, run_dir)
    job.start()
    meta["job_ready"] = job.wait_ready()
    sender = lab.Sender(broker.bootstrap, topic)
    sent: list[dict] = []
    t0 = time.time() + 2
    meta["t0"] = t0
    fault_at = t0 + FAULT_AT * offsets[-1]
    if scenario == "kill_jvm":
        phase = KILL_PHASE[run % len(KILL_PHASE)]
        aligned = fault_at - fault_at % 60 + phase
        fault_at = aligned if aligned >= fault_at else aligned + 60
    fault: dict = {"planned": fault_at}

    def inject():
        lab.sleep_until(fault_at)
        if scenario == "kill_jvm":
            fault["time"] = job.kill_jvm()
            fault["second_of_minute"] = fault["time"] % 60
        elif scenario == "kafka_restart":
            fault.update(broker.restart())
            fault["time"] = fault["issued"]
        elif scenario == "offsets_lost":
            fault["time"] = time.time()
            job.stop()
            fault["consumer_stopped"] = time.time()
            time.sleep(STOP_S)
            low, high = high_watermark(broker.bootstrap, topic)
            fault["low_offset_at_restart"], fault["high_offset_at_restart"] = int(low), int(high)
            fault["restart"] = time.time()
            job.start()
        elif scenario == "duplicates":
            block = list(sent[-BLOCK:])
            fault["time"] = time.time()
            for ev in block:
                sender.send(ev, kind="duplicate")
            fault["block_span_s"] = block[-1]["ts"] - block[0]["ts"]
            fault["block_age_s"] = [fault["time"] - block[0]["ts"], fault["time"] - block[-1]["ts"]]
        elif scenario == "late":
            block = list(sent[-BLOCK:])
            fault["time"] = time.time()
            for j, ev in enumerate(block):
                sender.send(dict(ev, ts=ev["ts"] - LATE_S, seq=10_000_000 + j), kind="late")
        fault["done"] = time.time()

    injector = None
    if scenario != "none":
        injector = threading.Thread(target=inject, daemon=True)
        injector.start()

    for i, (ev, off) in enumerate(zip(events, offsets)):
        target = t0 + off
        d = target - time.time()
        if d > 0:
            time.sleep(d)
        ev = dict(ev, user=lab.REPLAY_USER, ts=float(target), seq=i)
        sender.send(ev)
        sent.append(ev)
    if injector is not None:
        injector.join()
    last = t0 + offsets[-1]
    sender.send({"type": "move", "x": 0, "y": 0, "user": "_sentinel",
                 "ts": float(last + SENTINEL_AHEAD_S), "seq": -1}, kind="sentinel")
    meta["unflushed"] = sender.flush(120)
    meta["replay_end"] = time.time()
    meta["drain"] = wait_drain(job, broker.bootstrap, topic)
    job.stop()
    meta["fault"] = fault
    meta["job_history"] = job.history
    meta["producer_failed"] = len(sender.failed)
    sender.frame().to_parquet(os.path.join(run_dir, "sent.parquet"))
    with open(os.path.join(run_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2, default=float)


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------
def recovery(meta: dict, checkpoint: str) -> dict | None:
    """Seconds from the fault (for offsets_lost: from the consumer restart) to the
    commit of the first batch planned after it that read new input, and to the
    first such batch of the job process that then stayed up (a restart can itself
    fail and be retried)."""
    f = meta.get("fault", {})
    if meta["scenario"] in ("none", "duplicates", "late") or "time" not in f:
        return None
    start = f["restart"] if meta["scenario"] == "offsets_lost" else f["time"]
    log = lab.batch_log(checkpoint)
    before = log[log["committed"] <= f["time"]]
    end0 = before["end_offset"].max() if len(before) else -1
    new_input = log[log["end_offset"] > end0]
    first = new_input[new_input["planned"] > start]
    starts = [h["start"] for h in meta["job_history"] if h["start"] > f["time"]]
    last_start = max(starts) if starts else start
    stable = new_input[new_input["planned"] > max(start, last_start)]
    gaps = np.diff(log["committed"].to_numpy())
    return {
        "seconds": float(first["committed"].iloc[0] - start) if len(first) else None,
        "stable_seconds": float(stable["committed"].iloc[0] - start) if len(stable) else None,
        "job_starts_after_fault": len(starts),
        "longest_commit_gap_s": float(gaps.max()) if len(gaps) else None,
    }


def analyze_run(run_dir: str) -> dict:
    with open(os.path.join(run_dir, "meta.json")) as fh:
        meta = json.load(fh)
    sent = pd.read_parquet(os.path.join(run_dir, "sent.parquet"))
    intended = sent[sent["kind"].isin(["replay", "late"])].drop_duplicates("seq")
    truth = lab.window_counts(intended)
    out = {"scenario": meta["scenario"], "run": meta["run"], "events_sent": len(intended),
           "resent": int((sent["kind"] == "duplicate").sum()),
           "producer_failed": meta["producer_failed"], "drained": meta["drain"]["drained"],
           "job_restarts": len(meta["job_history"]) - 1}
    for view, committed in (("spark_log", True), ("all_files", False)):
        metrics, minfo = lab.read_sink(os.path.join(run_dir, "metrics"), committed)
        speed = lab.speed_counts(metrics)
        speed = speed[speed["user"] != "_sentinel"]
        archive, einfo = lab.read_sink(os.path.join(run_dir, "events"), committed)
        archive = archive[archive["user"] != "_sentinel"] if len(archive) else archive
        counts = archive["seq"].value_counts() if len(archive) else pd.Series(dtype=int)
        out[view] = {
            "windows": lab.window_agreement(truth, speed),
            "archive_events": int(len(archive)),
            "lost": int(len(set(intended["seq"]) - set(counts.index))),
            "duplicated": int((counts - 1).clip(lower=0).sum()),
            "metrics_files": minfo, "events_files": einfo,
        }
    out["recovery_metrics"] = recovery(meta, os.path.join(run_dir, "checkpoint", "metrics"))
    out["recovery_events"] = recovery(meta, os.path.join(run_dir, "checkpoint", "events"))
    f = meta.get("fault", {})
    if meta["scenario"] == "kill_jvm" and "time" in f:
        out["kill_second_of_minute"] = f["second_of_minute"]
    if meta["scenario"] == "kafka_restart" and "issued" in f:
        out["broker_down_s"] = f["broker_ready"] - f["issued"]
    if meta["scenario"] == "offsets_lost" and "restart" in f:
        out["consumer_down_s"] = f["restart"] - f["time"]
        out["offsets_deleted_by_retention"] = f["low_offset_at_restart"]
    if meta["scenario"] == "duplicates" and "block_age_s" in f:
        out["duplicate_block_age_s"] = f["block_age_s"]
    with open(os.path.join(run_dir, "job.log"), errors="replace") as fh:
        log = fh.read()
    out["job_log_data_loss_warnings"] = log.count("may have been lost")
    lines = log.splitlines()
    out["query_errors"] = [lines[i + 1].split(":")[0].strip() for i, line in enumerate(lines[:-1])
                           if "terminated with error" in line]
    return out


def summarize(runs: list[dict]) -> dict:
    def col(get):
        vals = [get(r) for r in runs]
        vals = [v for v in vals if v is not None]
        if not vals:
            return None
        return {"mean": float(np.mean(vals)), "min": float(np.min(vals)), "max": float(np.max(vals))}
    return {
        "precision": col(lambda r: r["spark_log"]["windows"]["precision"]),
        "recall": col(lambda r: r["spark_log"]["windows"]["recall"]),
        "lost": col(lambda r: r["spark_log"]["lost"]),
        "duplicated_spark_log": col(lambda r: r["spark_log"]["duplicated"]),
        "duplicated_all_files": col(lambda r: r["all_files"]["duplicated"]),
        "lost_all_files": col(lambda r: r["all_files"]["lost"]),
        "precision_all_files": col(lambda r: r["all_files"]["windows"]["precision"]),
        "recovery_s": col(lambda r: (r["recovery_metrics"] or {}).get("seconds")),
        "stable_recovery_s": col(lambda r: (r["recovery_metrics"] or {}).get("stable_seconds")),
        "job_starts_after_fault": col(lambda r: (r["recovery_metrics"] or {}).get("job_starts_after_fault")),
    }


def fmt_range(s: dict | None, places: int = 0) -> str:
    if s is None:
        return "--"
    lo, hi = f"{s['min']:,.{places}f}", f"{s['max']:,.{places}f}"
    return lo if lo == hi else f"{lo}--{hi}"


def write_table(result: dict) -> None:
    rows = []
    for s in SCENARIOS:
        sm = result["scenarios"].get(s, {}).get("summary")
        if not sm:
            continue
        rows.append([LABELS[s], fmt_range(sm["precision"], 3), fmt_range(sm["recall"], 3),
                     fmt_range(sm["lost"]), fmt_range(sm["duplicated_spark_log"]),
                     fmt_range(sm["duplicated_all_files"]), fmt_range(sm["job_starts_after_fault"]),
                     fmt_range(sm["stable_recovery_s"])])
    cfg = result["config"]
    notes = (
        f"Each row gives the range over {cfg['runs']} runs. One active archive hour "
        f"({cfg['events']:,} events) replayed in real time into a private broker; the deployed job "
        "(two queries, Parquet sinks, checkpoints, \\texttt{failOnDataLoss=false}) runs under the "
        "restart loop of the deployment. Precision and recall are over one-minute windows: a "
        "speed-layer window is correct when all four counts equal the batch ground truth. Lost and "
        "duplicated events are counted by sequence number in the archive sink, as Spark's commit log "
        "records it (log) and as the batch layer and API read it, every part file (files). Starts "
        "counts job processes the restart loop launched after the fault. Recovery is the time from "
        "the fault (for retention, from the consumer restart) to the commit of the first metrics "
        "batch planned after it that read new input, in the job process that then stayed up. Host "
        "sleep was not tested."
    )
    lines = [
        "% generated by paper/scripts/faults.py -- do not edit by hand",
        r"\begin{table}[H]",
        r"\caption{Fault injection on the deployed speed layer.}",
        r"\label{tab:faults}",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r" & \multicolumn{2}{c}{Windows} & & \multicolumn{2}{c}{Duplicated} & & \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){5-6}",
        r"Scenario & Prec. & Rec. & Lost & Log & Files & Starts & Recovery (s) \\",
        r"\midrule",
        *[" & ".join(r) + r" \\" for r in rows],
        r"\bottomrule",
        r"\end{tabular}",
        rf"\\[2pt] \footnotesize {notes}",
        r"\end{table}",
    ]
    with open(os.path.join(TABLES, "faults.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


def analyze(scratch: str, minutes: float, runs: int) -> dict:
    result = {
        "config": {"slice_start": SLICE_START, "minutes": minutes, "runs": runs,
                   "fault_at_fraction": FAULT_AT, "kill_phase_s": KILL_PHASE,
                   "retention_ms": RETENTION_MS, "segment_ms": SEGMENT_MS,
                   "consumer_stop_s": STOP_S, "block_events": BLOCK, "late_age_s": LATE_S,
                   "broker_retention_check_interval_ms": 5000,
                   "spark_ui": "disabled for test jobs (port range); processing unchanged"},
        "out_of_scope": {
            "host_sleep": "Not tested: host-sleep scenarios need the author at the machine.",
        },
        "scenarios": {},
    }
    for s in SCENARIOS:
        per_run = []
        for r in range(runs):
            d = os.path.join(scratch, "faults", f"{s}-{r}")
            if not os.path.exists(os.path.join(d, "meta.json")):
                per_run.append({"scenario": s, "run": r, "error": "run did not finish"})
                continue
            per_run.append(analyze_run(d))
        ok = [r for r in per_run if "error" not in r]
        result["scenarios"][s] = {"runs": per_run, "summary": summarize(ok) if ok else None}
        if ok:
            result["config"]["events"] = ok[0]["events_sent"] - (BLOCK if s == "late" else 0)
    os.makedirs(TABLES, exist_ok=True)
    with open(os.path.join(TABLES, "faults.json"), "w") as fh:
        json.dump(result, fh, indent=2)
    write_table(result)
    return result


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def run_all(args) -> None:
    shutil.rmtree(os.path.join(args.scratch, "faults"), ignore_errors=True)
    os.makedirs(os.path.join(args.scratch, "faults"), exist_ok=True)
    lab.load_slice(args.archive, SLICE_START, os.path.join(args.scratch, "slice.parquet"), args.minutes)
    brokers = [lab.Broker(*BROKER_A, {"KAFKA_LOG_RETENTION_CHECK_INTERVAL_MS": "5000"})]
    brokers += [lab.Broker(n, p) for n, p in RESTART_BROKERS[:args.runs]]
    procs = []
    try:
        for b in brokers:
            b.start()
        for r in range(args.runs):
            for s in SCENARIOS:
                name, port = RESTART_BROKERS[r] if s == "kafka_restart" else BROKER_A
                log = open(os.path.join(args.scratch, "faults", f"{s}-{r}.out"), "w")
                cmd = [sys.executable, os.path.abspath(__file__), "run", "--scenario", s,
                       "--run", str(r), "--container", name, "--port", str(port),
                       "--scratch", args.scratch, "--archive", args.archive,
                       "--minutes", str(args.minutes)]
                procs.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT))
                time.sleep(args.stagger)
        for p in procs:
            p.wait()
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
        for b in brokers:
            b.remove()
    print(json.dumps(analyze(args.scratch, args.minutes, args.runs)["scenarios"], indent=1,
                     default=float)[:4000])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("all", "run", "analyze", "cleanup"):
        p = sub.add_parser(name)
        p.add_argument("--scratch", default=os.path.join("/tmp", "keyspark-lab"))
        p.add_argument("--archive", default=os.path.join(REPO_ROOT, "output", "events"))
        p.add_argument("--minutes", type=float, default=60.0)
        p.add_argument("--runs", type=int, default=3)
        p.add_argument("--stagger", type=float, default=20.0)
        p.add_argument("--scenario", choices=SCENARIOS)
        p.add_argument("--run", type=int, default=0)
        p.add_argument("--container")
        p.add_argument("--port", type=int)
    args = parser.parse_args()
    output_dir = os.path.dirname(os.path.abspath(args.archive))
    if os.path.abspath(args.scratch).startswith(output_dir):
        sys.exit("scratch must not be under the archive's output/ directory")
    if args.cmd == "all":
        run_all(args)
    elif args.cmd == "run":
        run_one(args.scenario, args.run, args.container, args.port, args.scratch,
                args.archive, args.minutes)
    elif args.cmd == "analyze":
        analyze(args.scratch, args.minutes, args.runs)
    else:
        print("removed", lab.remove_all_lab_containers())


if __name__ == "__main__":
    main()
