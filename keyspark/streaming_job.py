"""Spark Structured Streaming job (the Lambda speed layer). Two queries share one
parsed Kafka stream: Query A aggregates per-(user, 1-minute) counts to
output/metrics/ in append mode; Query B archives the parsed events to
output/events/ for the batch job and the liveness model. The spatial heatmap is
computed in the batch job so it can be filtered to user-selected ranges.

Schema-change note: Structured Streaming refuses to recover
output/checkpoint/metrics when the metrics column set changes - delete that dir
before restart. The events checkpoint is unaffected.
"""

import time

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    from_json,
    window,
)
from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    LongType,
    StringType,
    StructType,
)

from keyspark.aggregations import event_count_exprs

# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
# Kafka broker is a single-node Apache Kafka 3.8.1 (KRaft, no Zookeeper)
# defined in docker-compose.yml, on localhost:9092. Source topic "events.raw"
# has 1 PARTITION, created in startup-tmux.sh (--partitions 1). One partition:
# one agent, one user, global event ordering for the watermark.
# spark.sql.shuffle.partitions=4 below is Spark shuffle parallelism, not Kafka.
KAFKA_BOOTSTRAP = "localhost:9092"   # tune: Kafka broker address
KAFKA_TOPIC = "events.raw"           # tune: source topic

# Kafka connector artifact. Must match the installed pyspark/Scala build exactly
# (pyspark 4.1.1 is Scala 2.13, hence the _2.13 suffix).
KAFKA_PACKAGE = "org.apache.spark:spark-sql-kafka-0-10_2.13:4.1.1"

METRICS_PATH = "output/metrics"
EVENTS_PATH = "output/events"
METRICS_CHECKPOINT = "output/checkpoint/metrics"
EVENTS_CHECKPOINT = "output/checkpoint/events"

# 5s watermark balances late-event tolerance against time-to-first-window.
# Append + watermark means a window cannot emit until window+watermark of
# wall-clock has elapsed since its first event; larger values (we tried 30s)
# pushed that to ~90s and made the live dashboard dot flicker and recover slowly.
WATERMARK = "5 seconds"        # tune: how long to wait for late/out-of-order events
WINDOW_DURATION = "1 minute"   # tune: metric window size (also sets the min end-to-end latency)


# --------------------------------------------------------------------------
# Spark session + event schema
# --------------------------------------------------------------------------
def build_session():
    return (
        SparkSession.builder
        .appName("keyspark")
        .master("local[*]")                           # tune: Spark master (local[*] = all cores)
        .config("spark.jars.packages", KAFKA_PACKAGE)
        .config("spark.sql.shuffle.partitions", "4")  # tune: shuffle parallelism
        .getOrCreate()
    )


def event_schema(with_seq: bool = False):
    # Explicit schema (no inference): predictable and cheap. with_seq adds the
    # replay sequence number carried by the fault-injection tests
    # (paper/scripts/faults.py); the deployed job never sets it.
    schema = (
        StructType()
        .add("type", StringType())
        .add("key", StringType())
        .add("x", LongType())
        .add("y", LongType())
        .add("button", StringType())
        .add("pressed", BooleanType())
        .add("dx", LongType())
        .add("dy", LongType())
        .add("user", StringType())
        .add("ts", DoubleType())
    )
    return schema.add("seq", LongType()) if with_seq else schema


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------
def main(bootstrap: str = KAFKA_BOOTSTRAP, topic: str = KAFKA_TOPIC,
         metrics_path: str = METRICS_PATH, events_path: str = EVENTS_PATH,
         metrics_checkpoint: str = METRICS_CHECKPOINT,
         events_checkpoint: str = EVENTS_CHECKPOINT, with_seq: bool = False):
    # The defaults are the deployed configuration. The parameters only redirect
    # broker, topic, and output paths, so the fault-injection and latency tests
    # (paper/scripts/faults.py, kafkabench.py) run this exact job elsewhere.
    spark = build_session()
    spark.sparkContext.setLogLevel("WARN")

    # Kafka consumer. Spark Structured Streaming subscribes to "events.raw" and
    # pulls every JSON event the capture agent produced. The matching producer
    # is keyspark/agent.py _kafka_sink.
    # ----- Read from Kafka -----
    raw = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", bootstrap)
        .option("subscribe", topic)
        .option("startingOffsets", "latest")
        # failOnDataLoss=false: if a host sleep/wake left the checkpoint's offset
        # past what the broker still has, reset to the earliest available offset
        # and continue instead of crash-looping forever. Losing input events is
        # cheap; a permanent crash loop is not.
        .option("failOnDataLoss", "false")
        .load()
    )

    # Preprocessing: decode the Kafka value bytes to a string, parse JSON into
    # the explicit event schema, derive event_time from the epoch ts, and attach
    # a 5s watermark so windowed aggregation tolerates late/out-of-order events.
    # ----- Parse JSON value + attach event-time watermark -----
    parsed = (
        raw.selectExpr("CAST(value AS STRING) AS json")
        .select(from_json(col("json"), event_schema(with_seq)).alias("e"))
        .select("e.*")
        .withColumn("event_time", col("ts").cast("timestamp"))
        .withWatermark("event_time", WATERMARK)
    )

    # Spark Streaming runs TWO queries off this one parsed stream (the Lambda
    # speed layer). Query A aggregates live per-(user, 1-min) counts for the
    # dashboard; Query B archives raw events for the batch job and the ML model.
    # ----- Query A: per-(user, 1-minute window) counts -> output/metrics/ -----
    metrics = (
        parsed.groupBy(window(col("event_time"), WINDOW_DURATION), col("user"))
        .agg(*event_count_exprs())
        .select(
            col("window.start").alias("window_start"),
            col("window.end").alias("window_end"),
            col("user"),
            col("keystrokes"),
            col("words"),
            col("corrections"),
            col("clicks"),
        )
    )

    # Triggers bound the file-creation rate (without them every micro-batch writes
    # a part file - that produced 100k+ tiny files). 30s keeps the live dashboard
    # dot fresh; the events archive only feeds the 5-min batch, so 60s is plenty
    # and halves its file count.
    metrics_query = (
        metrics.writeStream
        .format("parquet")
        .option("path", metrics_path)
        .option("checkpointLocation", metrics_checkpoint)
        .outputMode("append")
        .trigger(processingTime="30 seconds")   # tune: metrics write cadence
        .start()
    )

    # ----- Query B: raw parsed events -> output/events/ -----
    events_query = (
        parsed.writeStream
        .format("parquet")
        .option("path", events_path)
        .option("checkpointLocation", events_checkpoint)
        .outputMode("append")
        .trigger(processingTime="60 seconds")   # tune: archive write cadence
        .start()
    )

    # ----- Run loop -----
    queries = [metrics_query, events_query]
    try:
        # Poll instead of awaitAnyTermination(): if a query fails - or the Spark
        # RPC dies on a host sleep, leaving the JVM up but unresponsive - the next
        # poll flips isActive or raises across Py4J, so this process EXITS and the
        # outer restart loop respawns a clean interpreter rather than lingering as
        # a zombie that produces nothing.
        while all(q.isActive for q in queries):
            time.sleep(10)
    finally:
        for q in queries:
            try:
                q.stop()
            except Exception:
                pass
        spark.stop()


if __name__ == "__main__":
    main()
