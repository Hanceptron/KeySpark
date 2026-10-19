"""Shared event-count aggregations: the single source of truth for the four
per-window counts (keystrokes, words, corrections, clicks). Used by the
streaming live metrics and by the batch session + per-day windows, so they can
never silently disagree.
"""

from pyspark.sql import Column
from pyspark.sql import functions as F

# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
# pynput emits " " for space and "Key.<name>" for special keys, so a space can
# arrive in either form. A "correction" is a backspace or delete.
WORD_KEYS = (" ", "Key.space")                     # tune: keys counted as a word boundary
CORRECTION_KEYS = ("Key.backspace", "Key.delete")  # tune: keys counted as a correction


# --------------------------------------------------------------------------
# Aggregation expressions
# --------------------------------------------------------------------------
# Core preprocessing step: turns raw events into the four per-window counts.
# Single source of truth shared by the streaming live metrics and the batch
# job, so they can never disagree.
def event_count_exprs() -> list[Column]:
    """The four per-window count aggregations, in a stable order. Spread into a
    groupBy as ``.agg(*event_count_exprs())``; needs the ``type`` and ``key``
    columns from the parsed event schema.
    """
    is_kd = F.col("type") == "key_down"
    return [
        F.count(F.when(is_kd, 1)).alias("keystrokes"),
        F.count(F.when(is_kd & F.col("key").isin(*WORD_KEYS), 1)).alias("words"),
        F.count(
            F.when(is_kd & F.col("key").isin(*CORRECTION_KEYS), 1)
        ).alias("corrections"),
        F.count(F.when(F.col("type") == "click", 1)).alias("clicks"),
    ]
