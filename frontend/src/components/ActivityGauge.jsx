import { ACTIVITY_RANGES, bucketizeWindows, isActiveBucket } from "../utils";
import { useNow } from "../useNow";

// Aggregate of active vs idle time over the selected range. A
// bucket counts as active if any keystrokes, words, corrections,
// or clicks were recorded in it (see isActiveBucket) — same
// definition the IdleStrip uses, so the two panels stay in sync.
// The bucket resolution comes from ACTIVITY_RANGES, so "% active"
// means "share of the range's buckets that saw activity", at the
// same granularity the strip is drawn.
// `anchorMs` overrides the bucketizer end time (default = now) so the
// gauge can summarize a historical day; `label` overrides the "Active
// time · last 60 minutes" suffix for that same day view.
export function ActivityGauge({ metrics, range = "1h", anchorMs, label }) {
  const now = useNow();
  const cfg = ACTIVITY_RANGES[range] ?? ACTIVITY_RANGES["1h"];
  const buckets = bucketizeWindows(
    metrics,
    cfg.bucketCount,
    cfg.bucketSizeMin,
    anchorMs ?? now,
  );
  let activeBuckets = 0;
  for (const b of buckets) {
    if (isActiveBucket(b)) activeBuckets++;
  }
  const total = buckets.length;
  const inactiveBuckets = total - activeBuckets;
  const activePct = total > 0 ? (activeBuckets / total) * 100 : 0;

  const activeMin = activeBuckets * cfg.bucketSizeMin;
  const idleMin = inactiveBuckets * cfg.bucketSizeMin;
  const rangeLabel = (label ?? cfg.label).toLowerCase();

  return (
    <div className="glass-panel">
      <div className="flex items-baseline justify-between gap-3">
        <div className="text-sm text-zinc-400">Active time · {rangeLabel}</div>
        <div className="text-xs text-zinc-500">
          {formatDuration(activeMin)} active · {formatDuration(idleMin)} idle
        </div>
      </div>
      <div className="text-3xl font-semibold text-zinc-100 mt-1">
        {activePct.toFixed(0)}%
      </div>
      <div
        className="mt-3 h-2.5 glass-track rounded-full overflow-hidden"
        role="img"
        aria-label={`${activePct.toFixed(0)}% active over ${rangeLabel}`}
      >
        <div
          className="h-full bg-green-500"
          style={{ width: `${activePct}%` }}
        />
      </div>
    </div>
  );
}

// Compact duration: minutes for short spans, hours (one decimal
// while small, integer once large) for longer ones. Keeps the
// "X active · Y idle" line readable across the full range presets.
function formatDuration(min) {
  if (min < 120) return `${min} min`;
  const hours = min / 60;
  return hours >= 24 ? `${hours.toFixed(0)} h` : `${hours.toFixed(1)} h`;
}
