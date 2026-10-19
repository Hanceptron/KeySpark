import { formatBucketTime, isActiveBucket, parseUtc } from "../utils";

// Horizontal strip of cells, one per bucket over the selected
// range. Green when any activity happened in the bucket, dark
// gray when the user was idle. Hovering a cell reveals the time
// + counts via the native title attribute.
//
// flex-1 on each cell makes the strip exactly fill its container,
// which is sized to match the chart below it in ActivityPanel —
// so cell N visually corresponds to the same bucket as the chart's
// Nth x-axis tick.
export function IdleStrip({ buckets, totalMinutes }) {
  return (
    <div
      className="flex gap-px"
      role="img"
      aria-label="Per-bucket activity timeline for the selected range"
    >
      {buckets.map((m, i) => {
        const active = isActiveBucket(m);
        const time = parseUtc(m.window_start);
        const timeStr = time ? formatBucketTime(time, totalMinutes) : "";
        const counts = active
          ? `${m.keystrokes ?? 0} keys, ${m.clicks ?? 0} clicks`
          : "idle";

        return (
          <div
            key={i}
            className={`flex-1 h-3 rounded-sm ${
              active ? "bg-green-500" : "glass-track"
            }`}
            title={`${timeStr} — ${counts}`}
          />
        );
      })}
    </div>
  );
}
