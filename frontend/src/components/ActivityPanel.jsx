import { ACTIVITY_RANGES, bucketizeWindows } from "../utils";
import { useNow } from "../useNow";
import { IdleStrip } from "./IdleStrip";
import { MetricsChart } from "./MetricsChart";

// One card combining the idle/active strip with the per-bucket
// line chart. Both views share the same buckets so the strip's
// cells line up with the chart's x-axis ticks below.
//
// Bucketing happens here (not in the children) so the same array
// drives both views — guaranteeing visual alignment regardless of
// the selected range. The chrome (card border + title) lives here
// rather than in MetricsChart so the strip and the chart visually
// belong together.
// `anchorMs` overrides the bucketizer's end time (default = now) so a
// historical day's panel spans that day instead of the trailing
// window. `label` overrides the card title for the same reason.
export function ActivityPanel({ metrics, range = "1h", anchorMs, label, showMouse = false }) {
  const now = useNow();
  const cfg = ACTIVITY_RANGES[range] ?? ACTIVITY_RANGES["1h"];
  const buckets = bucketizeWindows(
    metrics,
    cfg.bucketCount,
    cfg.bucketSizeMin,
    anchorMs ?? now,
  );
  const totalMinutes = cfg.bucketCount * cfg.bucketSizeMin;

  return (
    <div className="glass-panel">
      <h2 className="text-sm text-zinc-400 mb-3">{label ?? cfg.label}</h2>
      <div className="mb-3">
        <IdleStrip buckets={buckets} totalMinutes={totalMinutes} />
      </div>
      <MetricsChart buckets={buckets} totalMinutes={totalMinutes} showMouse={showMouse} />
    </div>
  );
}
