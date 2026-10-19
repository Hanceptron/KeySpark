import { useMemo } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { formatBucketTime, parseUtc } from "../utils";

// Two stacked line charts that share one x-axis, split by magnitude so
// the series stop overlapping into an unreadable tangle:
//   - "Volume": keystrokes (left axis) and, on the daily graph, mouse
//     movement (right axis). These are the high-count signals (hundreds
//     to tens of thousands).
//   - "Detail": words, corrections and clicks on their own auto-scaled
//     axis, where small per-minute counts (tens, not thousands) finally
//     have room to separate instead of being crushed flat beneath the
//     keystroke line.
//
// Both charts use identical axis gutters and margins so their plot areas
// line up vertically; only the lower (Detail) chart draws the x-axis
// labels. The Volume chart reserves a right-axis gutter (for mouse) and
// the Detail chart matches it with margin.right so the two stay aligned.
//
// One shared legend sits ABOVE both charts (not inside them) so the
// hover tooltip can never collide with it - each chart is short, and an
// in-chart legend at the bottom got overlapped by the tooltip.
//
// Buckets come from the parent ActivityPanel and are shared with the
// IdleStrip above. connectNulls={false} keeps synthetic (idle) rows from
// being smoothed across. The card chrome (border, title, padding) is
// provided by ActivityPanel.

const TOOLTIP_STYLE = {
  background: "rgba(24,24,27,0.92)",
  border: "1px solid rgba(255,255,255,0.12)",
  borderRadius: 8,
  color: "#e4e4e7",
  backdropFilter: "blur(8px)",
};
const AXIS_STROKE = "#a1a1aa";
const GRID_STROKE = "rgba(255,255,255,0.10)";
const LEFT_AXIS_WIDTH = 44;
const RIGHT_AXIS_WIDTH = 52;

export function MetricsChart({ buckets, totalMinutes, showMouse = false }) {
  // useMemo so the array isn't rebuilt on unrelated parent renders.
  // At 1w there are up to 168 buckets, so the savings are small but
  // real and the dependency list is honest.
  const data = useMemo(
    () =>
      buckets.map((m) => ({
        time: formatBucketTime(parseUtc(m.window_start), totalMinutes),
        keystrokes: m.keystrokes,
        words: m.words,
        corrections: m.corrections,
        clicks: m.clicks ?? 0,
        mouse: m.mouse_moves ?? 0,
      })),
    [buckets, totalMinutes],
  );

  // Empty state: when no bucket in the window carries any plotted
  // activity, show a calm placeholder instead of bare axes so the panel
  // reads as intentional rather than broken.
  const hasData = data.some(
    (d) => d.keystrokes || d.words || d.corrections || d.clicks || d.mouse,
  );
  if (!hasData) {
    return (
      <div className="h-[420px] flex items-center justify-center text-sm text-zinc-500">
        No activity in this window
      </div>
    );
  }

  // Thin the x-axis to ~8 labels regardless of bucket count (60 at 1h,
  // 96 at 1d) so the axis never crowds. interval is the number of ticks
  // skipped between rendered labels.
  const tickInterval = Math.max(0, Math.ceil(data.length / 8) - 1);

  // The Detail chart has no right axis, so it matches the Volume chart's
  // right gutter (16 margin + mouse axis) via margin.right instead. That
  // keeps both plot areas the same width so the shared x-axis lines up.
  const volumeMargin = { top: 8, right: 16, bottom: 0, left: 0 };
  const detailMargin = {
    top: 8,
    right: showMouse ? 16 + RIGHT_AXIS_WIDTH : 16,
    bottom: 0,
    left: 0,
  };

  return (
    <div className="space-y-1">
      {/* Single shared legend, above the plots. Order mirrors the two
          charts: Volume series first, then Detail series. */}
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-zinc-300 pl-11">
        <LegendKey color="var(--color-brand-cyan)" label="keystrokes" />
        {showMouse && <LegendKey color="var(--color-brand-magenta)" label="mouse" />}
        <LegendKey color="var(--color-brand-violet)" label="words" />
        <LegendKey color="var(--color-brand-pink)" label="corrections" />
        <LegendKey color="var(--color-brand-orange)" label="clicks" />
      </div>

      {/* Volume: keystrokes (+ mouse on the daily graph). x-axis hidden;
          the shared time labels are drawn under the Detail chart. */}
      <ResponsiveContainer width="100%" height={showMouse ? 230 : 210}>
        <LineChart data={data} margin={volumeMargin}>
          <CartesianGrid strokeDasharray="3 3" stroke={GRID_STROKE} vertical={false} />
          <XAxis dataKey="time" hide />
          <YAxis yAxisId="left" stroke={AXIS_STROKE} fontSize={12} width={LEFT_AXIS_WIDTH} />
          {showMouse && (
            <YAxis yAxisId="right" orientation="right" stroke={AXIS_STROKE} fontSize={12} width={RIGHT_AXIS_WIDTH} />
          )}
          <Tooltip contentStyle={TOOLTIP_STYLE} />
          <Line yAxisId="left" type="monotone" dataKey="keystrokes" stroke="var(--color-brand-cyan)" dot={false} strokeWidth={2} connectNulls={false} />
          {showMouse && (
            <Line yAxisId="right" type="monotone" dataKey="mouse" stroke="var(--color-brand-magenta)" dot={false} strokeWidth={2} connectNulls={false} />
          )}
        </LineChart>
      </ResponsiveContainer>

      {/* Detail: words / corrections / clicks on their own small scale,
          carrying the shared x-axis labels at the bottom. */}
      <ResponsiveContainer width="100%" height={190}>
        <LineChart data={data} margin={detailMargin}>
          <CartesianGrid strokeDasharray="3 3" stroke={GRID_STROKE} vertical={false} />
          <XAxis dataKey="time" stroke={AXIS_STROKE} fontSize={12} interval={tickInterval} />
          <YAxis yAxisId="left" stroke={AXIS_STROKE} fontSize={12} width={LEFT_AXIS_WIDTH} />
          <Tooltip contentStyle={TOOLTIP_STYLE} />
          <Line yAxisId="left" type="monotone" dataKey="words" stroke="var(--color-brand-violet)" dot={false} strokeWidth={2} connectNulls={false} />
          <Line yAxisId="left" type="monotone" dataKey="corrections" stroke="var(--color-brand-pink)" dot={false} strokeWidth={2} connectNulls={false} />
          <Line yAxisId="left" type="monotone" dataKey="clicks" stroke="var(--color-brand-orange)" dot={false} strokeWidth={2} connectNulls={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

// One legend entry: a short colored line segment plus the series name.
// Used in the shared legend row above the charts.
function LegendKey({ color, label }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <span
        className="inline-block h-[3px] w-3.5 rounded-full"
        style={{ backgroundColor: color }}
        aria-hidden="true"
      />
      {label}
    </span>
  );
}
