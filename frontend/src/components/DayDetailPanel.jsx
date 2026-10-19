import { useEffect, useMemo, useState } from "react";
import {
  buildActivityRatings,
  endOfLocalDayMs,
  formatDayLabel,
  groupSessionsByDay,
  isActiveNow,
  localDayKey,
  parseUtc,
  sessionTimer,
} from "../utils";
import { usePolling } from "../usePolling";
import { SessionRow, SessionTableHeader } from "./SessionTable";
import { ActivityGauge } from "./ActivityGauge";
import { ActivityPanel } from "./ActivityPanel";
import { Heatmap } from "./Heatmap";

// Drill-down panel for a single calendar day. Rendered conditionally
// by App.jsx when the user clicks a cell in MonthCalendar. Shows that
// day's keystroke timeline (same style as the top live graph, but
// anchored to the day instead of "now"), the movement + click
// heatmaps, and the day's sessions table.
//
// The graph and heatmaps come from batch outputs sourced from the
// event archive (/api/day_metrics, /api/heatmap_day), so historical
// days work and mouse data is available without touching the live
// streaming path. Both are keyed by dayKey+user, so selecting a
// different day refetches.
// Day-level ACTIVITY tier shown in the header (text-color variant of
// the MonthCalendar intensity ramp). Distinct from the per-session
// totals shown in the rows below.
const RATING_META = {
  not_active: { text: "Not active", color: "text-zinc-500" },
  below_average: { text: "Below average", color: "text-green-300/70" },
  standard: { text: "Standard", color: "text-green-300" },
  productive: { text: "High-output", color: "text-green-400" },
};

export function DayDetailPanel({ sessions, metrics, dayKey, user, onClose }) {
  // 10 s ticker so the live badge and its counting-up duration update
  // smoothly and clear shortly after activity stops, even between data
  // refreshes. Lazy init plus the interval callback keep Date.now() out
  // of the render body.
  const [nowMs, setNowMs] = useState(() => Date.now());
  useEffect(() => {
    const id = setInterval(() => setNowMs(Date.now()), 10_000);
    return () => clearInterval(id);
  }, []);

  // "Live" is bound to the same current sitting the header times: the
  // user is active now (fresh /api/metrics) AND today's most-recent
  // session starts at the current sitting's start. The start-match stops
  // a stale prior session from lighting up before the batch emits the
  // new one. Driven by the live stream, not the lagged session_end.
  const activeNow = isActiveNow(metrics, nowMs);
  const todayKey = localDayKey(new Date(nowMs));
  const sittingMs = sessionTimer(metrics, 5 * 60 * 1000, nowMs);
  const sittingStartMs = sittingMs != null ? nowMs - sittingMs : null;

  const byDay = useMemo(() => groupSessionsByDay(sessions), [sessions]);
  const daySessions = byDay.get(dayKey) ?? [];
  // Newest first, so the current / most recent session sits at the top
  // where the eye looks for "what am I doing now". Historical days then
  // read latest-to-earliest, an acceptable trade.
  const sorted = [...daySessions].sort((a, b) => {
    const ta = parseUtc(a.session_start)?.getTime() ?? 0;
    const tb = parseUtc(b.session_start)?.getTime() ?? 0;
    return tb - ta;
  });
  // Day-level activity tier, from the same builder the calendar uses
  // (so the header word matches the cell the user just clicked).
  const { byDay: activityByDay } = useMemo(
    () => buildActivityRatings(sessions),
    [sessions],
  );
  const dayActivity = activityByDay.get(dayKey);
  const tier = dayActivity ? dayActivity.tier : "not_active";
  const totalMin = dayActivity ? dayActivity.totalMin : 0;
  const dateLabel = formatDayLabel(dayKey);
  const meta = RATING_META[tier];

  // Per-day timeline + heatmap from the batch outputs. user is always
  // set when the panel renders (App passes effectiveUser), but guard
  // the URL anyway so a transient null doesn't fetch "/...user=null".
  const { data: dayMetrics } = usePolling(
    user ? `/api/day_metrics?day=${dayKey}&user=${user}` : null,
    30_000,
  );
  const { data: dayHeatmap } = usePolling(
    user ? `/api/heatmap_day?day=${dayKey}&user=${user}` : null,
    30_000,
  );
  // Primary-screen grid bounds so the heatmaps frame to the real
  // screen and clip any external-monitor tail. Rarely changes, so a
  // slow poll is plenty. Falls back to MacBook 16" defaults until the
  // first response lands.
  const { data: display } = usePolling("/api/display", 5 * 60_000);
  const frameW = display?.grid_w ?? 108;
  const frameH = display?.grid_h ?? 70;

  const anchorMs = endOfLocalDayMs(dayKey);

  return (
    <div className="glass-panel">
      <div className="flex items-baseline justify-between mb-3 gap-3 flex-wrap">
        <div className="flex items-baseline gap-3 flex-wrap">
          <h2 className="text-sm text-zinc-300">{dateLabel}</h2>
          <span className={`text-sm font-medium ${meta.color}`}>
            {meta.text}
          </span>
          <span className="text-xs text-zinc-500">
            {sorted.length} session{sorted.length === 1 ? "" : "s"} ·{" "}
            {totalMin} active minute{totalMin === 1 ? "" : "s"}
          </span>
        </div>
        <button
          type="button"
          onClick={onClose}
          className="text-zinc-500 hover:text-brand-cyan text-xl leading-none px-2"
          aria-label="Close day detail"
          title="Close"
        >
          ×
        </button>
      </div>

      <div className="space-y-4">
        <ActivityGauge
          metrics={dayMetrics}
          range="1d"
          anchorMs={anchorMs}
          label={dateLabel}
        />
        <ActivityPanel metrics={dayMetrics} range="1d" anchorMs={anchorMs} showMouse />
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <Heatmap
            data={dayHeatmap}
            type="move"
            title="Movement"
            frameW={frameW}
            frameH={frameH}
          />
          <Heatmap
            data={dayHeatmap}
            type="click"
            title="Clicks"
            frameW={frameW}
            frameH={frameH}
          />
        </div>

        {sorted.length === 0 ? (
          <div className="text-sm text-zinc-500 py-2">
            no sessions recorded on this date
          </div>
        ) : (
          <div className="max-h-96 overflow-y-auto">
            <SessionTableHeader />
            {sorted.map((s, i) => {
              const startMs = parseUtc(s.session_start)?.getTime();
              const live =
                activeNow &&
                dayKey === todayKey &&
                i === 0 &&
                sittingStartMs != null &&
                startMs != null &&
                Math.abs(startMs - sittingStartMs) < 90_000;
              return (
                <SessionRow key={s.session_id} s={s} live={live} now={nowMs} />
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}

