import { formatClock, formatSessionRange, parseUtc } from "../utils";

// Shared session-table primitives: the three-column header and one row,
// reused by the day drill-down (DayDetailPanel). Extracted from the
// former SessionsList so the table outlives the removed "Recent
// sessions" panel.

// Column headers for the three-column session table. Exported so the
// DayDetailPanel can reuse the exact same layout for its scoped view
// of a single day's sessions.
export function SessionTableHeader() {
  return (
    <div className="grid grid-cols-3 gap-4 text-xs text-zinc-500 pb-2 border-b border-white/10">
      <div>Session</div>
      <div>Duration</div>
      <div>Keystrokes</div>
    </div>
  );
}

// One session row. Exported alongside SessionTableHeader so the
// DayDetailPanel renders rows identical across views. start/end can be
// null if the backend ever emits a malformed timestamp; the guards
// keep the row from rendering "NaN min".
export function SessionRow({ s, live = false, now = 0 }) {
  const start = parseUtc(s.session_start);
  const end = parseUtc(s.session_end);
  // A live session has no real end yet, so its duration counts up from
  // the start to now (floored, to match the header timer's minute);
  // finished sessions use their recorded span.
  const durationMin = live
    ? start
      ? Math.max(1, Math.floor((now - start.getTime()) / 60000))
      : null
    : start && end
      ? Math.max(1, Math.round((end - start) / 60000))
      : null;

  return (
    <div className="grid grid-cols-3 gap-4 text-sm py-2 border-b border-white/5 text-zinc-200 items-center">
      <div className="flex items-center gap-2">
        {live ? (
          <>
            <span>Started {formatClock(start)}</span>
            <LiveBadge />
          </>
        ) : (
          <span>{formatSessionRange(start, end)}</span>
        )}
      </div>
      <div>{durationMin != null ? `${durationMin} min` : "-"}</div>
      <div>{s.keystrokes_total}</div>
    </div>
  );
}

// Pulsing marker for the session still in progress (its last activity is
// within DayDetailPanel's live window). The aria-label carries the
// meaning so the green dot is not the only signal.
function LiveBadge() {
  return (
    <span
      className="inline-flex items-center gap-1 text-[10px] font-medium uppercase tracking-wide text-green-400"
      aria-label="live, in progress"
    >
      <span
        className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse"
        aria-hidden="true"
      />
      live
    </span>
  );
}
