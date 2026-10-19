import { formatStaleness, getTodaysSessions, parseUtc, sumSessions } from "../utils";

// Daily summary card aggregating sessions whose start falls within
// today's local-time boundary. Reuses the existing /api/sessions
// poll - no new request.
//
// Honest active time: this sums window_count (one row per minute of
// activity in the batch job) rather than (session_end - session_start),
// because the sessionizer in batch_job.py treats gaps under 5 min as
// the same session and would otherwise inflate the figure.
//
// Multi-user note: this component receives sessions ALREADY filtered
// to the selected user (App.jsx applies filterByUser before passing
// them in), so totals are per-user. To restore cross-user totals,
// pass the unfiltered sessions array.
export function TodayTotals({ sessions }) {
  // Reuses the shared helper so the midnight boundary is defined in
  // one place (utils.js); Header's "today" chips and these totals
  // stay consistent.
  const todays = getTodaysSessions(sessions);

  const totals = sumSessions(todays);

  // Staleness is measured against the newest session_end across ALL
  // sessions, not just today's, so a user opening the dashboard
  // mid-morning sees "as of 14 hours ago" rather than nothing when
  // the batch job has not yet run today.
  const newestEnd = (sessions || []).reduce((m, s) => {
    const end = parseUtc(s.session_end);
    return end && (!m || end > m) ? end : m;
  }, null);

  const subtitle = formatStaleness(newestEnd);

  const stats = [
    { label: "Keystrokes", value: totals.keystrokes },
    { label: "Words", value: totals.words },
    { label: "Corrections", value: totals.corrections },
    { label: "Clicks", value: totals.clicks },
    { label: "Sessions", value: todays.length },
    { label: "Active", value: `${totals.activeMin} min` },
  ];

  const isEmpty = todays.length === 0;

  return (
    <div className="glass-panel">
      <div className="flex items-baseline justify-between mb-3 gap-3">
        <h2 className="text-sm text-zinc-400">
          Today <span className="text-zinc-600">(sessions started since midnight)</span>
        </h2>
        {subtitle && (
          <span className="text-xs text-zinc-500 shrink-0">{subtitle}</span>
        )}
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4">
        {stats.map((s) => (
          <div key={s.label}>
            <div className="text-xs text-zinc-500">{s.label}</div>
            <div className="text-xl font-semibold text-zinc-100 mt-1">
              {isEmpty ? "—" : s.value}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
