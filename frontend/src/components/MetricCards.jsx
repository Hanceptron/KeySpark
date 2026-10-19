import { getTodaysSessions, sumSessions } from "../utils";

// Five cards summarising TODAY at a glance. The four input metrics show
// today's AVERAGE per active minute (today's total / today's active
// minutes); the last card shows today's correction ratio. Everything is
// computed from today's sessions - the same source as the Today panel -
// so the numbers stay stable through the day rather than jumping with
// the last single window. Each card keeps the user's all-time per-minute
// "baseline avg" (from /api/baseline) underneath so today reads against
// the norm.
//
// `sessions` arrives already filtered to the selected user (App applies
// filterByUser). `baseline` is that user's baseline row, or null until
// /api/baseline returns.
export function MetricCards({ sessions, baseline }) {
  const today = sumSessions(getTodaysSessions(sessions));
  const perMin = (total) =>
    today.activeMin > 0 ? Math.round(total / today.activeMin) : null;
  const b = baseline ?? null;

  const cards = [
    {
      label: "Keystrokes per minute",
      value: perMin(today.keystrokes),
      meanKey: "keystrokes_mean",
    },
    {
      label: "Words per minute",
      value: perMin(today.words),
      meanKey: "words_mean",
    },
    {
      label: "Corrections per minute",
      value: perMin(today.corrections),
      meanKey: "corrections_mean",
    },
    {
      label: "Clicks per minute",
      value: perMin(today.clicks),
      meanKey: "clicks_mean",
    },
    {
      label: "Correction ratio",
      value:
        today.keystrokes > 0
          ? `${((today.corrections / today.keystrokes) * 100).toFixed(1)}%`
          : null,
      // The ratio's baseline is derived from the two mean columns, not a
      // single baseline field, so it carries its own line.
      baselineLine:
        b && b.corrections_mean != null && b.keystrokes_mean
          ? `baseline: ${((b.corrections_mean / Math.max(b.keystrokes_mean, 1)) * 100).toFixed(1)}%`
          : null,
    },
  ];

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
      {cards.map((c) => {
        let baselineLine = null;
        if (c.baselineLine !== undefined) {
          baselineLine = c.baselineLine;
        } else if (c.meanKey && b && b[c.meanKey] != null) {
          baselineLine = `baseline avg: ${b[c.meanKey].toFixed(1)}`;
        }

        return (
          <div key={c.label} className="glass-panel">
            <div className="text-sm text-zinc-400">{c.label}</div>
            <div className="text-3xl font-semibold text-zinc-100 mt-1">
              {c.value ?? "—"}
            </div>
            {baselineLine && (
              <div className="text-xs text-zinc-500 mt-2">{baselineLine}</div>
            )}
          </div>
        );
      })}
    </div>
  );
}
