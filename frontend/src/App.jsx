import { useMemo, useState } from "react";
import { Header } from "./components/Header";
import { TodayTotals } from "./components/TodayTotals";
import { ActivityPanel } from "./components/ActivityPanel";
import { MetricCards } from "./components/MetricCards";
import { InputMixIndicator } from "./components/InputMixIndicator";
import { ActivityGauge } from "./components/ActivityGauge";
import { MonthCalendar } from "./components/MonthCalendar";
import { DayDetailPanel } from "./components/DayDetailPanel";
import { usePolling } from "./usePolling";
import { filterByUser, parseUtc } from "./utils";

// Top-level wiring. Polling intervals match how often each
// upstream output changes:
// - metrics: 5 s (streaming job emits a new window every minute).
// - sessions, baseline, heatmap, batch_status: 30 s. These four
//   come from the batch job (or its scheduler state) and only
//   change when the batch job runs, so polling any faster would
//   just resend the same payload. The scheduler runs every 5 min
//   inside the API process; see keyspark/api.py.
//
// Multi-user filtering: every endpoint includes a `user` field.
// The dashboard scopes every panel to a single user. `selectedUser`
// is the explicit user choice from the dropdown; `effectiveUser` is
// the user actually rendered — either the explicit choice (if it
// is still present in the metrics stream) or the user with the
// newest metric window otherwise. Deriving `effectiveUser` during
// render avoids a setState-in-effect cascade.
export default function App() {
  const [selectedUser, setSelectedUser] = useState(null);
  // localDayKey ("YYYY-MM-DD") of the day the user is drilling into,
  // or null when no day is selected. Driven by clicks in the
  // MonthCalendar.
  const [selectedDay, setSelectedDay] = useState(null);

  // The top Activity card always shows the last 60 minutes (5 s poll);
  // there is no range selector anymore. This single /api/metrics poll
  // feeds the header, the metric cards, and the Activity card.
  const { data: metrics, failing: metricsFailing } = usePolling("/api/metrics", 5_000);
  const { data: sessions } = usePolling("/api/sessions", 30_000);
  const { data: baseline } = usePolling("/api/baseline", 5 * 60_000);
  const { data: batchStatus } = usePolling("/api/batch_status", 30_000);

  const lastBatchRun = batchStatus?.last_run ?? null;
  const batchStatusName = batchStatus?.status ?? "idle";

  // Derived user list from the metrics stream. null while metrics
  // is null so UserSelector can stay hidden until first fetch.
  const users = useMemo(() => {
    if (!metrics) return null;
    return Array.from(new Set(metrics.map((m) => m.user).filter(Boolean))).sort();
  }, [metrics]);

  // Pure derivation: honor the dropdown choice when it's still
  // valid, otherwise fall back to the newest-activity user.
  const effectiveUser = useMemo(() => {
    if (selectedUser && users && users.includes(selectedUser)) {
      return selectedUser;
    }
    if (!metrics || metrics.length === 0) return null;

    let newestUser = null;
    let newestTime = -Infinity;
    for (const m of metrics) {
      const t = parseUtc(m.window_start)?.getTime();
      if (t != null && t > newestTime) {
        newestTime = t;
        newestUser = m.user;
      }
    }
    return newestUser;
  }, [metrics, users, selectedUser]);

  const metricsForUser = filterByUser(metrics, effectiveUser);
  const sessionsForUser = filterByUser(sessions, effectiveUser);
  const baselineForUser =
    baseline && effectiveUser
      ? baseline.find((b) => b.user === effectiveUser) ?? null
      : null;

  // Per-day human/non-human flags for the calendar (red = automation
  // detected). Scoped to the rendered user; a null url skips the fetch
  // until a user is known.
  const { data: liveness } = usePolling(
    effectiveUser ? `/api/liveness?user=${effectiveUser}` : null,
    30_000,
  );
  const nonhumanDays = useMemo(
    () =>
      new Set(
        (liveness || [])
          .filter((r) => r.nonhuman && r.user === effectiveUser)
          .map((r) => r.day),
      ),
    [liveness, effectiveUser],
  );

  return (
    <div className="min-h-screen text-zinc-100">
      <div className="max-w-[1450px] mx-auto p-6 space-y-6">
        <Header
          metrics={metricsForUser}
          sessions={sessionsForUser}
          users={users}
          selectedUser={effectiveUser}
          onSelectUser={setSelectedUser}
          connectionLost={metricsFailing}
        />
        <section className="space-y-4">
          <ActivityGauge metrics={metricsForUser} range="1h" />
          <ActivityPanel metrics={metricsForUser} range="1h" />
        </section>
        <TodayTotals sessions={sessionsForUser} />
        <MetricCards sessions={sessionsForUser} baseline={baselineForUser} />
        <InputMixIndicator
          latest={
            metricsForUser && metricsForUser.length > 0
              ? metricsForUser[metricsForUser.length - 1]
              : null
          }
        />
        <section className="space-y-4">
          <MonthCalendar
            sessions={sessionsForUser}
            selectedDay={selectedDay}
            onSelectDay={setSelectedDay}
            lastRunIso={lastBatchRun}
            status={batchStatusName}
            nonhumanDays={nonhumanDays}
          />
          {selectedDay && (
            <DayDetailPanel
              sessions={sessionsForUser}
              metrics={metricsForUser}
              dayKey={selectedDay}
              user={effectiveUser}
              onClose={() => setSelectedDay(null)}
            />
          )}
        </section>
      </div>
    </div>
  );
}
