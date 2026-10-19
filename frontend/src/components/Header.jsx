import { useEffect, useState } from "react";
import {
  getTodaysSessions,
  isActiveNow,
  peakKsPerMinToday,
  sessionTimer,
} from "../utils";
import { UserSelector } from "./UserSelector";

// Single compact header band. Left group is identity + current
// state, right group is summary chips + live indicator.
//
// The 1-Hz ticker keeps Date.now() out of render (the
// react-hooks/purity rule rejects impure calls during render) and
// drives both the live-status freshness check and the session
// timer. Cleanup on unmount.
//
// Live indicator tri-states via shape AND color AND word:
//   live         = green dot (round)        + "live"
//   reconnecting = amber dot (round, pulse) + "reconnecting"  (backend unreachable)
//   offline      = red square               + "offline"       (data genuinely stale)
// role="status" makes the span a live region so screen readers
// announce the transition rather than needing the user to revisit.
export function Header({
  metrics,
  sessions,
  users,
  selectedUser,
  onSelectUser,
  connectionLost = false,
}) {
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, []);

  // Freshness rationale (window_end, 2-min threshold) now lives in
  // isActiveNow, shared with the day-detail live badge so they agree.
  // connectionLost (the /api/metrics poll is failing) takes priority: we
  // cannot confirm liveness while the backend is unreachable, so show a
  // distinct "reconnecting" state rather than letting the frozen metrics
  // age into a misleading "offline" while the user is still typing.
  const isLive = isActiveNow(metrics, now);
  const status = connectionLost ? "reconnecting" : isLive ? "live" : "offline";

  const sessionMs = sessionTimer(metrics, 5 * 60 * 1000, now);
  const todays = getTodaysSessions(sessions, now);
  const todayKeystrokes = todays.reduce(
    (sum, s) => sum + (s.keystrokes_total ?? 0),
    0
  );
  const longestSessionMin =
    todays.length > 0
      ? Math.max(...todays.map((s) => s.window_count ?? 0))
      : 0;
  const peakKpm = peakKsPerMinToday(metrics, now);

  return (
    <header className="flex items-center justify-between gap-4 flex-wrap">
      <div className="flex items-center gap-4 flex-wrap">
        <img
          src="/logo.png"
          alt="Keyspark"
          className="h-20 w-auto object-contain select-none drop-shadow-[0_0_24px_rgba(139,92,246,0.65)]"
        />
        <UserSelector
          users={users}
          value={selectedUser}
          onChange={onSelectUser}
        />
        <span className="text-sm text-zinc-400 tabular-nums" aria-live="off">
          {sessionMs != null ? (
            <span className="text-zinc-200">
              {formatSessionTimer(sessionMs)}
            </span>
          ) : (
            <span className="text-zinc-500">idle</span>
          )}
        </span>
        {todayKeystrokes > 0 && (
          <span className="text-sm text-zinc-400">
            <span className="text-zinc-200 font-medium">
              {formatCompact(todayKeystrokes)}
            </span>{" "}
            keys today
          </span>
        )}
      </div>
      <div className="flex items-center gap-3 text-sm flex-wrap">
        {peakKpm != null && peakKpm > 0 && (
          <Chip aria-label={`Peak keystrokes per minute over the last hour: ${peakKpm}`}>
            <span aria-hidden="true">🔥</span> peak {peakKpm} kpm (1h)
          </Chip>
        )}
        {longestSessionMin > 0 && (
          <Chip
            aria-label={`Longest session today: ${longestSessionMin} active minutes`}
          >
            longest {longestSessionMin}m today
          </Chip>
        )}
        <span
          role="status"
          className="flex items-center gap-2 text-zinc-400"
        >
          <span
            className={`inline-block w-2 h-2 transition-all ${
              status === "live"
                ? "bg-green-500 rounded-full"
                : status === "reconnecting"
                  ? "bg-amber-400 rounded-full animate-pulse"
                  : "bg-red-500 rounded-none"
            }`}
            aria-hidden="true"
          />
          <span>{status}</span>
        </span>
      </div>
    </header>
  );
}

// Small status chip used for personal-best indicators. The visible
// text always carries the same meaning as the aria-label so screen
// reader users hear an equivalent phrasing of what sighted users
// see.
function Chip({ children, ...rest }) {
  return (
    <span
      className="glass-chip text-xs text-zinc-200"
      {...rest}
    >
      {children}
    </span>
  );
}

// h:mm:ss with hours hidden when zero. tabular-nums on the parent
// keeps the timer from jittering when digits change width.
function formatSessionTimer(ms) {
  const totalSec = Math.floor(ms / 1000);
  const hours = Math.floor(totalSec / 3600);
  const min = Math.floor((totalSec % 3600) / 60);
  const sec = totalSec % 60;
  const pad = (n) => String(n).padStart(2, "0");
  if (hours > 0) return `${hours}:${pad(min)}:${pad(sec)}`;
  return `${min}:${pad(sec)}`;
}

// 1234 -> "1.2k", 12345 -> "12.3k", anything under 1000 stays as
// the raw integer. Keeps the header compact.
function formatCompact(n) {
  if (n >= 1000) return `${(n / 1000).toFixed(1)}k`;
  return String(n);
}
