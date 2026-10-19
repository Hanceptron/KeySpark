// The backend serves timestamps in two shapes: the Spark/pandas
// endpoints write UTC with a trailing "Z" (e.g.
// "2026-05-21T13:44:00.000Z"), while batch_status.last_run is a Python
// isoformat that carries an offset (e.g. "2026-06-05T11:12:15.973451+00:00").
// A naive (tz-less) string is also possible. JavaScript parses a tz-less
// ISO string as LOCAL time, so we append "Z" to force UTC, but only when
// no timezone designator is already present. Appending to "+00:00" would
// produce an Invalid Date (which previously surfaced as "as of NaN d ago").
export function parseUtc(s) {
  if (!s) return null;
  return new Date(/(Z|[+-]\d{2}:\d{2})$/.test(s) ? s : s + "Z");
}

// Filter a list of API rows down to a single user. Preserves the
// null distinction so callers can still tell pre-fetch (null) apart
// from "no rows for this user" ([]). A null user passes the rows
// through unchanged — the brief interval between metrics arriving
// and the auto-select picking a user.
export function filterByUser(rows, user) {
  if (!rows) return null;
  if (!user) return rows;
  return rows.filter((r) => r && r.user === user);
}

// Range presets shared by the activity gauge, idle strip, and
// metrics chart. Each entry specifies how many buckets to render
// and how wide each bucket is in minutes; total minutes covered =
// bucketCount * bucketSizeMin. Bucket counts are tuned to keep the
// strip and chart readable (60-100 cells) regardless of range.
// pollMs scales with range: short windows poll fast for freshness,
// long windows poll slowly to keep payloads under control.
export const ACTIVITY_RANGES = {
  "1h": { bucketCount: 60, bucketSizeMin: 1,   label: "Last 60 minutes", pollMs: 5_000 },
  "6h": { bucketCount: 72, bucketSizeMin: 5,   label: "Last 6 hours",    pollMs: 15_000 },
  "1d": { bucketCount: 96, bucketSizeMin: 15,  label: "Last 24 hours",   pollMs: 30_000 },
  "3d": { bucketCount: 72, bucketSizeMin: 60,  label: "Last 3 days",     pollMs: 60_000 },
  "1w": { bucketCount: 84, bucketSizeMin: 120, label: "Last 7 days",     pollMs: 60_000 },
};

// A bucket counts as "active" if any keystrokes, words,
// corrections, or clicks were recorded. Shared between
// ActivityGauge and IdleStrip so the two panels stay in sync.
export function isActiveBucket(b) {
  if (!b) return false;
  return (
    (b.keystrokes ?? 0) +
      (b.words ?? 0) +
      (b.corrections ?? 0) +
      (b.clicks ?? 0) >
    0
  );
}

// Aggregate per-minute /api/metrics rows into `bucketCount` fixed-
// width buckets ending at `now`. Each bucket sums the count
// columns across the per-minute rows that fall within it; missing
// buckets become zero-valued synthetic rows so callers can render
// idle stretches as flat zero rather than connecting across gaps.
// Buckets are keyed by floor(time / bucketMs) so a per-minute row
// snaps into whichever bucket its window_start belongs to.
export function bucketizeWindows(metrics, bucketCount, bucketSizeMin, now = Date.now()) {
  const bucketMs = bucketSizeMin * 60_000;
  const nowBucket = Math.floor(now / bucketMs);
  const startBucket = nowBucket - bucketCount + 1;

  const sums = new Map();
  for (const m of metrics || []) {
    const t = parseUtc(m.window_start);
    if (!t) continue;
    const b = Math.floor(t.getTime() / bucketMs);
    if (b < startBucket || b > nowBucket) continue;
    const cur = sums.get(b);
    if (cur) {
      cur.keystrokes += m.keystrokes ?? 0;
      cur.words += m.words ?? 0;
      cur.corrections += m.corrections ?? 0;
      cur.clicks += m.clicks ?? 0;
      cur.mouse_moves += m.mouse_moves ?? 0;
    } else {
      sums.set(b, {
        keystrokes: m.keystrokes ?? 0,
        words: m.words ?? 0,
        corrections: m.corrections ?? 0,
        clicks: m.clicks ?? 0,
        mouse_moves: m.mouse_moves ?? 0,
      });
    }
  }

  const result = [];
  for (let b = startBucket; b <= nowBucket; b++) {
    const window_start = new Date(b * bucketMs).toISOString();
    const sum = sums.get(b);
    if (sum) {
      result.push({ window_start, ...sum });
    } else {
      result.push({
        window_start,
        keystrokes: 0,
        words: 0,
        corrections: 0,
        clicks: 0,
        mouse_moves: 0,
        synthetic: true,
      });
    }
  }
  return result;
}

// --- Display formatting ------------------------------------------------
// Every user-facing date/time renders in Istanbul time, 24-hour clock,
// DD.MM.YYYY. The Intl formatters are built once at module load (calling
// .format() during render is pure). en-GB gives day-first ordering and
// 24-hour times; we swap its "/" separators for ".".
const ISTANBUL_TZ = "Europe/Istanbul";

const _istDate = new Intl.DateTimeFormat("en-GB", {
  timeZone: ISTANBUL_TZ,
  day: "2-digit",
  month: "2-digit",
  year: "numeric",
});
const _istTime = new Intl.DateTimeFormat("en-GB", {
  timeZone: ISTANBUL_TZ,
  hour: "2-digit",
  minute: "2-digit",
  hour12: false,
});
const _istWeekday = new Intl.DateTimeFormat("en-GB", {
  timeZone: ISTANBUL_TZ,
  weekday: "long",
});
const _istDayMonth = new Intl.DateTimeFormat("en-GB", {
  timeZone: ISTANBUL_TZ,
  day: "2-digit",
  month: "2-digit",
});

// "DD.MM.YYYY" — Istanbul calendar day of a real instant.
export function formatDate(date) {
  if (!date) return "—";
  return _istDate.format(date).replace(/\//g, ".");
}

// "HH:mm" — Istanbul wall-clock, 24-hour.
export function formatClock(date) {
  if (!date) return "—";
  return _istTime.format(date);
}

// "DD.MM.YYYY HH:mm" — Istanbul, 24-hour. Used for session start/end.
export function formatDateTime(date) {
  if (!date) return "—";
  return `${formatDate(date)} ${formatClock(date)}`;
}

// "DD.MM.YYYY HH:mm - HH:mm" for a session's start..end span (Istanbul).
// The end shows just its time when it falls on the same calendar day;
// if the session crosses midnight the end keeps its own date so the
// span stays unambiguous. A null end (or start) degrades gracefully.
export function formatSessionRange(start, end) {
  if (!start) return "—";
  const startStr = formatDateTime(start);
  if (!end) return startStr;
  const endStr =
    formatDate(start) === formatDate(end) ? formatClock(end) : formatDateTime(end);
  return `${startStr} - ${endStr}`;
}

// "Friday, 31.05.2026" from a "YYYY-MM-DD" day key. The numeric part is
// the key reformatted; the weekday comes from a midday-UTC anchor so it
// stays on the right calendar day regardless of the viewer's timezone.
export function formatDayLabel(dayKey) {
  if (!dayKey) return "—";
  const [y, m, d] = dayKey.split("-").map(Number);
  const ddmmyyyy = `${String(d).padStart(2, "0")}.${String(m).padStart(2, "0")}.${y}`;
  const weekday = _istWeekday.format(new Date(Date.UTC(y, m - 1, d, 12)));
  return `${weekday}, ${ddmmyyyy}`;
}

// "DD.MM.YYYY" from a Date that already represents a calendar day via its
// local components (e.g. MonthCalendar cells built at local midnight) -
// read straight from those components, no timezone conversion.
export function formatLocalDay(date) {
  if (!date) return "—";
  const dd = String(date.getDate()).padStart(2, "0");
  const mm = String(date.getMonth() + 1).padStart(2, "0");
  return `${dd}.${mm}.${date.getFullYear()}`;
}

// Tick / tooltip label for a single bucket. When the total range fits in
// a day the date is implied by "today" and showing only HH:mm keeps the
// axis terse; once the range spans multiple days the date is prepended
// to disambiguate the same hour appearing repeatedly. Istanbul, 24-hour.
export function formatBucketTime(date, totalMinutes) {
  if (!date) return "";
  if (totalMinutes <= 1440) {
    return formatClock(date);
  }
  return `${_istDayMonth.format(date).replace(/\//g, ".")} ${formatClock(date)}`;
}

// Share of input that was mouse clicks vs keyboard keys for a
// single window. Returns null when both are zero — the caller
// (InputMixIndicator) treats that as "no input" rather than 0%.
export function keyboardMouseMix(window) {
  if (!window) return null;
  const k = window.keystrokes ?? 0;
  const c = window.clicks ?? 0;
  if (k + c === 0) return null;
  return c / (k + c);
}

// Milliseconds since the start of the current unbroken run of ACTIVE
// metric windows ("the current sitting"). A run ends when the gap
// between two consecutive active window_starts exceeds `gapMs` (default
// 5 min, matching the batch job's session boundary). Returns null when
// the newest active window is older than `gapMs` (the user is idle).
//
// Idle / 0-count windows are filtered out first so a pause cannot bridge
// a break: without this the timer merged across a sub-window gap and
// over-counted (e.g. 21 min when the real sitting was 4). This is the
// frontend's definition of "current session"; the batch session list
// lags by however often the batch runs.
export function sessionTimer(metrics, gapMs = 5 * 60 * 1000, now = Date.now()) {
  if (!metrics || metrics.length === 0) return null;

  const active = metrics.filter(isActiveBucket);
  if (active.length === 0) return null;

  const sorted = [...active].sort((a, b) => {
    const at = parseUtc(a.window_start);
    const bt = parseUtc(b.window_start);
    return bt.getTime() - at.getTime();
  });

  const newestTime = parseUtc(sorted[0].window_start)?.getTime();
  if (newestTime == null || now - newestTime > gapMs) return null;

  let runStartTime = newestTime;
  for (let i = 0; i < sorted.length - 1; i++) {
    const thisTime = parseUtc(sorted[i].window_start)?.getTime();
    const nextTime = parseUtc(sorted[i + 1].window_start)?.getTime();
    if (thisTime == null || nextTime == null) break;
    if (thisTime - nextTime > gapMs) break;
    runStartTime = nextTime;
  }
  return now - runStartTime;
}

// True when the freshest measured minute is recent enough that the user
// is active right now. Scans window_end (not window_start) for the max,
// matching Header's freshness rationale: a 1-min window + watermark makes
// window_start age oscillate, while window_end age stays tight. Shared by
// the header live dot and the day-detail live badge so they agree.
export function isActiveNow(metrics, now = Date.now(), maxAgeMs = 2 * 60 * 1000) {
  if (!metrics || metrics.length === 0) return false;
  let newestEnd = -Infinity;
  for (const m of metrics) {
    const t = parseUtc(m.window_end)?.getTime();
    if (t != null && t > newestEnd) newestEnd = t;
  }
  return newestEnd !== -Infinity && now - newestEnd < maxAgeMs;
}

// Filter /api/sessions down to sessions started on or after today's
// local midnight. Shared between TodayTotals and the Header's
// personal-best chips so the same "today" boundary is used in both.
export function getTodaysSessions(sessions, now = Date.now()) {
  if (!sessions) return [];
  const midnight = new Date(now);
  midnight.setHours(0, 0, 0, 0);
  return sessions.filter((s) => {
    const start = parseUtc(s.session_start);
    return start && start >= midnight;
  });
}

// Sum a set of sessions into input totals plus active minutes (one
// window_count = one active minute). Shared by the Today panel and the
// per-minute cards so both read off identical figures.
export function sumSessions(sessions) {
  return (sessions || []).reduce(
    (acc, s) => {
      acc.keystrokes += s.keystrokes_total ?? 0;
      acc.words += s.words_total ?? 0;
      acc.corrections += s.corrections_total ?? 0;
      acc.clicks += s.clicks_total ?? 0;
      acc.activeMin += s.window_count ?? 0;
      return acc;
    },
    { keystrokes: 0, words: 0, corrections: 0, clicks: 0, activeMin: 0 },
  );
}

// Render a timestamp as a relative-time string like "as of 3 min
// ago", "as of 2 h ago", "as of 4 d ago". Appends "(batch job
// pending)" once the gap exceeds 30 minutes — used both for the
// data-staleness chip in TodayTotals (newest session_end) and the
// compute-staleness chips elsewhere (last batch_status.last_run).
// Returns null when end is missing so the caller can skip rendering.
export function formatStaleness(end, now = Date.now()) {
  if (!end) return null;
  const ageMs = now - end.getTime();
  const ageMin = Math.round(ageMs / 60000);

  let text;
  if (ageMin < 1) text = "as of just now";
  else if (ageMin < 60) text = `as of ${ageMin} min ago`;
  else if (ageMin < 60 * 24) text = `as of ${Math.round(ageMin / 60)} h ago`;
  else text = `as of ${Math.round(ageMin / (60 * 24))} d ago`;

  if (ageMin > 30) text += " (batch job pending)";
  return text;
}

// "YYYY-MM-DD" from a Date using local-timezone components. Used as
// the stable key for grouping sessions by day in the History
// section. UTC would split a typing session that crosses local
// midnight into two buckets the user doesn't think of as separate.
export function localDayKey(date) {
  if (!date) return null;
  const y = date.getFullYear();
  const m = String(date.getMonth() + 1).padStart(2, "0");
  const d = String(date.getDate()).padStart(2, "0");
  return `${y}-${m}-${d}`;
}

// Group sessions by their local-day key. Returns a Map so callers
// can iterate in insertion order or look up O(1). Sessions with no
// session_start (shouldn't happen but defensive) are dropped.
export function groupSessionsByDay(sessions) {
  const byDay = new Map();
  for (const s of sessions || []) {
    const start = parseUtc(s?.session_start);
    if (!start) continue;
    const key = localDayKey(start);
    if (!key) continue;
    const list = byDay.get(key);
    if (list) list.push(s);
    else byDay.set(key, [s]);
  }
  return byDay;
}

// Millisecond timestamp at the end of a local calendar day given a
// "YYYY-MM-DD" key. Used as the bucketizeWindows anchor (its `now`
// argument) so a historical day's graph spans exactly that day:
// bucketizeWindows(dayMetrics, 96, 15, endOfLocalDayMs(key)) yields
// 96 fifteen-minute buckets covering [start-of-day, end-of-day].
export function endOfLocalDayMs(dayKey) {
  if (!dayKey) return Date.now();
  const [y, m, d] = dayKey.split("-").map(Number);
  // Day d+1 at 00:00 local is the exclusive end of day d.
  return new Date(y, m - 1, d + 1).getTime();
}

// Activity tiers for the History grid, in ascending intensity.
// Exported so MonthCalendar's order/legend and DayDetailPanel stay in
// lockstep with the binning in buildActivityRatings.
export const ACTIVITY_TIERS = [
  "not_active",
  "below_average",
  "standard",
  "productive",
];

// Build per-day ACTIVITY ratings for the History calendar.
//
// A day's "active minutes" = sum of its sessions' window_count (each
// window is one minute that had real keyboard/mouse activity). Tiers
// are RELATIVE to the user's typical active day: take the median
// active-minutes over days with >0 activity, then bin each day by its
// ratio to that median. Multiplicative bands - these adapt to the
// user and do NOT force fixed proportions the way quartiles would:
//   not_active     : 0 active minutes
//   below_average  : > 0 and < 0.5 * median
//   standard       : 0.5 * median .. 1.5 * median (inclusive)
//   productive     : > 1.5 * median
//
// Edge cases: 0 active days -> median null -> every day not_active;
// 1 active day -> ratio 1.0 -> standard; all-equal -> all standard.
// Returns { byDay: Map<dayKey, {tier, totalMin}>, median }.
export function buildActivityRatings(sessions) {
  const byDaySessions = groupSessionsByDay(sessions);

  // Pass 1: total active minutes per day.
  const totals = new Map();
  for (const [key, list] of byDaySessions) {
    let sum = 0;
    for (const s of list) sum += s.window_count ?? 0;
    totals.set(key, sum);
  }

  // Median over days that had any activity.
  const active = [...totals.values()].filter((v) => v > 0).sort((a, b) => a - b);
  let median = null;
  if (active.length) {
    const mid = Math.floor(active.length / 2);
    median =
      active.length % 2 ? active[mid] : (active[mid - 1] + active[mid]) / 2;
  }

  // Pass 2: bin each day against the median.
  const byDay = new Map();
  for (const [key, totalMin] of totals) {
    byDay.set(key, { tier: tierFor(totalMin, median), totalMin });
  }
  return { byDay, median };
}

// Classify one day's active-minutes against the median of active days.
function tierFor(totalMin, median) {
  if (totalMin <= 0 || median == null || median <= 0) return "not_active";
  const ratio = totalMin / median;
  if (ratio < 0.5) return "below_average";
  if (ratio <= 1.5) return "standard";
  return "productive";
}

// Peak keystrokes-per-minute observed today across the supplied
// metrics slice. NOTE: /api/metrics returns only the last 60 min,
// so this is in practice "peak over the last hour" — callers should
// label accordingly. Returns null when nothing has been recorded
// today.
export function peakKsPerMinToday(metrics, now = Date.now()) {
  if (!metrics || metrics.length === 0) return null;
  const midnight = new Date(now);
  midnight.setHours(0, 0, 0, 0);
  const midnightMs = midnight.getTime();

  let peak = 0;
  for (const m of metrics) {
    const t = parseUtc(m.window_start);
    if (!t || t.getTime() < midnightMs) continue;
    if ((m.keystrokes ?? 0) > peak) peak = m.keystrokes;
  }
  return peak > 0 ? peak : null;
}
