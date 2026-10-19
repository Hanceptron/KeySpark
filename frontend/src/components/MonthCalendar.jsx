import { useEffect, useMemo, useState } from "react";
import { buildActivityRatings, formatLocalDay, localDayKey } from "../utils";
import { StalenessChip } from "./StalenessChip";

// 8 weeks x 7 days (Mon-Sun) GitHub-contributions style grid. Each
// cell is one day, colored by how much ACTIVITY happened that day
// (active minutes = summed window_count).
// Tiers come from buildActivityRatings(), which bins each day against
// the median of the user's active days. One green hue at rising
// intensity answers "which days did I work, and how hard."
//
// Computation is client-side off the sessions prop. No new endpoint.
// A 60-second ticker keeps the "today" pointer up to date so the
// calendar advances even on a long-open browser tab.
const WEEKS = 12;
const DAYS_IN_WEEK = 7;
const DAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

// Single-color (green) intensity ramp, brighter = more active. The
// hue is constant; only opacity rises, so it reads as one scale.
// not_active reuses the muted recessed cell. `number` is the date
// color that stays legible on each fill.
const RATING_META = {
  not_active: {
    // Borderless faint fill (no glass-inset border) so inactive cells
    // read as a soft contribution heatmap separated by the grid gap,
    // not a bordered spreadsheet. Color does the work.
    bg: "bg-[rgba(24,16,42,0.55)]",
    text: "Not active",
    number: "text-zinc-500",
  },
  below_average: {
    bg: "bg-green-400/30",
    text: "Below average",
    number: "text-zinc-200",
  },
  standard: {
    bg: "bg-green-400/60",
    text: "Standard",
    number: "text-zinc-100",
  },
  productive: {
    bg: "bg-green-400",
    text: "High-output",
    number: "text-zinc-900",
  },
};
const RATING_ORDER = ["not_active", "below_average", "standard", "productive"];

export function MonthCalendar({
  sessions,
  selectedDay,
  onSelectDay,
  lastRunIso,
  status,
  nonhumanDays,
}) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 60_000);
    return () => clearInterval(id);
  }, []);

  const { byDay } = useMemo(() => buildActivityRatings(sessions), [sessions]);

  const cells = useMemo(() => {
    const today = new Date(now);
    today.setHours(0, 0, 0, 0);
    // Distance from today to the most recent Monday. JS getDay():
    // Sunday=0 .. Saturday=6. (day+6)%7 maps Mon=0, Sun=6.
    const distToMonday = (today.getDay() + 6) % 7;
    const thisMonday = new Date(today);
    thisMonday.setDate(today.getDate() - distToMonday);

    const out = [];
    // Column-first ordering matches `grid-flow-col`: filling row 0..6
    // before advancing the column. Column 0 is the earliest week,
    // column WEEKS-1 is the current week.
    for (let weekIdx = 0; weekIdx < WEEKS; weekIdx++) {
      const weekStart = new Date(thisMonday);
      weekStart.setDate(thisMonday.getDate() - (WEEKS - 1 - weekIdx) * 7);
      for (let dow = 0; dow < DAYS_IN_WEEK; dow++) {
        const cellDate = new Date(weekStart);
        cellDate.setDate(weekStart.getDate() + dow);
        const key = localDayKey(cellDate);
        const info = byDay.get(key);
        out.push({
          key,
          date: cellDate,
          tier: info ? info.tier : "not_active",
          totalMin: info ? info.totalMin : 0,
          nonhuman: nonhumanDays ? nonhumanDays.has(key) : false,
          isFuture: cellDate.getTime() > today.getTime(),
        });
      }
    }
    return out;
  }, [byDay, now, nonhumanDays]);

  return (
    <div className="glass-panel">
      <div className="flex items-baseline justify-between mb-3 gap-3 flex-wrap">
        <div className="flex items-baseline gap-3">
          <h2 className="text-sm text-zinc-400">
            History (last {WEEKS} weeks)
          </h2>
          <StalenessChip lastRunIso={lastRunIso} status={status} />
        </div>
        <Legend />
      </div>
      <div className="flex gap-2">
        <div className="grid grid-rows-7 gap-1 text-[10px] text-zinc-500 pr-1">
          {DAY_LABELS.map((d) => (
            <div key={d} className="h-7 flex items-center">
              {d}
            </div>
          ))}
        </div>
        <div className="grid grid-rows-7 grid-flow-col gap-1 flex-1">
          {cells.map((cell) => (
            <DayCell
              key={cell.key}
              cell={cell}
              isSelected={cell.key === selectedDay}
              onClick={() => onSelectDay(cell.key)}
            />
          ))}
        </div>
      </div>
    </div>
  );
}

// Single calendar cell. Coloring rules:
//  - active day -> the tier's green fill, intensity scaling with
//    active minutes; date number from RATING_META.number
//  - not active -> the muted glass-inset cell, zinc-500 date
//  - future day -> bg-zinc-900 + 30 % opacity, click disabled
// The ring shows selection. aria-label and the native title carry the
// full readable summary so keyboard / screen-reader / hover users all
// get the same information.
function DayCell({ cell, isSelected, onClick }) {
  const meta = RATING_META[cell.tier];
  // A day with detected automation is painted red, overriding the green
  // activity shade. Future days stay muted.
  const bg = cell.isFuture
    ? "bg-zinc-900 opacity-30"
    : cell.nonhuman
      ? "bg-red-500/70"
      : meta.bg;
  const numberColor = !cell.isFuture && cell.nonhuman ? "text-white" : meta.number;
  const ring = isSelected ? "ring-2 ring-brand-cyan" : "";

  const summary = cell.isFuture
    ? `${formatLocalDay(cell.date)}: not yet`
    : `${formatLocalDay(cell.date)}: ${meta.text}, ${cell.totalMin} active minute${cell.totalMin === 1 ? "" : "s"}${cell.nonhuman ? " · automation detected" : ""}`;

  return (
    <button
      type="button"
      onClick={onClick}
      disabled={cell.isFuture}
      className={`h-7 rounded text-[10px] tabular-nums leading-none flex items-center justify-center transition-all ${bg} ${ring} ${
        cell.isFuture
          ? "cursor-not-allowed"
          : "cursor-pointer hover:ring-2 hover:ring-brand-violet/70"
      }`}
      aria-label={summary}
      title={summary}
    >
      <span className={`${numberColor} font-medium`}>
        {cell.date.getDate()}
      </span>
    </button>
  );
}

// Legend doubles as the intensity key: the four tiers in ascending
// order read left-to-right as less -> more active, each labeled with
// the name the user chose.
function Legend() {
  return (
    <div className="flex items-center gap-3 text-[10px] text-zinc-400 flex-wrap">
      <span className="text-zinc-500">Less</span>
      {RATING_ORDER.map((tier) => (
        <div key={tier} className="flex items-center gap-1">
          <span
            className={`inline-block w-2.5 h-2.5 rounded ${RATING_META[tier].bg}`}
            aria-hidden="true"
          />
          {RATING_META[tier].text}
        </div>
      ))}
      <span className="text-zinc-500">More</span>
      <span className="flex items-center gap-1 ml-2">
        <span
          className="inline-block w-2.5 h-2.5 rounded bg-red-500/70"
          aria-hidden="true"
        />
        Automation
      </span>
    </div>
  );
}
