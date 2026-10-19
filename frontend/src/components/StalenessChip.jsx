import { useEffect, useState } from "react";
import { formatStaleness, parseUtc } from "../utils";

// "as of N min ago" chip that re-renders every 30 s so the relative
// time stays current between /api/batch_status polls. Hidden when
// no last_run timestamp has been received yet (the first render
// before the first poll resolves). Turns red and appends a "(last
// refresh failed)" note when the most recent batch attempt errored
// — the timestamp still moves forward, but the user knows the data
// underneath isn't from a clean run.
export function StalenessChip({ lastRunIso, status }) {
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 30_000);
    return () => clearInterval(id);
  }, []);

  if (!lastRunIso) return null;

  const lastRun = parseUtc(lastRunIso);
  const text = formatStaleness(lastRun, now);
  if (!text) return null;

  const failed = status === "failed";
  const colorClass = failed ? "text-red-400" : "text-zinc-500";
  const suffix = failed ? " (last refresh failed)" : "";

  return (
    <span
      className={`text-xs shrink-0 ${colorClass}`}
      aria-label={failed ? `batch refresh failed; ${text}` : text}
    >
      {text}
      {suffix}
    </span>
  );
}
